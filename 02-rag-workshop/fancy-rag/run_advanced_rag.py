import os
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.documents import Document

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Vector store configuration
ELASTIC_SEARCH_INDEX = "rag-workshop"
RETRIEVAL_K = 3  # Number of documents to retrieve per sub-question
MAX_FINAL_DOCS = 8  # Maximum documents in final context after deduplication

# Elasticsearch connection settings
ELASTICSEARCH_CONFIG = {
    "es_url": "https://localhost:9200/",
    "es_user": "elastic", 
    "es_password": "xM-_eV5Zqk3yjNqO*ld6",
    "es_params": {"verify_certs": False}  # For development only
}

# Azure OpenAI model configurations
LLM_CONFIG = {
    "azure_deployment": "gpt-4.1-mini",
    "model": "gpt-4.1-mini", 
    "temperature": 0.0  # Deterministic responses for consistency
}

EMBEDDING_CONFIG = {
    "azure_deployment": "embed-new"
}


# ============================================================================
# ENVIRONMENT SETUP FUNCTIONS
# ============================================================================

def setup_environment() -> None:
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://evozon-bots-new.openai.azure.com/'
    os.environ['AZURE_OPENAI_API_KEY'] = 'placeholder'
    os.environ['OPENAI_API_VERSION'] = '2024-12-01-preview'


# ============================================================================
# MODEL CREATION FUNCTIONS
# ============================================================================

def create_embeddings_model() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(**EMBEDDING_CONFIG)


def create_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(**LLM_CONFIG)


# ============================================================================
# VECTOR STORE CONNECTION
# ============================================================================

def create_vector_store(embeddings: AzureOpenAIEmbeddings) -> ElasticsearchStore:
    print(f"📡 Connecting to Elasticsearch vector store: {ELASTIC_SEARCH_INDEX}")
    
    vector_store = ElasticsearchStore(
        es_url=ELASTICSEARCH_CONFIG["es_url"],
        index_name=ELASTIC_SEARCH_INDEX,
        embedding=embeddings,
        es_user=ELASTICSEARCH_CONFIG["es_user"],
        es_password=ELASTICSEARCH_CONFIG["es_password"],
        es_params=ELASTICSEARCH_CONFIG["es_params"]
    )
    
    return vector_store


# ============================================================================
# DOCUMENT PROCESSING UTILITIES
# ============================================================================

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# QUERY DECOMPOSITION - ADVANCED TECHNIQUE #1
# ============================================================================

def create_query_decomposition_prompt() -> PromptTemplate:
    decompose_template = """Break down the following question into 3-5 clear and simple sub-questions that would help gather relevant information to answer the main question comprehensively knowing the main question is related to a Curse of Strahd D&D campaign and what happened during that campaign.
Think step-by-step about what information would be needed to provide a complete answer.

Main question: {question}

Generate 3-5 sub-questions, each on a new line, prefixed with "Q: ".
Do not number the questions, just prefix with Q:.

Sub-questions:"""
    
    return PromptTemplate.from_template(decompose_template)


def create_query_decomposition_chain(llm) -> object:
    print("🧩 Creating query decomposition chain...")
    
    decompose_prompt = create_query_decomposition_prompt()
    
    # Simple chain: prompt -> LLM -> text output
    decompose_chain = decompose_prompt | llm | StrOutputParser()
    
    return decompose_chain


def parse_sub_questions(decompose_output: str) -> List[str]:
    lines = decompose_output.strip().split('\n')
    questions = [line[3:].strip() for line in lines if line.startswith('Q:')]
    return questions


# ============================================================================
# QUERY ENHANCEMENT - ADVANCED TECHNIQUE #2
# ============================================================================

def create_query_rewrite_prompt() -> PromptTemplate:
    rewrite_template = """Rewrite the provided user query to be optimized for RAG retrieval knowing the user query is related to a Curse of Strahd campaign and what happened during the campaign.
Make sure it's detailed enough to facilitate retrieval of data from the vector store.

User query: {question}

Rewritten user query:"""
    
    return PromptTemplate.from_template(rewrite_template)


def create_query_rewrite_chain(llm) -> object:
 
    print("✍️ Creating query rewrite chain...")
    
    rewrite_prompt = create_query_rewrite_prompt()
    
    # Simple chain: prompt -> LLM -> text output
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    return rewrite_chain


# ============================================================================
# MULTI-QUERY RETRIEVAL - ADVANCED TECHNIQUE #3
# ============================================================================

def multi_query_retrieval(query: str, retriever, decompose_chain, rewrite_chain) -> List[Document]:
    """
    Perform advanced document retrieval with query decomposition and rewriting.
    
    Args:
        query: Original complex user question
        retriever: Basic document retriever
        decompose_chain: Query decomposition chain
        rewrite_chain: Query rewriting chain
        
    Returns:
        List of relevant documents from multiple optimized searches
    """
    print(f"🎯 Processing complex query: {query}")
    
    # Step 1: Decompose the main query into sub-questions
    print("🧩 Decomposing query into sub-questions...")
    decomposed_output = decompose_chain.invoke({"question": query})
    sub_questions = parse_sub_questions(decomposed_output)
    
    print(f"📝 Generated {len(sub_questions)} sub-questions:")
    for i, sub_q in enumerate(sub_questions, 1):
        print(f"   {i}. {sub_q}")
    
    # Step 2: Retrieve documents for each sub-question
    all_docs = []
    for i, sub_q in enumerate(sub_questions, 1):
        print(f"\n🔍 Processing sub-question {i}: {sub_q}")
        
        # Step 2a: Rewrite the sub-question for better retrieval
        rewritten_sub_q = rewrite_chain.invoke({"question": sub_q})
        print(f"   ✨ Rewritten: {rewritten_sub_q}")
        
        # Step 2b: Retrieve documents using the optimized sub-question
        docs = retriever.get_relevant_documents(rewritten_sub_q)
        print(f"   📚 Retrieved {len(docs)} documents")
        
        all_docs.extend(docs)
    
    # Step 3: Deduplicate documents based on content
    print(f"\n🔄 Deduplicating {len(all_docs)} total documents...")
    unique_docs = []
    unique_contents = set()
    
    for doc in all_docs:
        # Use the first 100 characters as a simple deduplication key
        # In production, you might use more sophisticated deduplication
        content_key = doc.page_content[:100]
        if content_key not in unique_contents:
            unique_contents.add(content_key)
            unique_docs.append(doc)
    
    print(f"📊 After deduplication: {len(unique_docs)} unique documents")
    
    # Step 4: Limit to maximum documents to stay within context limits
    final_docs = unique_docs[:MAX_FINAL_DOCS] if len(unique_docs) > MAX_FINAL_DOCS else unique_docs
    
    if len(final_docs) < len(unique_docs):
        print(f"⚡ Limited to top {len(final_docs)} documents for optimal performance")
    
    print(f"✅ Returning {len(final_docs)} documents for answer generation")
    return final_docs


# ============================================================================
# RAG CHAIN CONSTRUCTION
# ============================================================================

def create_rag_prompt() -> PromptTemplate:
    template = """You are Dungeon Master's assistant for the Curse of Strahd campaign.
The context you will receive will be pieces of information taken from the notes taken after every session.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Include all relevant information but limit the response to 5 sentences.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

    return PromptTemplate.from_template(template)


def create_advanced_rag_chain(retriever, decompose_chain, rewrite_chain, llm) -> object:
    """
    Build the complete Advanced RAG processing chain.
    
    This function creates a LangChain pipeline that:
    1. Takes a complex user question
    2. Uses multi-query retrieval (with decomposition + rewriting) to find relevant documents
    3. Formats the comprehensive document set as context
    4. Applies the RAG prompt template
    5. Sends everything to the LLM for answer generation
    6. Parses the response to return clean text
    
    Args:
        retriever: Basic document retriever
        decompose_chain: Query decomposition chain
        rewrite_chain: Query rewriting chain
        llm: Language model for text generation
        
    Returns:
        Runnable chain that can process complex questions and return comprehensive answers
    """
    print("🔗 Building Advanced RAG processing chain...")
    
    # Create the RAG prompt template
    rag_prompt = create_rag_prompt()
    
    # Build the complete processing chain with multi-query retrieval
    advanced_rag_chain = (
        {
            # Use advanced multi-query retrieval, then format documents
            "context": lambda query: format_docs(multi_query_retrieval(query, retriever, decompose_chain, rewrite_chain)),
            # Pass the original question through unchanged
            "question": RunnablePassthrough()
        }
        | rag_prompt          # Apply the prompt template
        | llm                 # Generate response with LLM
        | StrOutputParser()   # Extract text from response object
    )
    
    return advanced_rag_chain


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_advanced_rag_system(advanced_rag_chain, test_questions: List[str]) -> None:
    """
    Test the Advanced RAG system.
    
    Args:
        advanced_rag_chain: The advanced RAG processing chain
        test_questions: List of complex questions to test
    """
    print(f"\n🧪 Testing Advanced RAG system with {len(test_questions)} questions...")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Complex Question {i}: {question}")
        print("-" * 50)
        
        try:
            # Test Advanced RAG with query decomposition + rewriting
            response = advanced_rag_chain.invoke(question)
            print(f"🤖 Answer: {response}")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main() -> int:
    """
    Main function that orchestrates the Advanced RAG system execution.
    
    This function demonstrates the most sophisticated RAG workflow:
    1. Environment setup and model initialization
    2. Vector store connection
    3. Query decomposition and rewriting chain creation
    4. Advanced RAG chain construction
    5. System testing
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print("🎓 Advanced RAG System with Query Decomposition Starting...")
    print("=" * 70)

    try:
        # ====================================================================
        # STEP 1: Environment and Model Setup
        # ====================================================================
        print("\n📋 Step 1: Initializing components...")
        
        # Configure Azure OpenAI environment
        setup_environment()
        print("  ✅ Environment configured")

        # Create embeddings model for document similarity search
        embeddings = create_embeddings_model()
        print("  ✅ Embeddings model created")
        
        # Create language model for all text generation tasks
        llm = create_llm()
        print("  ✅ Language model created")

        # ====================================================================
        # STEP 2: Vector Store Connection
        # ====================================================================
        print("\n📋 Step 2: Connecting to vector store...")
        
        # Connect to the pre-loaded Elasticsearch vector store
        vector_store = create_vector_store(embeddings)
        print("  ✅ Connected to vector store")

        # Create basic retriever for sub-question processing
        basic_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        print(f"  ✅ Document retriever created (k={RETRIEVAL_K})")

        # ====================================================================
        # STEP 3: Advanced Query Processing System
        # ====================================================================
        print("\n📋 Step 3: Building advanced query processing system...")
        
        # Create query decomposition chain - breaks complex questions into sub-questions
        decompose_chain = create_query_decomposition_chain(llm)
        print("  ✅ Query decomposition chain created")
        
        # Create query rewriting chain - optimizes each sub-question
        rewrite_chain = create_query_rewrite_chain(llm)
        print("  ✅ Query rewrite chain created")

        # ====================================================================
        # STEP 4: RAG Chain Construction
        # ====================================================================
        print("\n📋 Step 4: Building Advanced RAG processing chain...")
        
        # Create Advanced RAG chain with full multi-query processing
        advanced_rag_chain = create_advanced_rag_chain(basic_retriever, decompose_chain, rewrite_chain, llm)
        print("  ✅ Advanced RAG chain assembled")

        # ====================================================================
        # STEP 5: System Testing
        # ====================================================================
        print("\n📋 Step 5: Testing the Advanced RAG system...")
        
        # Define complex test questions that benefit from decomposition
        test_questions = [
            "What are key moments between Ziva and Milos?",
            "How do the different party members interact with each other and what are their relationships?",
            "What supernatural threats have the party encountered and how did they handle them?",
            "Describe the progression of the campaign from beginning to current state",
            "What are the main conflicts and how have they evolved over time?"
        ]
        
        # Test the Advanced RAG system
        test_advanced_rag_system(advanced_rag_chain, test_questions)
        
        # ====================================================================
        # SUCCESS SUMMARY
        # ====================================================================
        print("\n🎉 Advanced RAG System Successfully Initialized!")
        print("📊 System Components:")
        print(f"  📚 Vector Store Index: {ELASTIC_SEARCH_INDEX}")
        print(f"  🔍 Retrieval Count: {RETRIEVAL_K} documents per sub-question")
        print(f"  📄 Max Final Documents: {MAX_FINAL_DOCS} after deduplication")
        print(f"  🤖 Language Model: {LLM_CONFIG['model']}")
        print(f"  📡 Embedding Model: {EMBEDDING_CONFIG['azure_deployment']}")
        print(f"  🧩 Enhancements: Query Decomposition + Query Rewriting + Multi-Retrieval")
        print("\n💡 The Advanced RAG system provides comprehensive answers to complex questions!")
        print("🎯 Key Improvements:")
        print("   - Complex questions broken into focused sub-questions")
        print("   - Each sub-question optimized for better retrieval")
        print("   - Multiple document retrievals provide broader context")
        print("   - Deduplication ensures quality without redundancy")

    except Exception as e:
        print(f"\n💥 Error during Advanced RAG system initialization: {e}")
        print("🔧 Check that:")
        print("  - Elasticsearch is running on localhost:9200")
        print("  - Vector store was created (run load_vector_store.py first)")
        print("  - Azure OpenAI credentials are correct")
        print("  - All required packages are installed")
        return 1

    return 0


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    This allows the script to be executed as:
        python run_advanced_rag.py
    
    The exit() call ensures the script returns the proper exit code
    to the operating system for scripting integration.
    """
    exit(main())
