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
RETRIEVAL_K = 3  # Number of documents to retrieve for each query

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
    os.environ['AZURE_OPENAI_API_KEY'] = '1906f94ca9a24877ab2423c4cf4b746c'
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
# QUERY ENHANCEMENT - THE KEY IMPROVEMENT
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
# ENHANCED RETRIEVAL SYSTEM
# ============================================================================

def enhanced_retrieval(query: str, retriever, rewrite_chain) -> List[Document]:
    """
    Perform enhanced document retrieval with query rewriting.
    
    Args:
        query: Original user question
        retriever: Basic document retriever
        rewrite_chain: Query rewriting chain
        
    Returns:
        List of relevant documents
    """
    # Step 1: Rewrite the query for better retrieval
    print(f"🔄 Original query: {query}")
    rewritten_query = rewrite_chain.invoke({"question": query})
    print(f"✨ Rewritten query: {rewritten_query}")
    
    # Step 2: Use the improved query to retrieve documents
    documents = retriever.get_relevant_documents(rewritten_query)
    print(f"📚 Retrieved {len(documents)} documents using enhanced query")
    
    return documents


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


def create_enhanced_rag_chain(retriever, rewrite_chain, llm) -> object:
    """
    Build the complete Enhanced RAG processing chain.
    
    This function creates a LangChain pipeline that:
    1. Takes a user question
    2. Uses enhanced retrieval (with query rewriting) to find relevant documents
    3. Formats the documents as context
    4. Applies the RAG prompt template
    5. Sends everything to the LLM for answer generation
    6. Parses the response to return clean text
    
    Args:
        retriever: Basic document retriever
        rewrite_chain: Query rewriting chain
        llm: Language model for text generation
        
    Returns:
        Runnable chain that can process questions and return answers
    """
    print("🔗 Building Enhanced RAG processing chain...")
    
    # Create the RAG prompt template
    rag_prompt = create_rag_prompt()
    
    # Build the complete processing chain with enhanced retrieval
    enhanced_rag_chain = (
        {
            # Use enhanced retrieval with query rewriting, then format documents
            "context": lambda query: format_docs(enhanced_retrieval(query, retriever, rewrite_chain)),
            # Pass the original question through unchanged
            "question": RunnablePassthrough()
        }
        | rag_prompt          # Apply the prompt template
        | llm                 # Generate response with LLM
        | StrOutputParser()   # Extract text from response object
    )
    
    return enhanced_rag_chain


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_enhanced_rag_system(enhanced_rag_chain, test_questions: List[str]) -> None:
    """
    Test the Enhanced RAG system.
    
    Args:
        enhanced_rag_chain: The enhanced RAG processing chain
        test_questions: List of questions to test
    """
    print(f"\n🧪 Testing Enhanced RAG system with {len(test_questions)} questions...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Test Enhanced RAG
            response = enhanced_rag_chain.invoke(question)
            print(f"🤖 Answer: {response}")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
    
    print("\n" + "=" * 60)


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main() -> int:
    """
    Main function that orchestrates the Enhanced RAG system execution.
    
    This function demonstrates an enhanced RAG workflow with query rewriting:
    1. Environment setup and model initialization
    2. Vector store connection
    3. Query rewriting chain creation
    4. Enhanced RAG chain construction
    5. System testing
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print("🚀 Enhanced RAG System with Query Rewriting Starting...")
    print("=" * 60)

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
        
        # Create language model for response generation AND query rewriting
        llm = create_llm()
        print("  ✅ Language model created")

        # ====================================================================
        # STEP 2: Vector Store Connection
        # ====================================================================
        print("\n📋 Step 2: Connecting to vector store...")
        
        # Connect to the pre-loaded Elasticsearch vector store
        vector_store = create_vector_store(embeddings)
        print("  ✅ Connected to vector store")

        # Create basic retriever
        basic_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        print(f"  ✅ Document retriever created (k={RETRIEVAL_K})")

        # ====================================================================
        # STEP 3: Query Enhancement System
        # ====================================================================
        print("\n📋 Step 3: Building query enhancement system...")
        
        # Create query rewriting chain - the key enhancement
        rewrite_chain = create_query_rewrite_chain(llm)
        print("  ✅ Query rewrite chain created")

        # ====================================================================
        # STEP 4: RAG Chain Construction
        # ====================================================================
        print("\n📋 Step 4: Building Enhanced RAG processing chain...")
        
        # Create Enhanced RAG chain with query rewriting
        enhanced_rag_chain = create_enhanced_rag_chain(basic_retriever, rewrite_chain, llm)
        print("  ✅ Enhanced RAG chain assembled")

        # ====================================================================
        # STEP 5: System Testing
        # ====================================================================
        print("\n📋 Step 5: Testing the Enhanced RAG system...")
        
        # Define test questions to demonstrate the enhancement
        test_questions = [
            "What are Sally's abilities?",
            "Tell me about Milos",
            "What happened with the vampire?",
            "Describe the party dynamics",
            "Any magical encounters?"
        ]
        
        # Test the Enhanced RAG system
        test_enhanced_rag_system(enhanced_rag_chain, test_questions)
        
        # ====================================================================
        # SUCCESS SUMMARY
        # ====================================================================
        print("\n🎉 Enhanced RAG System Successfully Initialized!")
        print("📊 System Components:")
        print(f"  📚 Vector Store Index: {ELASTIC_SEARCH_INDEX}")
        print(f"  🔍 Retrieval Count: {RETRIEVAL_K} documents per query")
        print(f"  🤖 Language Model: {LLM_CONFIG['model']}")
        print(f"  📡 Embedding Model: {EMBEDDING_CONFIG['azure_deployment']}")
        print(f"  ✍️ Enhancement: Query Rewriting for better retrieval")
        print("\n💡 The Enhanced RAG system optimizes user questions for better document matching!")

    except Exception as e:
        print(f"\n💥 Error during Enhanced RAG system initialization: {e}")
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
        python run_enhanced_rag.py
    
    The exit() call ensures the script returns the proper exit code
    to the operating system for scripting integration.
    """
    exit(main())
