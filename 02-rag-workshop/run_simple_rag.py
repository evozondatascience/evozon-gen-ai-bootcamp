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


def create_rag_chain(retriever, llm) -> object:
    print("🔗 Building RAG processing chain...")
    
    # Create the RAG prompt template
    rag_prompt = create_rag_prompt()
    
    # Build the complete processing chain
    rag_chain = (
        {
            # Retrieve relevant documents and format them as context
            "context": retriever | format_docs,
            # Pass the original question through unchanged
            "question": RunnablePassthrough()
        }
        | rag_prompt          # Apply the prompt template
        | llm                 # Generate response with LLM
        | StrOutputParser()   # Extract text from response object
    )
    
    return rag_chain


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_rag_system(rag_chain, test_questions: List[str]) -> None:
    print(f"\n🧪 Testing RAG system with {len(test_questions)} questions...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Get answer from RAG system
            response = rag_chain.invoke(question)
            print(f"🤖 Answer: {response}")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
    
    print("\n" + "=" * 60)


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main() -> int:
    print("🚀 Simple RAG System Starting...")
    print("=" * 50)

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
        
        # Create language model for response generation
        llm = create_llm()
        print("  ✅ Language model created")

        # ====================================================================
        # STEP 2: Vector Store Connection
        # ====================================================================
        print("\n📋 Step 2: Connecting to vector store...")
        
        # Connect to the pre-loaded Elasticsearch vector store
        vector_store = create_vector_store(embeddings)
        print("  ✅ Connected to vector store")

        # Create retriever that will find relevant documents for queries
        # k=3 means we'll retrieve the top 3 most similar documents
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        print(f"  ✅ Document retriever created (k={RETRIEVAL_K})")

        # ====================================================================
        # STEP 3: RAG Chain Construction
        # ====================================================================
        print("\n📋 Step 3: Building RAG processing chain...")
        
        # Create the complete RAG processing pipeline
        rag_chain = create_rag_chain(retriever, llm)
        print("  ✅ RAG chain assembled")

        # ====================================================================
        # STEP 4: System Testing
        # ====================================================================
        print("\n📋 Step 4: Testing the RAG system...")
        
        # Define test questions to demonstrate the system
        test_questions = [
            "What happened between Milos and Ireena?",
            "Who is Strahd and what makes him dangerous?", 
            "Tell me about the town of Barovia",
            "What magical items have the characters found?",
            "Describe the relationship between different party members"
        ]
        
        # Run tests to demonstrate functionality
        test_rag_system(rag_chain, test_questions)
        
        # ====================================================================
        # SUCCESS SUMMARY
        # ====================================================================
        print("\n🎉 RAG System Successfully Initialized!")
        print("📊 System Components:")
        print(f"  📚 Vector Store Index: {ELASTIC_SEARCH_INDEX}")
        print(f"  🔍 Retrieval Count: {RETRIEVAL_K} documents per query")
        print(f"  🤖 Language Model: {LLM_CONFIG['model']}")
        print(f"  📡 Embedding Model: {EMBEDDING_CONFIG['azure_deployment']}")
        print("\n💡 The RAG system is ready to answer questions about the Curse of Strahd campaign!")

    except Exception as e:
        print(f"\n💥 Error during RAG system initialization: {e}")
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
        python run_simple_rag.py
    
    The exit() call ensures the script returns the proper exit code
    to the operating system for scripting integration.
    """
    exit(main())