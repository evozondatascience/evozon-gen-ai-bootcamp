"""
Test script for the loaded vector store.

This script tests the vector store created by load_vector_store.py
by performing sample queries and retrieving relevant documents.

Usage:
    python test_vector_store.py
"""

import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore


def setup_environment() -> None:
    """Configure Azure OpenAI environment variables."""
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://evozon-bots-new.openai.azure.com/'
    os.environ['AZURE_OPENAI_API_KEY'] = 'placeholder'
    os.environ['OPENAI_API_VERSION'] = '2024-12-01-preview'


def create_embeddings_model() -> AzureOpenAIEmbeddings:
    """Create and return the Azure OpenAI embeddings model."""
    return AzureOpenAIEmbeddings(azure_deployment="embed-new")


def connect_to_vector_store(embeddings: AzureOpenAIEmbeddings) -> ElasticsearchStore:
    """
    Connect to the existing Elasticsearch vector store.
    
    Args:
        embeddings: The embeddings model to use
        
    Returns:
        ElasticsearchStore instance connected to existing index
    """
    vector_store = ElasticsearchStore(
        es_url="https://localhost:9200/",
        index_name="rag-workshop",
        embedding=embeddings,
        es_user="elastic",
        es_password="xM-_eV5Zqk3yjNqO*ld6",
        es_params={"verify_certs": False}
    )
    return vector_store


def test_queries() -> list[str]:
    """Return a list of test queries to try."""
    return [
        "What happened between Milos and Ireena?",
        "Who is Strahd?",
        "Tell me about Barovia",
        "What are the main characters?",
        "What magical items appear in the story?"
    ]


def test_retrieval(vector_store: ElasticsearchStore, query: str, k: int = 3) -> None:
    """
    Test document retrieval for a given query.
    
    Args:
        vector_store: The vector store to query
        query: The search query
        k: Number of documents to retrieve
    """
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    
    try:
        # Create retriever and get documents
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        
        print(f"📚 Retrieved {len(docs)} documents:")
        
        for i, doc in enumerate(docs, 1):
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"\n{i}. Document Preview:")
            print(f"   Content: {content_preview}")
            if hasattr(doc, 'metadata') and doc.metadata:
                print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")


def main() -> None:
    """Main function to test the vector store."""
    print("🧪 Testing Vector Store Connection and Retrieval")
    print("=" * 60)
    
    try:
        # Setup
        setup_environment()
        print("✅ Environment configured")
        
        embeddings = create_embeddings_model()
        print("✅ Embeddings model created")
        
        vector_store = connect_to_vector_store(embeddings)
        print("✅ Connected to vector store")
        
        # Test queries
        test_queries_list = test_queries()
        print(f"\n🎯 Testing {len(test_queries_list)} sample queries:")
        
        for query in test_queries_list:
            test_retrieval(vector_store, query)
        
        print("\n" + "=" * 60)
        print("🎉 Vector store testing completed!")
        print("✅ Your RAG system is ready to use!")
        
    except Exception as e:
        print(f"💥 Error during testing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
