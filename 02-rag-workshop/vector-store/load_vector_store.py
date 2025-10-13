import os
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document


# Configuration Constants
ELASTIC_SEARCH_INDEX = "rag-workshop"
DOCS_DIRECTORY = "docs/"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

# Elasticsearch Configuration
ELASTICSEARCH_CONFIG = {
    "es_url": "https://localhost:9200/",
    "es_user": "elastic",
    "es_password": "xM-_eV5Zqk3yjNqO*ld6",
    "es_params": {"verify_certs": False}
}


def setup_environment() -> None:
    """Configure Azure OpenAI environment variables."""
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://evozon-bots-new.openai.azure.com/'
    os.environ['AZURE_OPENAI_API_KEY'] = 'SECRET'
    os.environ['OPENAI_API_VERSION'] = '2024-12-01-preview'


def create_embeddings_model() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(azure_deployment="embed-new")


def load_documents(directory: str = DOCS_DIRECTORY) -> List[Document]:
    print(f"Loading documents from {directory}...")
    loader = DirectoryLoader(
        directory, 
        glob="**/*.md", 
        loader_cls=UnstructuredMarkdownLoader
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs


def split_documents(documents: List[Document], 
                   chunk_size: int = CHUNK_SIZE, 
                   chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    print(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks")
    return splits


def create_vector_store(embeddings: AzureOpenAIEmbeddings) -> ElasticsearchStore:
    print(f"Creating Elasticsearch vector store with index: {ELASTIC_SEARCH_INDEX}")
    vector_store = ElasticsearchStore(
        es_url=ELASTICSEARCH_CONFIG["es_url"],
        index_name=ELASTIC_SEARCH_INDEX,
        embedding=embeddings,
        es_user=ELASTICSEARCH_CONFIG["es_user"],
        es_password=ELASTICSEARCH_CONFIG["es_password"],
        es_params=ELASTICSEARCH_CONFIG["es_params"]
    )
    return vector_store


def index_documents(vector_store: ElasticsearchStore, documents: List[Document]) -> None:
    print(f"Indexing {len(documents)} document chunks...")
    print("This may take a few minutes as embeddings are generated...")
    
    try:
        vector_store.add_documents(documents)
        print("✅ Documents successfully indexed in the vector store!")
    except Exception as e:
        print(f"❌ Error indexing documents: {e}")
        raise


def main() -> None:
    """Main function to execute the vector store loading process."""
    print("🚀 Starting Vector Store Loading Process")
    print("=" * 50)
    
    try:
        # Step 1: Setup environment
        setup_environment()
        print("✅ Environment configured")
        
        # Step 2: Create embeddings model
        embeddings = create_embeddings_model()
        print("✅ Embeddings model created")
        
        # Step 3: Load documents
        documents = load_documents()
        
        # Step 4: Split documents into chunks
        document_chunks = split_documents(documents)
        
        # Step 5: Create vector store
        vector_store = create_vector_store(embeddings)
        
        # Step 6: Index documents
        index_documents(vector_store, document_chunks)
        
        print("=" * 50)
        print("🎉 Vector store loading completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Documents loaded: {len(documents)}")
        print(f"   - Document chunks created: {len(document_chunks)}")
        print(f"   - Elasticsearch index: {ELASTIC_SEARCH_INDEX}")
        print(f"   - Ready for RAG queries!")
        
    except Exception as e:
        print(f"💥 Error during vector store loading: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

