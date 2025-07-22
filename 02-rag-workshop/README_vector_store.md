# Vector Store Setup and Usage

This directory contains utilities for setting up and testing the Elasticsearch vector store used in the RAG workshop.

## Files Overview

- **`load_vector_store.py`** - Loads documents and creates the vector store
- **`test_vector_store.py`** - Tests the vector store with sample queries
- **`evo-rag-workshop.ipynb`** - Main workshop notebook with RAG implementations
- **`requirements.txt`** - Python dependencies

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Elasticsearch
Make sure Elasticsearch is running on `localhost:9200` with the credentials specified in the scripts.

### 3. Load the Vector Store
```bash
python load_vector_store.py
```

This will:
- Load all Markdown documents from `docs/` directory
- Split them into chunks of 2000 characters with 100 character overlap
- Generate embeddings using Azure OpenAI
- Store everything in Elasticsearch index `rag-workshop`

Expected output:
```
🚀 Starting Vector Store Loading Process
==================================================
✅ Environment configured
✅ Embeddings model created
Loading documents from docs/...
Loaded 1 documents
Splitting documents into chunks (size: 2000, overlap: 100)...
Created X document chunks
Creating Elasticsearch vector store with index: rag-workshop
Indexing X document chunks...
This may take a few minutes as embeddings are generated...
✅ Documents successfully indexed in the vector store!
==================================================
🎉 Vector store loading completed successfully!
📊 Summary:
   - Documents loaded: 1
   - Document chunks created: X
   - Elasticsearch index: rag-workshop
   - Ready for RAG queries!
```

### 4. Test the Vector Store
```bash
python test_vector_store.py
```

This will run sample queries against your vector store to verify it's working correctly.

### 5. Use the Workshop Notebook
Open `evo-rag-workshop.ipynb` in Jupyter/VS Code and run through the RAG workshop!

## Configuration

### Customizing Document Loading
In `load_vector_store.py`, you can modify:

```python
# Configuration Constants
DOCS_DIRECTORY = "docs/"  # Change source directory
CHUNK_SIZE = 2000         # Adjust chunk size
CHUNK_OVERLAP = 100       # Adjust overlap
```

### Elasticsearch Settings
Update the `ELASTICSEARCH_CONFIG` dictionary in `load_vector_store.py`:

```python
ELASTICSEARCH_CONFIG = {
    "es_url": "https://localhost:9200/",  # Your ES URL
    "es_user": "elastic",                 # Your username
    "es_password": "your_password",       # Your password
    "es_params": {"verify_certs": False}  # SSL settings
}
```

## Troubleshooting

### Import Errors
Make sure you've installed all requirements:
```bash
pip install -r requirements.txt
```

### Elasticsearch Connection Issues
- Verify Elasticsearch is running: `curl http://localhost:9200`
- Check credentials in the configuration
- Ensure your firewall allows connections to port 9200

### Azure OpenAI Issues
- Verify your API key is correct and active
- Check that your deployment names match (`gpt-4.1-mini`, `embed-new`)
- Ensure you have sufficient quota/credits

## Script Structure

### load_vector_store.py Functions:
- `setup_environment()` - Configure Azure OpenAI settings
- `create_embeddings_model()` - Initialize embedding model
- `load_documents()` - Load Markdown files from directory
- `split_documents()` - Split documents into chunks
- `create_vector_store()` - Setup Elasticsearch connection
- `index_documents()` - Add documents to vector store
- `main()` - Orchestrates the entire process

### test_vector_store.py Functions:
- `connect_to_vector_store()` - Connect to existing vector store
- `test_queries()` - Define sample test queries
- `test_retrieval()` - Test document retrieval for queries
- `main()` - Run all tests

## Next Steps

After successfully loading your vector store:

1. **Explore the Notebook**: Work through `evo-rag-workshop.ipynb` to learn about different RAG implementations
2. **Try Custom Queries**: Modify the test queries to match your use case
3. **Experiment with Parameters**: Adjust chunk sizes, overlap, and retrieval parameters
4. **Scale Up**: Add more documents to your `docs/` directory and re-run the loader

## Production Considerations

For production use, consider:
- **Security**: Use proper authentication and HTTPS
- **Monitoring**: Add logging and error tracking
- **Scalability**: Consider distributed Elasticsearch setup
- **Cost Optimization**: Monitor API usage and implement caching
- **Data Management**: Implement document versioning and updates
