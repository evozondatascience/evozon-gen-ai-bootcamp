# RAG Workshop - Complete Implementation Suite

This repository contains a comprehensive suite of Retrieval-Augmented Generation (RAG) implementations, from basic to advanced, designed for educational purposes using a Dungeons & Dragons Curse of Strahd campaign as the knowledge base.

## 📁 File Overview

### 🏗️ Setup and Infrastructure
- **`load_vector_store.py`** - Creates and populates the Elasticsearch vector store
- **`test_vector_store.py`** - Tests the vector store functionality
- **`requirements.txt`** - Python dependencies
- **`evo-rag-workshop.ipynb`** - Interactive Jupyter notebook with all implementations

### 🤖 RAG Implementations (Python Scripts)
- **`run_simple_rag.py`** - Basic RAG implementation
- **`run_enhanced_rag.py`** - Enhanced RAG with query rewriting
- **`run_advanced_rag.py`** - Advanced RAG with query decomposition

### 📚 Knowledge Base
- **`docs/`** - Contains Curse of Strahd campaign notes (Markdown files)

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Elasticsearch
Ensure Elasticsearch is running on `localhost:9200` with the credentials specified in the scripts.

### 3. Create Vector Store
```bash
python load_vector_store.py
```

### 4. Test Vector Store (Optional)
```bash
python test_vector_store.py
```

### 5. Run RAG Implementations

#### Basic RAG
```bash
python run_simple_rag.py
```

#### Enhanced RAG (with Query Rewriting)
```bash
python run_enhanced_rag.py
```

#### Advanced RAG (with Query Decomposition)
```bash
python run_advanced_rag.py
```

## 🎯 RAG Implementation Comparison

### 📝 Basic RAG (`run_simple_rag.py`)
**How it works:**
1. User asks a question
2. System finds similar documents in vector store
3. LLM generates answer using retrieved documents

**Strengths:**
- Simple and fast
- Easy to understand and implement
- Works well for straightforward questions

**Use cases:**
- Simple fact-finding questions
- When low latency is critical
- Proof of concept implementations

---

### 🚀 Enhanced RAG (`run_enhanced_rag.py`)
**How it works:**
1. User asks a question
2. **Query Rewriting:** LLM rewrites the question to optimize retrieval
3. System finds similar documents using the improved query
4. LLM generates answer using retrieved documents

**Key Enhancement:** Query Rewriting
- Transforms casual user language into domain-specific terminology
- Adds relevant context (e.g., "D&D", "Curse of Strahd")
- Expands abbreviated or vague questions

**Example:**
- **Original:** "What are Sally's abilities?"
- **Rewritten:** "What are Sally's magical powers, combat skills, and special abilities in the Curse of Strahd D&D campaign?"

**Strengths:**
- Better retrieval quality
- Handles casual/informal questions better
- Minimal additional complexity

---

### 🎓 Advanced RAG (`run_advanced_rag.py`)
**How it works:**
1. User asks a complex question
2. **Query Decomposition:** LLM breaks question into 3-5 focused sub-questions
3. **Query Rewriting:** Each sub-question is optimized for retrieval
4. **Multi-Retrieval:** System retrieves documents for each sub-question
5. **Deduplication:** Removes duplicate documents
6. LLM generates comprehensive answer using rich context

**Key Enhancements:**
- **Query Decomposition:** Handles complex, multi-faceted questions
- **Multi-Retrieval:** Gathers broader context from multiple searches
- **Deduplication:** Optimizes context quality

**Example:**
- **Original:** "What are key moments between Ziva and Milos?"
- **Decomposed into:**
  1. "Who is Ziva in the Curse of Strahd campaign?"
  2. "Who is Milos and what role does he play?"
  3. "What interactions occurred between Ziva and Milos?"
  4. "What significant events involved both characters?"

**Strengths:**
- Handles complex, nuanced questions
- Provides comprehensive, well-rounded answers
- Reduces information gaps

**Trade-offs:**
- Higher latency (multiple LLM calls)
- Increased API costs
- More complex to debug

## 📊 Performance Comparison

| Feature | Basic RAG | Enhanced RAG | Advanced RAG |
|---------|-----------|--------------|--------------|
| **Query Processing** | Direct | Rewriting | Decomposition + Rewriting |
| **Retrieval Calls** | 1 | 1 | 3-5 |
| **LLM Calls** | 1 | 2 | 6-8 |
| **Latency** | Low | Medium | High |
| **Cost** | Low | Medium | High |
| **Answer Quality** | Good | Better | Best |
| **Complex Questions** | Limited | Good | Excellent |

## 🛠️ Configuration

All scripts share common configuration that can be customized:

### Vector Store Settings
```python
ELASTIC_SEARCH_INDEX = "rag-workshop"
RETRIEVAL_K = 3  # Documents per query
```

### Azure OpenAI Settings
```python
LLM_CONFIG = {
    "azure_deployment": "gpt-4.1-mini",
    "temperature": 0.0  # Deterministic responses
}
```

### Advanced RAG Settings
```python
MAX_FINAL_DOCS = 8  # Maximum documents after deduplication
```

## 🧪 Testing Strategy

Each implementation includes comprehensive testing:

1. **Component Testing:** Individual functions tested in isolation
2. **Integration Testing:** Complete RAG pipelines tested end-to-end
3. **Comparison Testing:** Side-by-side comparison of different approaches
4. **Error Handling:** Robust error handling with helpful messages

## 📚 Educational Features

### For Learning:
- **Clear Documentation:** Every function has detailed docstrings
- **Step-by-Step Execution:** Main functions show clear workflow
- **Comparison Examples:** Direct comparisons between approaches
- **Real-World Context:** Uses engaging D&D campaign data

### For Teaching:
- **Modular Design:** Each concept is isolated and teachable
- **Progressive Complexity:** Build from simple to advanced
- **Visual Output:** Clear progress indicators and formatting
- **Interactive Notebook:** Jupyter notebook for hands-on learning

## 🔧 Troubleshooting

### Common Issues:

**Import Errors:**
```bash
pip install -r requirements.txt
```

**Elasticsearch Connection:**
- Verify Elasticsearch is running: `curl http://localhost:9200`
- Check credentials in configuration
- Ensure firewall allows port 9200

**Azure OpenAI Issues:**
- Verify API key and endpoint
- Check deployment names match your Azure setup
- Ensure sufficient quota/credits

**Vector Store Issues:**
- Run `load_vector_store.py` first
- Check that documents exist in `docs/` directory
- Verify indexing completed successfully

## 🚀 Next Steps

### For Students:
1. **Run All Implementations:** Compare outputs for the same questions
2. **Modify Test Questions:** Try your own questions
3. **Experiment with Parameters:** Adjust chunk sizes, retrieval counts
4. **Add New Documents:** Expand the knowledge base

### For Advanced Users:
1. **Hybrid Search:** Combine semantic and keyword search
2. **Evaluation Metrics:** Implement RAGAS or custom evaluation
3. **Production Deployment:** Add monitoring, caching, scaling
4. **Multi-Modal RAG:** Add image and table processing

### For Production:
1. **Environment Variables:** Externalize all configuration
2. **Logging:** Add comprehensive logging
3. **Monitoring:** Track performance and costs
4. **Caching:** Implement query and response caching
5. **Security:** Add proper authentication and authorization

## 📖 Additional Resources

- **LangChain Documentation:** https://python.langchain.com/
- **Elasticsearch Documentation:** https://www.elastic.co/guide/
- **Azure OpenAI Documentation:** https://docs.microsoft.com/azure/cognitive-services/openai/
- **RAG Best Practices:** Research papers and industry guides

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new RAG techniques
- Improve documentation
- Add more test cases
- Optimize performance
- Add new data sources

---

**Happy Learning!** 🎉

This workshop provides a complete journey through RAG implementations, from basic concepts to advanced techniques. Each level builds upon the previous one, giving you both theoretical understanding and practical experience with state-of-the-art RAG systems.
