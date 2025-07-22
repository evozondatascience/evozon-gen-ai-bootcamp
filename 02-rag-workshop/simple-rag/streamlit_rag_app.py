import streamlit as st
import sys
import os
from typing import Optional

# Import the RAG components from the existing simple RAG file
from run_simple_rag import (
    setup_environment,
    create_embeddings_model,
    create_llm,
    create_vector_store,
    create_rag_chain,
    RETRIEVAL_K,
    ELASTIC_SEARCH_INDEX,
    LLM_CONFIG,
    EMBEDDING_CONFIG
)

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

# Configure the Streamlit page
st.set_page_config(
    page_title="Curse of Strahd - DM Assistant",
    page_icon="🧛‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state variables for the Streamlit app."""
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False


# ============================================================================
# RAG SYSTEM INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components with caching for performance."""
    try:
        with st.spinner("🚀 Initializing RAG system..."):
            # Setup environment
            setup_environment()
            
            # Create models
            embeddings = create_embeddings_model()
            llm = create_llm()
            
            # Connect to vector store
            vector_store = create_vector_store(embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            
            # Create RAG chain
            rag_chain = create_rag_chain(retriever, llm)
            
            return rag_chain, True, None
            
    except Exception as e:
        return None, False, str(e)


# ============================================================================
# USER INTERFACE COMPONENTS
# ============================================================================

def render_sidebar():
    """Render the sidebar with system information and controls."""
    with st.sidebar:
        st.title("🧛‍♂️ DM Assistant")
        st.markdown("---")
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.system_initialized:
            st.success("✅ RAG System Ready")
            st.info(f"📚 Index: {ELASTIC_SEARCH_INDEX}")
            st.info(f"🔍 Retrieval: {RETRIEVAL_K} docs")
            st.info(f"🤖 Model: {LLM_CONFIG['model']}")
        else:
            st.error("❌ System Not Initialized")
        
        st.markdown("---")
        
        # Quick help
        st.subheader("💡 Quick Tips")
        st.markdown("""
        **Sample Questions:**
        - What happened between Milos and Ireena?
        - Who is Strahd and what makes him dangerous?
        - Tell me about the town of Barovia
        - What magical items have the characters found?
        - Describe party member relationships
        """)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_main_interface():
    """Render the main chat interface."""
    st.title("🎲 Curse of Strahd Campaign Assistant")
    st.markdown("*Your AI-powered Dungeon Master's companion for the Curse of Strahd campaign*")
    
    # Initialize RAG system if not already done
    if not st.session_state.system_initialized:
        with st.container():
            st.info("🔧 Initializing the RAG system. This may take a moment...")
            
            rag_chain, success, error = initialize_rag_system()
            
            if success:
                st.session_state.rag_chain = rag_chain
                st.session_state.system_initialized = True
                st.success("🎉 RAG system initialized successfully!")
                st.rerun()
            else:
                st.error(f"❌ Failed to initialize RAG system: {error}")
                st.error("""
                **Troubleshooting Steps:**
                1. Ensure Elasticsearch is running on localhost:9200
                2. Run `load_vector_store.py` to create the vector store
                3. Check Azure OpenAI credentials
                4. Verify all required packages are installed
                """)
                return
    
    # Display chat messages
    display_chat_history()
    
    # Chat input
    handle_user_input()


def display_chat_history():
    """Display the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input():
    """Handle user input and generate responses."""
    # Chat input
    if prompt := st.chat_input("Ask me about your Curse of Strahd campaign..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching through campaign notes..."):
                try:
                    # Get response from RAG chain
                    response = st.session_state.rag_chain.invoke(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"❌ Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


# ============================================================================
# SAMPLE QUESTIONS SECTION
# ============================================================================

def render_sample_questions():
    """Render a section with sample questions for easy testing."""
    if st.session_state.system_initialized and len(st.session_state.messages) == 0:
        st.markdown("---")
        st.subheader("🎯 Try These Sample Questions")
        
        # Create columns for sample questions
        col1, col2 = st.columns(2)
        
        sample_questions = [
            "What happened between Milos and Ireena?",
            "Who is Strahd and what makes him dangerous?",
            "Tell me about the town of Barovia",
            "What magical items have the characters found?",
            "Describe the relationship between different party members"
        ]
        
        for i, question in enumerate(sample_questions):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    # Simulate user input
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    # Generate response
                    with st.spinner("🔍 Searching through campaign notes..."):
                        try:
                            response = st.session_state.rag_chain.invoke(question)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_message = f"❌ Error generating response: {str(e)}"
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
                    
                    st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main interface
    render_main_interface()
    
    # Render sample questions
    render_sample_questions()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
        "🧛‍♂️ Curse of Strahd Campaign Assistant | Powered by RAG & Azure OpenAI"
        "</div>", 
        unsafe_allow_html=True
    )


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
