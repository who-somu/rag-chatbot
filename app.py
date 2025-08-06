import streamlit as st
import torch
import faiss
import numpy as np
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from huggingface_hub import login
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="📄 RAG Chatbot with Llama 3.1",
    page_icon="🤖",
    layout="wide"
)

# --- Hugging Face Login ---
# Securely handle Hugging Face token using Streamlit's secrets management.
# The user should add their token to secrets.toml, e.g., HF_TOKEN = "hf_..."
try:
    hf_token = st.secrets["HF_TOKEN"]
    login(token=hf_token)
    st.sidebar.success("Successfully logged in to Hugging Face.")
except (KeyError, Exception):
    st.sidebar.error("Hugging Face token not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- Caching Expensive Models ---
# Use st.cache_resource to load models only once.
@st.cache_resource
def load_models():
    """
    Loads the embedding and language models from Hugging Face.
    Uses quantization for the language model to save memory and improve speed.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    model_name = "distilgpt2"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return embedding_model, generator

with st.spinner("Loading models... This may take a moment."):
    embedding_model, generator = load_models()

# --- Document Processing and Caching ---
# Use st.cache_data to process the document only when its content changes.
@st.cache_data
def process_document(file_content_bytes):
    """
    Processes the uploaded text file by chunking it and creating a FAISS index.
    The FAISS index is a searchable vector database of the document chunks.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(file_content_bytes)
        tmp_path = tmp.name

    try:
        loader = TextLoader(tmp_path)
        text_content = loader.load()

        chunk_size = 512
        words = text_content[0].page_content.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
        embedding_dim = chunk_embeddings.shape[1]

        index = faiss.IndexFlatL2(embedding_dim)
        index.add(chunk_embeddings.cpu().numpy())

    finally:
        os.remove(tmp_path)

    return chunks, index

# --- Main App Logic ---
st.title("RAG Chatbot with Llama 3.1")
st.markdown("Upload a `.txt` document and ask questions about its content.")

# Initialize session state for chat history and processed data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

# Sidebar for file upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])

    if uploaded_file:
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            with st.spinner("Processing document..."):
                file_bytes = uploaded_file.getvalue()
                chunks, index = process_document(file_bytes)
                st.session_state.processed_data = {"chunks": chunks, "index": index}
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.messages = [] # Clear chat history for new file
                st.success(f"Document '{uploaded_file.name}' processed successfully.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_query := st.chat_input("Ask a question about your document..."):
    if not st.session_state.processed_data:
        st.warning("Please upload a document first.")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve relevant chunks from the document
    with st.spinner("Searching document..."):
        chunks, index = st.session_state.processed_data["chunks"], st.session_state.processed_data["index"]
        query_embedding = embedding_model.encode([user_query], convert_to_tensor=True)
        distances, indices = index.search(query_embedding.cpu().numpy(), k=3)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

    # Generate response from the language model
    with st.spinner("Generating answer..."):
        chat = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question based only on the context provided. Do not use any outside knowledge."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
        ]
        prompt = generator.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        generation_args = {
            "max_new_tokens": 250,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }

        output = generator(prompt, **generation_args)
        raw_text = output[0]["generated_text"]
        final_answer = raw_text[len(prompt):].strip()

    # Add assistant response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    with st.chat_message("assistant"):
        st.markdown(final_answer)
