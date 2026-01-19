from __future__ import annotations

import json
import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path so modules can be imported
sys.path.append(str(Path(__file__).parent.parent))

from pipelines import IngestionPipeline, RAGPipeline
from tools.config_loader import load_yaml_file

st.set_page_config(page_title="Agentic RAG Console", layout="wide")

st.title("Agentic RAG System")
st.caption("All clients are pinned to Claude Sonnet 4.5 by policy.")

model_config = load_yaml_file(Path("config/model_config.yaml"))
st.sidebar.header("Model Policy")
st.sidebar.code(json.dumps(model_config.get("global", {}), indent=2))

with st.sidebar:
    if st.button("Rebuild Vector Store"):
        pipeline = IngestionPipeline()
        pipeline.ingest(Path("data/raw_docs"))
        st.success("Vector store rebuilt.")

question = st.text_area("Ask a question", placeholder="What is the main contribution of sample_paper_1.pdf?")

if st.button("Run Multi-Agent RAG") and question:
    pipeline = RAGPipeline()
    with st.spinner("Coordinating agents..."):
        result = pipeline.query(question)
    st.subheader("Plan")
    st.json(result["plan"])
    st.subheader("Retrieved context")
    st.json(result["retrieved"])
    st.subheader("Verification")
    st.json(result["verification"])
    st.subheader("Answer")
    st.markdown(result["answer"])
else:
    st.info("Enter a question and press the button to run the pipeline.")
