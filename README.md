# Agentic_RAG_System


## Overview

- Designed and implemented a multi-agent Retrieval-Augmented Generation (RAG) system using LangChain, enabling grounded question answering over large document corpora.

- Built specialized LLM agents (planner, retriever, reasoner, verifier) with tool-calling capabilities to improve factual accuracy and reduce hallucinations.

- Integrated vector databases (Chroma) with optimized chunking and embedding strategies for scalable semantic retrieval.

- Evaluated system robustness using adversarial and ambiguous queries, introducing a verification agent to perform self-critique and answer validation.

This system runs **100% locally** on consumer hardware (tested on i7, 16GB RAM, Ubuntu) using Ollama.

---

## Architecture

The system uses a **Directed Cyclic Graph** (DAG) approach:

```mermaid
graph TD;


```

### Key Agents

| Agent | Role | Model |
| --- | --- | --- |


---
## Project Structure

```bash
multi-agent-rag/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── model_config.yaml
│   ├── agent_prompts.yaml
│   └── retriever_config.yaml
│
├── data/
│   ├── raw_docs/
│   │   ├── sample_paper_1.pdf
│   │   └── sample_report_1.pdf
│   └── processed/
│
├── embeddings/
│   └── vector_store/
│
├── agents/
│   ├── planner_agent.py
│   ├── retriever_agent.py
│   ├── reasoning_agent.py
│   ├── verifier_agent.py
│   └── answer_agent.py
│
├── pipelines/
│   ├── ingestion_pipeline.py
│   ├── rag_pipeline.py
│   └── multi_agent_orchestrator.py
│
├── tools/
│   ├── pdf_loader.py
│   ├── embedding_utils.py
│   ├── retrieval_tools.py
│   └── evaluation_tools.py
│
├── evaluation/
│   ├── adversarial_queries.json
│   ├── hallucination_tests.py
│   └── metrics.py
│
├── app/
│   ├── cli.py
│   └── streamlit_app.py   
│
└── experiments/
    ├── chunking_ablation.py
    ├── prompt_ablation.py
    └── retriever_comparison.py


```
## Tech Stack



---

## Getting Started

### 1. Prerequisites

* **Hardware:** 16GB RAM recommended.
* **Software:** Python installed.

### 2. Install Ollama (The Engine)

This manages the local LLMs.

```bash
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

```

### 3. Pull Required Models

We use `mistral` for reasoning (efficient 7B model) and `nomic-embed-text` for embeddings.

```bash
ollama pull mistral
ollama pull nomic-embed-text

```

### 4. Project Setup

```bash
# Clone repository
git clone [https://github.com/amar-at-iitm/Agentic_RAG_System](https://github.com/amar-at-iitm/Agentic_RAG_System)
cd Agentic_RAG_System

# Install dependencies
pip install -r requirements.txt

```

---

## Usage

### 1. Start the LLM Server

Open a terminal and ensure Ollama is running:

```bash
ollama serve

```

### 2. Run the Application

In your project folder:

```bash
streamlit run streamlit_app.py

```

### 3. Interact

1. Open browser to `http://localhost:8501`.
2. 



---

## Future Improvements


---
