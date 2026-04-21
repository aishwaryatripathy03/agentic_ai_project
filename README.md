# PhysicsBot — Study Buddy (Agentic AI Project)

## Overview

**PhysicsBot — Study Buddy** is an intelligent AI assistant designed for B.Tech students to learn and revise physics concepts.
It uses an **Agentic AI architecture** with Retrieval-Augmented Generation (RAG), memory, tool usage, and self-evaluation to provide **accurate, context-aware, and non-hallucinated answers**.

---

## Objective

* Provide clear explanations of physics concepts
* Solve basic numerical problems
* Maintain conversation memory
* Avoid hallucination by grounding answers in a knowledge base

---

## Key Features

* **RAG (Retrieval-Augmented Generation)** using vector database
* **Memory Support** using `thread_id`
* **Smart Routing** (retrieve / tool / memory)
* **Calculator Tool** for numericals
* **Self-Evaluation Node** (faithfulness scoring)
* **Interactive Chat UI** using Streamlit

---

## Architecture

```
User (Streamlit UI)
        ↓
   run_query()
        ↓
   LangGraph App
        ↓
 [memory_node]
        ↓
 [router_node]
   ↓        ↓        ↓
retrieve   tool     skip
   ↓        ↓        ↓
[retrieval_node] [tool_node] [skip_node]
        ↓
   [answer_node]
        ↓
   [eval_node]
        ↓
   [save_node]
        ↓
       END
```

---

## Tech Stack

* **Frontend:** Streamlit
* **Agent Framework:** LangGraph
* **LLM:** Groq (LLaMA models)
* **Vector DB:** ChromaDB
* **Embeddings:** SentenceTransformers
* **Language:** Python

---

## Project Structure

```
project/
│
├── agent.py                # Core agent logic (nodes, graph, LLM)
├── capstone_streamlit.py  # Streamlit UI
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Groq SDK

```bash
pip install groq
```

### 4. Set API Key

#### Windows:

```bash
setx GROQ_API_KEY "your_api_key_here"
```

#### Mac/Linux:

```bash
export GROQ_API_KEY="your_api_key_here"
```

---

## Run the Application

```bash
python -m streamlit run capstone_streamlit.py
```

Open in browser:

```
http://localhost:8501
```

---

## Sample Questions

* What is Newton’s Second Law?
* Explain Simple Harmonic Motion
* What are equations of motion?
* Calculate force for 5kg body with 3 m/s² acceleration
* What is today’s date?

---

## Evaluation

The system evaluates responses using:

* **Faithfulness Score (0–1)**
* Retries if score < 0.7
* Ensures answers are grounded in retrieved context

---

## Limitations

* Covers only **B.Tech Physics (basic topics)**
* Does not provide advanced derivations
* Numerical solving is limited to simple calculations

---

## Future Improvements

* Add more physics topics
* Support advanced numerical solving
* Add diagrams and visual explanations
* Deploy on web/WhatsApp

---

## Author

**Aishwarya Tripathy**
B.Tech Student — Agentic AI Capstone Project

License
This project is for academic and learning purposes.

