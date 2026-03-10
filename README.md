# Knowledge-Based Q&A System (RAG)

A prototype Retrieval-Augmented Generation (RAG) system that answers questions from stored documents using LLMs.

## Features

- Document loading and preprocessing
- Document chunking for LLM context
- Embedding generation
- Vector similarity search
- Retrieval-Augmented Generation (RAG)
- Context-aware question answering

## Tech Stack

- Python
- OpenAI API
- NumPy
- Scikit-learn

## Project Architecture

User Question
↓
Embedding Generation
↓
Vector Similarity Search
↓
Retrieve Relevant Chunks
↓
Send Context + Question to LLM
↓
Generate Answer

## Installation

Clone the repository
git clone https://github.com/prateek6542/knowledge_qa_system/

Install dependencies
pip install -r requirements.txt

Add OpenAI API key in `.env`
OPENAI_API_KEY=your_key_here

Run the system
python main.py

## Example
Question:
What is Retrieval Augmented Generation?
Answer:
RAG combines information retrieval with language models to generate grounded responses.
