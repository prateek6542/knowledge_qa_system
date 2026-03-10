from dotenv import load_dotenv
import os
import numpy as np

from src.document_loader import load_document
from src.chunking import chunk_text
from src.embeddings import get_embedding
from src.retriever import retrieve
from src.qa_system import generate_answer


# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    print("ERROR: OpenAI API key not found. Please add it to the .env file.")
    exit()


def main():

    print("Loading knowledge documents...")

    # Load document
    text = load_document("data/knowledge.txt")

    # Split document into chunks
    chunks = chunk_text(text)

    print(f"Document split into {len(chunks)} chunks")

    print("Creating embeddings for document chunks...")

    # Generate embeddings
    chunk_embeddings = []

    for chunk in chunks:
        embedding = get_embedding(chunk)
        chunk_embeddings.append(embedding)

    chunk_embeddings = np.array(chunk_embeddings)

    print("System ready! Ask questions (type 'exit' to quit).")

    while True:

        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            print("Exiting Q&A system.")
            break

        # Retrieve relevant chunks
        relevant_chunks = retrieve(question, chunks, chunk_embeddings)

        context = "\n".join(relevant_chunks)

        # Generate answer using LLM
        answer = generate_answer(question, context)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()

print("API KEY:", os.getenv("OPENAI_API_KEY"))