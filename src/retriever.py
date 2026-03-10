import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import get_embedding


def retrieve(query, chunks, chunk_embeddings, top_k=2):

    query_embedding = get_embedding(query)

    similarities = cosine_similarity(
        [query_embedding], chunk_embeddings
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = [chunks[i] for i in top_indices]

    return results