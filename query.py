import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)
index = pc.Index("multi-lingual-index")

model = SentenceTransformer('l3cube-pune/indic-sentence-similarity-sbert', device='mps')

def query(query_text):
    query_embedding = model.encode([query_text])


    query_response = index.query(
        vector=query_embedding[0].tolist(),
        top_k=10,
        include_metadata=True,
        include_values=True
    )

    return query_response

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python query.py <query_text>")
        sys.exit(1)

    query_text = sys.argv[1]
    results = query(query_text)

    for result in results['matches']:
        print(f"Score: {result['score']}, Text: {result['metadata']['text']}")
