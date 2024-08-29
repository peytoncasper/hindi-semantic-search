import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)
index = pc.Index("multi-lingual-index")

def generate_embeddings(model, sentences):
    print("Generating embeddings for sentences...")
    embeddings = model.encode(sentences)
    print("Embeddings generated successfully.")
    return embeddings

def parse_content_from_hindi_dataset(file_path):
    df = pd.read_csv(file_path, usecols=['Content'])
    return df['Content'].tolist()

def parse_content_from_gujarati_dataset(file_path):
    df = pd.read_csv(file_path, usecols=['headline'])
    return df['headline'].tolist()

model = SentenceTransformer('l3cube-pune/indic-sentence-similarity-sbert', device='mps')

if len(sys.argv) < 2:
    print("Usage: python main.py <hindi|gujarati>")
    sys.exit(1)

dataset_type = sys.argv[1]

if dataset_type == "hindi":
    dataset = parse_content_from_hindi_dataset("hindi_dataset.csv")
    embeddings = generate_embeddings(model, dataset)
    for i, (vector, text) in enumerate(zip(embeddings, dataset)):
        index.upsert(vectors=[{
            "id": str(i),
            "values": vector,
            "metadata": {"text": text}
        }])
elif dataset_type == "gujarati":
    dataset = parse_content_from_gujarati_dataset("gujarati_dataset.csv")
    embeddings = generate_embeddings(model, dataset)
    for i, (vector, text) in enumerate(zip(embeddings, dataset)):
        index.upsert(vectors=[{
            "id": str(i),
            "values": vector,
            "metadata": {"text": text}
        }])
else:
    print("Invalid dataset type. Use 'hindi' or 'gujarati'.")
    sys.exit(1)
