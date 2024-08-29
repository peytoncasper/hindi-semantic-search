# Multi-Lingual Semantic Search

This project demonstrates how to perform semantic search on multi-lingual datasets using Sentence Transformers and Pinecone.

## Setup

1. Install the required dependencies:
   ```
   pip install pandas sentence_transformers pinecone-client
   ```

2. Set up the following environment variables:
   - `PINECONE_API_KEY`: Your Pinecone API key.

3. Create a Pinecone index:
   - Log in to your Pinecone account at https://app.pinecone.io.
   - Click on "Create Index".
   - Set the following index configuration:
     - Name: "multi-lingual-example"
     - Dimension: 768
     - Metric: cosine
   - Click "Create" to create the index.


## Usage

### Loading Data

To load data into the Pinecone index, run the `load.py` script with the dataset type as an argument:

```
python load.py <hindi|gujarati>
```


The script supports loading data from two types of datasets:
- Hindi dataset: Expects a CSV file named "hindi_dataset.csv" with a 'Content' column.
- Gujarati dataset: Expects a CSV file named "gujarati_dataset.csv" with a 'headline' column.

The script will generate embeddings for the text data using the 'l3cube-pune/indic-sentence-similarity-sbert' model and upsert the embeddings along with the text metadata into the Pinecone index.

### Querying

To perform a semantic search query, run the `query.py` script with the query text as an argument:

```
python query.py <query_text>
```


The script will generate an embedding for the query text using the same model and retrieve the top 10 most similar results from the Pinecone index. It will display the similarity score and the associated text for each result.

## Environment Variables

- `PINECONE_API_KEY`: Your Pinecone API key. This is required to connect to your Pinecone index.

Make sure to set the `PINECONE_API_KEY` environment variable before running the scripts.

## Dependencies

- pandas
- sentence_transformers
- pinecone-client

The project uses the 'l3cube-pune/indic-sentence-similarity-sbert' model for generating embeddings, which supports multi-lingual text.
