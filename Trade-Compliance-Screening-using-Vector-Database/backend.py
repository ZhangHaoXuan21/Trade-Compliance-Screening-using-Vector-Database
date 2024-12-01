from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_huggingface import HuggingFaceEmbeddings
from vector_shield.vector import QdrantVectorStorePredictor


from dotenv import load_dotenv
import os

def get_predictor():
    load_dotenv()

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    model_name = "jinaai/jina-embeddings-v3"
    model_kwargs = {'device': 'cuda', 'trust_remote_code':True}
    encode_kwargs = {'normalize_embeddings': False}
    jina_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


    qdrant_db_predictor = QdrantVectorStorePredictor(
        embeddings=jina_embeddings,
        retrieval_top_k=20,
        reranker_top_k=5,
        db_url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name="transaction_description"
    )

    qdrant_db_predictor.connect_vector_db()

    return qdrant_db_predictor


qdrant_db_predictor = get_predictor()

app = FastAPI()

# Define the Pydantic model
class Query(BaseModel):
    user_query: str


@app.post("/predict")
async def chat_response(request: Query):
    # Access the incoming JSON
    user_query = request.user_query

    transaction_description = user_query

    predict_data = qdrant_db_predictor.predict(transaction_description, explain=True)
    
    # Return a JSON response
    return {
        "response": predict_data
    }



