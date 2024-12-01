from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

import pandas as pd
from tqdm import tqdm
from collections import Counter
from qdrant_client import QdrantClient

class QdrantVectorStorePredictor:
    def __init__(self, embeddings, retrieval_top_k, reranker_top_k, db_url, api_key, collection_name):
        self.embeddings = embeddings
        self.retrieval_top_k = retrieval_top_k
        self.reranker_top_k = reranker_top_k
        self.db_url = db_url
        self.api_key = api_key
        self.collection_name = collection_name
        self.compression_retriever=None
        self.fitted=False
    
    def connect_vector_db(self):
        sparse_embeddings = FastEmbedSparse(
            model_name="Qdrant/bm25"
        )

        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            sparse_embedding=sparse_embeddings,
            url=self.db_url,
            prefer_grpc=True,
            api_key=self.api_key,
            collection_name=self.collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
        )

        qdrant_retriever = qdrant.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": self.retrieval_top_k},
        )

        compressor = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2",
            top_n=self.reranker_top_k    
        )
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=qdrant_retriever
        )

        self.fitted = True

    def delete_collection(self):
        qdrant_client = QdrantClient(
            url=self.db_url, 
            api_key=self.api_key,
        )

        qdrant_client.delete_collection(collection_name=self.collection_name)

    def fit(
        self,
        df:pd.DataFrame,
        transaction_description,
        sensitive_label:str,
        sensitive_type:str,
        transaction_type:str
    ):
        # 1. Prepare langchain documents
        langchain_documents = []
        for index, row in tqdm(df.iterrows(), desc="Preparing chunks", total=len(df)):
            transaction_description = row['transaction_description']
            sensitive_type = row['sensitive_type']
            transaction_type = row['transaction_type']
            sensitive_label = row['sensitive_label']

            document = Document(
                page_content=transaction_description,
                metadata={
                    "sensitive_type":sensitive_type,
                    "transaction_type":transaction_type,
                    "sensitive_label":sensitive_label
                }
            )

            langchain_documents.append(document)

        # -------------------------------------------------------------------------
        # 2. Ingest to Vector Database: Qdrant
        # -------------------------------------------------------------------------
        print("Ingest to Vector Database Start.")
        url = self.db_url
        api_key = self.api_key

        sparse_embeddings = FastEmbedSparse(
            model_name="Qdrant/bm25"
        )

        qdrant = QdrantVectorStore.from_documents(
            langchain_documents,
            embedding=self.embeddings,
            sparse_embedding=sparse_embeddings,
            url=url,
            prefer_grpc=True,
            api_key=api_key,
            collection_name=self.collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
        )

        qdrant_retriever = qdrant.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": self.retrieval_top_k},
        )

        compressor = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2",
            top_n=self.reranker_top_k    
        )
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=qdrant_retriever
        )
        print("Ingest to Vector Database End.")

        self.fitted = True


    def retrieve(self, query:str):
        if self.fitted:
            retrieved_docs = self.compression_retriever.invoke(query)
            return retrieved_docs
        else:
            raise ValueError("Please call the 'fit' function or 'connect_vector_db' first")

    def predict(self, query:str, explain=False):
        top_docs = self.retrieve(query=query)

        label_list = [doc.metadata['sensitive_label'] for doc in top_docs]

        label_counts = Counter(label_list)

        # Find the label with the highest count
        most_common_label, highest_count = label_counts.most_common(1)[0]

        probability_score = highest_count / len(label_list)

        if explain == False:
            return {
                'prediction':most_common_label,
                'probability_score':probability_score
            }
        else:
            relevant_top_docs = [doc for doc in top_docs if doc.metadata['sensitive_label'] == most_common_label]

            text_label = "sensitive" if most_common_label == "sensitive" else "not senstitive"

            explain_text = f"The transaction description seems having similar meaning with the **{text_label}** descriptions below: \n\n"

            for i, doc in enumerate(relevant_top_docs, start=1):
                sensitive_desc = f"{i}.**Transaction description:** {doc.page_content}\n  **Sensitive Type:** {doc.metadata['sensitive_type']}\n  **Transaction Type:** {doc.metadata['transaction_type']}\n\n"
                explain_text += sensitive_desc

            return {
                'prediction':most_common_label,
                'probability_score':probability_score,
                'explain':explain_text
            }
