from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df=pd.read_csv("realistic_restaurant_reviews.csv")

embeddings=OllamaEmbeddings(model="nomic-embed-text:latest")

# Create a Chroma vector store

db_location="./chroma_db"

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

add_documents = not os.path.exists(db_location)



if add_documents:
    documents=[]
    ids=[]

    for i,row in df.iterrows():
        documents.append(Document(
            page_content=row["review"],
            metadata={"date":row["Date"],"rating":row["Rating"]}
            ))
        ids.append(str(i))


    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

