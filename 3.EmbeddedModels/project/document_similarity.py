from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding= HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L12-v2")

documents = [
    "virat kohli is the best batsman in the world",
    "sachin tendulkar is the god of cricket",
    "messi is the best football player in the world",
    "rohit sharma is known for his elegant batting and record breaking centuries",
    "the capital of india is delhi",
    "the capital of france is paris",
    "the capital of germany is berlin"
]

query = "who is the best batsman in the world?"
query1="tell me about rohit sharma"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
query_embedding1 = embedding.embed_query(query1)

scores=cosine_similarity([query_embedding], doc_embeddings)[0]
scores1=cosine_similarity([query_embedding1], doc_embeddings)[0]


index, score=sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]
index1, score1=sorted(list(enumerate(scores1)),key = lambda x:x[1])[-1]

print("Query:", query)
print(documents[index])
print("Similarity Scores:", score)
print("------------------------------------------------")
print("Query1:", query1)
print(documents[index1])
print("Similarity Scores for Query1:", score1)

