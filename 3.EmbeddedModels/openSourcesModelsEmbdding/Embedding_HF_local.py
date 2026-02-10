from langchain_huggingface import HuggingFaceEmbeddings
embedding= HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L12-v2")

text= "What is the capital of India?"
vector= embedding.embed_query(text)
print(str(vector))


documents = [
    "The capital of France is Paris.",
    "The capital of India is New Delhi.",
    "The capital of Germany is Berlin."
]
result=embedding.embed_documents(documents)
print(str(result))