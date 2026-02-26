from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('Z:\langChain\9.DocumentLoder\docs\Advanced_Production_LangChain_RAG_Guide.pdf')

docs = loader.load()
print(docs)
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)