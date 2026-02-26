from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path="Z:\langChain\9.DocumentLoder\docs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs=loader.lazy_load()

# print(len(docs))

# print(docs[0].page_content)
# print(docs[0].metadata)

for document in docs:
    print(document.metadata)