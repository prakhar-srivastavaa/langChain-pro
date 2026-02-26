from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path="Z:\langChain\9.DocumentLoder\docs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs=loader.load()

print(len(docs))

print(docs[0].page_content)
