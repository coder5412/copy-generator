import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load PDFs
DOCS_PATH = "docs"
documents = []

for file in os.listdir(DOCS_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
        documents.extend(loader.load())

print("✅ Documents loaded")

# Create embeddings
embeddings = OpenAIEmbeddings()

# Vector DB
db = FAISS.from_documents(documents, embeddings)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

print("🤖 RAG Bot Ready (type 'exit' to quit)\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    try:
        result = qa.run(query)

        # Basic guardrail
        if not result or "I don't know" in result:
            print("❌ I can only answer from provided documents.\n")
        else:
            print(f"\n✅ Answer: {result}\n")

    except Exception as e:
        print(f"Error: {e}")