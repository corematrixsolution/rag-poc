import os
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.prompts import PromptTemplate

from app.embeddings_e5 import E5Embeddings

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_poc")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def build_llm():
    return OllamaLLM(model=OLLAMA_MODEL)


def build_retriever(k: int = 4):
    embeddings = E5Embeddings()

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None
    )

    vs = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings
    )

    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )


def build_qa_chain():
    llm = build_llm()
    retriever = build_retriever(k=4)

    template = """
You are a precise assistant.

Use ONLY the provided context to answer the question.

If the answer is not in the context, say:
"I don't know based on the provided documents."

Question:
{question}

Context:
{context}

Answer:
"""

    prompt = PromptTemplate.from_template(template)

    def run_chain(question: str):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            f"[SOURCE: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

        formatted_prompt = prompt.format(
            question=question,
            context=context
        )

        answer = llm.invoke(formatted_prompt)

        return {
            "result": answer,
            "source_documents": docs
        }

    return run_chain