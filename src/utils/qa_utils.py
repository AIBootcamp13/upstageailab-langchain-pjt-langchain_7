from typing import List
from langchain_core.documents import Document
from langchain.prompts import load_prompt as _load_prompt


def format_docs_with_citations(docs: List[Document]) -> str:
    """Formats documents with citation numbers for the final answer."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        content = doc.page_content.replace('\n', ' ')
        source = doc.metadata.get("source", "N/A")
        formatted_docs.append(f"[{i+1}] {content} (Source: {source})")
    return "\n\n".join(formatted_docs)


def load_prompt(path: str):
    """A wrapper around LangChain's load_prompt function."""
    return _load_prompt(path)


def is_broad_question(question: str) -> bool:
    """A simple heuristic to check if a question is too broad."""
    broad_terms = ["tell me about", "what is", "explain"]
    return any(term in question.lower() for term in broad_terms)