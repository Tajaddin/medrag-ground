"""Audit-logged RAG over OpenFDA drug labels with citation grounding + PII detection."""

from medrag_ground.audit import AuditEntry, AuditLog
from medrag_ground.embedders import Embedder, FastEmbedEmbedder, HashEmbedder, get_embedder
from medrag_ground.indexer import DrugDoc, Indexer, Retriever
from medrag_ground.llm import AnthropicClient, GroqClient, LLMClient, MockLLMClient
from medrag_ground.openfda import OpenFDAClient
from medrag_ground.pii import PIIDetection, PIIDetector
from medrag_ground.pipeline import MedRAG, RAGResponse

__version__ = "0.1.0"

__all__ = [
    # core
    "MedRAG",
    "RAGResponse",
    # building blocks
    "OpenFDAClient",
    "Indexer",
    "Retriever",
    "DrugDoc",
    "PIIDetector",
    "PIIDetection",
    "AuditLog",
    "AuditEntry",
    # embedders
    "Embedder",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "get_embedder",
    # llm
    "LLMClient",
    "MockLLMClient",
    "GroqClient",
    "AnthropicClient",
]
