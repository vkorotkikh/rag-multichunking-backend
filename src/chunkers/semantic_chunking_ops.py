#!/usr/bin/env python3
"""
Semantic Chunking - Minimal, self-contained standalone script for chunking and quick
testing of various embedding models.

- Splits text into sentences
- Embeds the sentences with SentenceTransformer
- Groups similar sentences using cosine similarity with a greedy pass
- Emits character-based chunks with optional overlap

Quick Start:
    pip install sentence-transformers numpy
    python semantic_chunking_ops.py --file your.txt --model all-MiniLM-L6-v2 
    
You can also pipe in your text via stdin.
    cat textfile.txt | python semantic_chunking_ops.py --model all-MiniLM-L6-v2

"""

from typing import List, Dict, Any, Optional


import argparse
import sys, re
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
# from pydantic import BaseModel
# from langchain_core.documents import Document
# from langchain_core.chunking import RecursiveCharacterTextSplitter

# ------------------------------------------------------------
# 1. Data Structures
# ------------------------------------------------------------

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentChunk:
    doc_id: str
    chunk_index: str
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
# ------------------------------------------------------------
# 2. Sentence Splitting
# ------------------------------------------------------------

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    """
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

def split_into_sentences(text: str, min_len: int = 10) -> List[str]:
    """
    Split text into sentences using regex, filtering out short sentences.
    """
    parts = _SENTENCE_PATTERN.split(text.strip()) if text else []
    out: List[str] = []
    for s in parts:
        s = s.strip()
        if s and len(s) >= min_len:
            out.append(s)
    # fallback: if regex finds nothing treat the whole text as a single sentence
    return out or ([text.strip()] if text.strip() else [])

# ------------------------------------------------------------
# 3. Embedding And Similarity
# ------------------------------------------------------------

def load_model(model_name: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer model.
    """
    print(f"[semantic] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    embed_dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
    if embed_dim:
        print(f"[semantic] Model dimension: {dim}")
        
    return model

def encode_sentences(model: SentenceTransformer, sentences: List[str]) -> np.ndarray:
    """
    Encode sentences using SentenceTransformer.
    """
    if not sentences:
        return np.empty((0, 0), dtype=np.float32)
    
    embed = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)
    embed = embed.astype(np.float32)
    # L2 normalize rows for cosine via dot product
    norms = np.linalg.norm(embed, axis=1, keepdims=True) + 1e-9
    return embed / norms

def cosine_sim_matrix(normed_embeddings: np.ndarray) -> np.ndarray:
    if normed_embeddings.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return normed_embeddings @ normed_embeddings.T

# ---------------------------- grouping logic ----------------------------------

def group_sentences_semantically(
    sentences: List[str],
    normed_embeddings: np.ndarray,
    similarity_threshold: float,
    chunk_size: int,
    min_sentences_per_chunk: int,
) -> List[List[int]]:
    n = len(sentences)
    if n == 0:
        return []

    sim = cosine_sim_matrix(normed_embeddings)
    groups: List[List[int]] = []
    used = set()

    for i in range(n):
        if i in used:
            continue
        current = [i]
        used.add(i)
        current_len = len(sentences[i])

        # greedy add by similarity to the seed 'i'
        order = np.argsort(-sim[i])  # descending
        for j in order:
            if j == i or j in used:
                continue
            if sim[i, j] < similarity_threshold:
                continue
            if current_len + len(sentences[j]) > chunk_size:
                continue
            current.append(j)
            used.add(j)
            current_len += len(sentences[j])

        current.sort()
        if len(current) < min_sentences_per_chunk and groups:
            groups[-1].extend(current)
            groups[-1] = sorted(groups[-1])
        else:
            groups.append(current)

    return groups