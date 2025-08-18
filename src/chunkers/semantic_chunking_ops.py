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

# ------------------------------------------------------------
# 4. Grouping Logic
# ------------------------------------------------------------
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

# ---------------------------- chunk assembly ----------------------------------

def index_of_sentence(text: str, sentence: str, start_at: int) -> int:
    idx = text.find(sentence, start_at)
    if idx >= 0:
        return idx
    # fallback: crude search without start hint
    return text.find(sentence)

def create_chunks_from_groups(
    groups: List[List[int]],
    sentences: List[str],
    original_text: str,
    doc: Document,
    chunk_overlap: int,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    cursor = 0  # where to start searching for the next group's first sentence

    for ci, idxs in enumerate(groups):
        if not idxs:
            continue
        group_sents = [sentences[k] for k in idxs]
        chunk_text = " ".join(group_sents)

        start = index_of_sentence(original_text, group_sents[0], cursor)
        if start < 0:
            start = cursor
        end = min(len(original_text), start + len(chunk_text))
        cursor = end  # advance cursor for next search

        chunks.append(DocumentChunk(
            doc_id=doc.id,
            chunk_index=ci,
            content=chunk_text,
            start_char=start,
            end_char=end,
            metadata={**doc.metadata}
        ))

    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = apply_overlap(chunks, chunk_overlap)

    return chunks

def apply_overlap(chunks: List[DocumentChunk], chunk_overlap: int) -> List[DocumentChunk]:
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = out[-1]
        curr = chunks[i]
        overlap_text = prev.content[-chunk_overlap:]
        merged = (overlap_text + " " + curr.content).strip()
        out.append(DocumentChunk(
            doc_id=curr.doc_id,
            chunk_index=curr.chunk_index,
            content=merged,
            start_char=curr.start_char,  # keep original approx
            end_char=curr.end_char,
            metadata=curr.metadata,
        ))
    return out

# ------------------------------------------------------------
# 5. Utilities
# ------------------------------------------------------------

def recommended_model() -> Dict[str, List[str]]:
    return {
        "lightweight": ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
        "balanced": ["all-mpnet-base-v2"],
        "retrieval/qa": ["msmarco-distilbert-base-v4", "multi-qa-mpnet-base-dot-v1"],
        "multilingual": ["paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-multilingual-mpnet-base-v2"],
    }

# ------------------------------------------------------------
# 6. Alternative Engines 
# ------------------------------------------------------------

def semantic_chunk_langchain(
    text:str,
    *,
    model_name: str = "sentence-transformers/msmarco-distilbert-base-v4",
    breakpoint_type: str = "percentile",
    breakpoint_amount: float = 0.5,
    buffer_size: int = 1,
    min_chunk_size: int = 0,
    chunk_overlap: int = 200,
    ) -> List[DocumentChunk]:
    """Use LangChain's SemanticChunker with a SentenceTransformer-backed embedder.
    Requires: langchain-experimental, langchain-community, sentence-transformers.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception as e:
        raise RuntimeError(
            "LangChain SemanticChunker is not available. "
            "Install: pip install langchain-experimental langchain-community"
        ) from e
    embed = HuggingFaceEmbeddings(model_name = model_name)
    splitter = SemanticChunker(embeddings=embed,
                               breakpoint_threshold=breakpoint_amount,
                               chunk_size=chunk_size,
                               chunk_overlap=chunk_overlap,
                               buffer_size=buffer_size,
                               min_chunk_size=(min_chunk_size if min_chunk_size > 0 else None),
                               )
    pieces: List[str] = splitter.split_text(text)
    
    # Map back to DocumentChunk list using sequential search to get char ranges
    chunks: List[DocumentChunk] = []
    cursor = 0
    for i, t in enumerate(pieces):
        start = text.find(t, cursor)
        if start < 0:
            start = cursor
        end = start + len(t)
        cursor = end
        chunks.append(DocumentChunk(doc_id=doc_id, chunk_index=i, content=t, start_char=start, end_char=end, metadata={**meta}))

    if chunks_overlap > 0 and len(chunks) > 1:
        chunks = apply_overlap(chunks, chunks_overlap)
    return chunks
    
# ---------------------------- CLI --------------------------------------------

def _read_stdin() -> str:
    if sys.stdin and not sys.stdin.isatty():
        return sys.stdin.read()
    return ""

def main():
    ap = argparse.ArgumentParser(description="Semantic chunking (functional)")
    ap.add_argument("--file", type=str, help="Input text file", default=None)
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--engine", type=str, choices=["native", "langchain", "llamaindex"], default="native",
                    help="Chunking engine: native (this script), langchain, or llamaindex")

    # Native engine params
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--min-sentences", type=int, default=2)

    # LangChain params
    ap.add_argument("--breakpoint-type", type=str, default="percentile",
                    help="percentile | stdev | gradient | interquartile")
    ap.add_argument("--breakpoint", type=int, default=95, help="Threshold amount for breakpoint-type")
    ap.add_argument("--buffer-size", type=int, default=1)
    ap.add_argument("--min-chars", type=int, default=0, help="Minimum chars per chunk (LangChain only)")

    # Utility flags
    ap.add_argument("--print", dest="do_print", action="store_true", help="Print chunk contents")
    ap.add_argument("--list-models", action="store_true", help="Show recommended models")
    args = ap.parse_args()

    if args.list_models:
        print("Recommended models:")
        for k, vs in recommended_models().items():
            print(f"  {k}: {', '.join(vs)}")
        return

    text = ""
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = _read_stdin()

    if not text.strip():
        print("No input text. Provide --file or pipe text via stdin.")
        return

    if args.engine == "native":
        chunks = semantic_chunk(
            text,
            model_name=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            similarity_threshold=args.threshold,
            min_sentences_per_chunk=args.min_sentences,
        )
    elif args.engine == "langchain":
        chunks = semantic_chunk_langchain(
            text,
            model_name=args.model,
            breakpoint_type=args.breakpoint_type,
            breakpoint_amount=args.breakpoint,
            buffer_size=args.buffer_size,
            min_chunk_size=args.min_chars,
            chunk_overlap=args.overlap,
        )
    else:  # llamaindex
        chunks = semantic_chunk_llamaindex(
            text,
            model_name=args.model,
            breakpoint_percentile_threshold=args.breakpoint,
            buffer_size=args.buffer_size,
            chunk_overlap=args.overlap,
        )

    print(f"[semantic] Produced {len(chunks)} chunks
")
    for c in chunks:
        preview = (c.content[:120] + "…") if len(c.content) > 120 else c.content
        print(f"#{c.chunk_index:02d} chars[{c.start_char}:{c.end_char}] len={len(c.content)}
  {preview}
")

    if args.do_print:
        for c in chunks:
            print("
" + "-"*80)
            print(f"CHUNK #{c.chunk_index} ({len(c.content)} chars)")
            print(c.content)

if __name__ == "__main__":
    main()
