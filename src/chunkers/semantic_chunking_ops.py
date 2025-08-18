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
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    