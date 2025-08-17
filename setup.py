"""
Setup script for RAG Multi-Chunking Backend.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-multichunking-backend",
    version="1.0.0",
    author="RAG Development Team",
    author_email="dev@rag-backend.com",
    description="A comprehensive RAG backend with multiple chunking strategies and vector stores",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/rag-multichunking-backend",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "semantic": [
            "spacy>=3.7.0",
            "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-backend=src.cli:main",  # Future CLI interface
        ],
    },
    keywords=[
        "rag", "retrieval", "augmented", "generation", "chunking", 
        "embeddings", "vector", "database", "ai", "nlp", "openai",
        "pinecone", "weaviate", "reranking", "langchain"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/rag-multichunking-backend/issues",
        "Source": "https://github.com/your-org/rag-multichunking-backend",
        "Documentation": "https://rag-multichunking-backend.readthedocs.io",
    },
)


