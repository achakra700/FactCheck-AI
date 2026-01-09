"""
Evidence Retrieval with Pathway + Open-Source Embeddings.
Uses SentenceTransformers (MiniLM) instead of OpenAI embeddings.
"""

from typing import List, Dict, Tuple
import re

try:
    import pathway as pw
    from pathway.stdlib.ml.index import KNNIndex
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False
    print("⚠️  Pathway not installed. Install with: pip install pathway")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

# Load open-source embedding model (lightweight and fast)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if EMBEDDINGS_AVAILABLE:
    embedder_model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Loaded embedding model: {MODEL_NAME}")
else:
    embedder_model = None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    if not EMBEDDINGS_AVAILABLE or embedder_model is None:
        raise RuntimeError("sentence-transformers not available")
    return embedder_model.encode(texts, show_progress_bar=False).tolist()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def build_vector_index(novel_text: str):
    """
    Build a Pathway vector index with open-source embeddings.
    
    Args:
        novel_text: Full novel text
    
    Returns:
        Pathway KNN index or fallback dictionary
    """
    chunks = chunk_text(novel_text)
    
    if not PATHWAY_AVAILABLE or not EMBEDDINGS_AVAILABLE:
        # Fallback to simple dictionary
        print("Using fallback index (no Pathway/embeddings)")
        return {"chunks": chunks, "fallback": True}
    
    try:
        # Generate embeddings
        embeddings = embed_texts(chunks)
        
        # Create Pathway table
        table = pw.debug.table_from_rows(
            rows=[(i, chunks[i], embeddings[i]) for i in range(len(chunks))],
            schema=pw.schema_from_types(id=int, text=str, embedding=list)
        )
        
        # Build KNN index
        index = KNNIndex(
            table=table,
            data_column="embedding",
            metadata_columns=["text"],
        )
        
        print(f"✅ Built Pathway index with {len(chunks)} chunks")
        return index
        
    except Exception as e:
        print(f"⚠️  Pathway indexing failed: {e}")
        print("Falling back to simple search")
        return {"chunks": chunks, "fallback": True}


def search_index(index, query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Search the index for relevant chunks.
    
    Args:
        index: Pathway index or fallback dict
        query: Search query
        k: Number of results
    
    Returns:
        List of result dictionaries with 'text' key
    """
    # Check if fallback mode
    if isinstance(index, dict) and index.get("fallback"):
        # Simple keyword search
        chunks = index["chunks"]
        query_words = set(query.lower().split())
        scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for word in query_words if word in chunk_lower)
            if score > 0:
                scores.append((score, i, chunk))
        
        scores.sort(reverse=True)
        return [{"text": text} for _, _, text in scores[:k]]
    
    # Pathway mode
    try:
        query_embedding = embed_texts([query])[0]
        results = index.query(query_embedding, k=k)
        return [{"text": r["text"]} for r in results]
    except Exception as e:
        print(f"Search error: {e}")
        return []
