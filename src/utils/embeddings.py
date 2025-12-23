import numpy as np
import aiohttp
import logging
from typing import List, Optional
from src.utils.logger import setup_logger

logger = logging.getLogger("Deep Research at Home")

class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls"""
    def __init__(self, max_size=10000000):
        self.cache = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def get(self, text_key):
        key = hash(text_key[:2000])
        result = self.cache.get(key)
        if result is not None:
            self.hit_count += 1
        return result

    def set(self, text_key, embedding):
        key = hash(text_key[:2000])
        self.cache[key] = embedding
        self.miss_count += 1
        if len(self.cache) > self.max_size:
            self.cache.pop(next(iter(self.cache)))

    def stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
        }

class TransformationCache:
    """Simple cache for transformed embeddings"""
    def __init__(self, max_size=2500000):
        self.cache = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def get(self, text, transform_id):
        key = f"{hash(text[:2000])}_{hash(str(transform_id))}"
        result = self.cache.get(key)
        if result is not None:
            self.hit_count += 1
        return result

    def set(self, text, transform_id, transformed_embedding):
        key = f"{hash(text[:2000])}_{hash(str(transform_id))}"
        self.cache[key] = transformed_embedding
        self.miss_count += 1
        if len(self.cache) > self.max_size:
            self.cache.pop(next(iter(self.cache)))

    def stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
        }

async def get_embedding(text: str, model: str, api_base: str, cache: EmbeddingCache) -> Optional[List[float]]:
    """Get embedding for a text string using the configured embedding model with caching."""
    if not text or not text.strip():
        return None

    text = text[:2000].replace(":", " - ")
    cached = cache.get(text)
    if cached is not None:
        return cached

    try:
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:
            payload = {"model": model, "input": text}
            async with session.post(f"{api_base}/v1/embeddings", json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = None
                    if isinstance(result, dict) and "data" in result:
                        data = result.get("data") or []
                        if data: embedding = data[0].get("embedding")
                    if embedding is None and "embedding" in result:
                        embedding = result.get("embedding")
                    if embedding:
                        cache.set(text, embedding)
                        return embedding
                else:
                    logger.warning(f"Embedding request failed with status {response.status}")
        return None
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

async def apply_semantic_transformation(embedding, transformation):
    """Apply semantic transformation to an embedding"""
    if not transformation or not embedding:
        return embedding
    try:
        embedding_array = np.array(embedding)
        if not isinstance(transformation, dict) or "matrix" not in transformation:
            return embedding
        transform_matrix = np.array(transformation["matrix"])
        if np.isnan(embedding_array).any() or np.isnan(transform_matrix).any():
            return embedding
        transformed = np.dot(embedding_array, transform_matrix)
        norm = np.linalg.norm(transformed)
        if norm > 1e-10:
            return (transformed / norm).tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error applying semantic transformation: {e}")
        return embedding
