"""
src/semantic.py
Optional semantic text similarity reranker utilizing sentence-transformers.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Optional semantic text similarity calculator for candidate reranking.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", enabled: bool = False):
        self.enabled = enabled
        self.model_name = model_name
        self._model = None

    def compute_similarity(self, text_a: str, text_b: str) -> Optional[float]:
        """
        Compute cosine similarity between two text strings if enabled.
        """
        if not self.enabled:
            return None

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer, util
                self._model = SentenceTransformer(self.model_name)
                self._util = util
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers model '{self.model_name}': {e}")
                self.enabled = False
                return None

        try:
            emb_a = self._model.encode(text_a, convert_to_tensor=True)
            emb_b = self._model.encode(text_b, convert_to_tensor=True)
            sim = float(self._util.cos_sim(emb_a, emb_b).item())
            return max(0.0, min(1.0, round(sim, 4)))
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return None
