"""
src/tutoring/document_processor.py
Carga, segmenta y recupera chunks de documentos de estudio (PDF/TXT).
Recuperación mediante TF-IDF simplificado.
"""

import math
import logging
import re
from collections import Counter
from config.settings import DOCUMENT_CHUNK_SIZE_WORDS

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Carga documentos y provee recuperación de contexto relevante."""

    def __init__(self):
        self.chunks: list[str] = []
        self._tfidf: list[dict[str, float]] = []

    def load_from_file(self, path: str) -> int:
        """
        Carga un documento PDF o TXT y lo segmenta en chunks.

        Returns:
            Número de chunks indexados.
        """
        text = self._read_file(path)
        if not text:
            logger.warning("Documento vacío o no legible: %s", path)
            return 0

        self.chunks = self._split_into_chunks(text, DOCUMENT_CHUNK_SIZE_WORDS)
        self._tfidf = self._build_tfidf(self.chunks)
        logger.info("Documento cargado: %d chunks de ~%d palabras.", len(self.chunks), DOCUMENT_CHUNK_SIZE_WORDS)
        return len(self.chunks)

    def load_from_text(self, text: str) -> int:
        """Carga texto plano directamente (útil para tests)."""
        self.chunks = self._split_into_chunks(text, DOCUMENT_CHUNK_SIZE_WORDS)
        self._tfidf = self._build_tfidf(self.chunks)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Devuelve los top_k chunks más relevantes para la consulta.

        Args:
            query: Pregunta en texto libre.
            top_k: Número máximo de chunks a retornar.

        Returns:
            Lista de strings con los chunks más relevantes.
        """
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        scores = []
        for i, chunk_vec in enumerate(self._tfidf):
            score = sum(chunk_vec.get(t, 0.0) for t in query_tokens)
            scores.append((score, i))

        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:top_k] if _ > 0]

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _read_file(self, path: str) -> str:
        if path.lower().endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                logger.error("pdfplumber no instalado. Ejecutar: pip install pdfplumber")
                return ""
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
        words = text.split()
        return [
            " ".join(words[i: i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-záéíóúüñ]+", text.lower())

    @staticmethod
    def _build_tfidf(chunks: list[str]) -> list[dict[str, float]]:
        n = len(chunks)
        tokenized = [DocumentProcessor._tokenize(c) for c in chunks]
        df: Counter = Counter()
        for tokens in tokenized:
            df.update(set(tokens))

        result = []
        for tokens in tokenized:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {
                t: (count / total) * math.log((n + 1) / (df[t] + 1))
                for t, count in tf.items()
            }
            result.append(vec)
        return result
