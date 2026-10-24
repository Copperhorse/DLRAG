"""Chunk quality filtering.

Scores each chunk for quality and usefulness, filtering out metadata, TOC,
references, and other non-content chunks. Wired into IngestionPipeline via
`filter_chunks()`, which actually drops rejected chunks rather than just
tagging them.
"""

from __future__ import annotations

import logging
import re
from typing import Any


class ChunkEvaluator:
    """
    Evaluates chunks for quality and usefulness.
    Filters out metadata, TOC, references, and other non-content chunks.
    """

    def __init__(self, log_level=logging.INFO):
        self.metadata_excludes: set[str] = {
            "table of contents",
            "contents",
            "references",
            "reference",
            "bibliography",
            "bibliographic",
            "further reading",
            "acknowledgment",
            "acknowledgement",
            "appendix",
            "author",
            "index",
            "funding",
            "glossary",
        }

        self.content_prefix_excludes = [
            "figure:",
            "fig.",
            "[removed: corrupted latex content]",
        ]

        self.content_anywhere_excludes = [
            "copyright",
            "all rights reserved",
            "[removed: corrupted latex content]",
        ]

        self.patterns = {
            "toc_dots": re.compile(r"\.{3,}\s*\d{1,4}(?:\s*\|)?$"),
            "url": re.compile(r"https?://|www\."),
            "doi": re.compile(r"doi:10\.\d{4,9}/"),
            "code_block": re.compile(r"def\s+\w+\(|import\s+\w+|class\s+\w+:|const\s+\w+\s*=|function\s+\w+\("),
        }

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _alpha_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        alpha = sum(1 for c in text if c.isalpha())
        return alpha / max(1, len(text))

    def _is_useful_table(self, text: str, metadata: dict[str, Any]) -> bool:
        lines = text.split("\n")
        table_rows = sum(1 for line in lines if "|" in line)
        header = (metadata.get("section_header") or "").lower()
        if any(x in header for x in ["table of contents", "contents"]):
            return False
        if table_rows > 5 and len(lines) > 0 and (table_rows / len(lines)) > 0.4:
            return True
        return False

    def _is_table_of_contents(self, text: str) -> bool:
        text_lower = text.lower()
        lines = text.splitlines()
        stripped_lines = [line.strip().rstrip("|").strip() for line in lines]
        toc_patterns = sum(1 for line in stripped_lines if self.patterns["toc_dots"].search(line))
        toc_keywords = "table of contents" in text_lower[:400]
        return toc_patterns > 3 or toc_keywords

    def evaluate_chunk(self, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        text = (content or "").strip()
        text_lower = text.lower()

        if len(text) < 30 or not text_lower:
            return self._result(False, "Too short/Empty", "noise", 0)

        header_text = str(metadata.get("section_header") or "").lower()

        if any(exclude in header_text for exclude in self.metadata_excludes):
            return self._result(False, f"Header filter: {header_text}", "metadata", 0)

        if any(text_lower.startswith(ex) for ex in self.content_prefix_excludes):
            return self._result(False, "Forbidden prefix", "other", 0)

        if len(text) < 150:
            if self.patterns["url"].search(text_lower) or self.patterns["doi"].search(text_lower):
                return self._result(False, "Citation/Link chunk", "metadata", 0)

        if any(ex in text_lower[:400] for ex in self.content_anywhere_excludes):
            return self._result(False, "Metadata keyword detected", "metadata", 0)

        if self._is_table_of_contents(text):
            return self._result(False, "Detected Table of Contents", "toc", 0)

        # Trust the chunker's own content_type where it's already decisive
        # (code/table were split out deliberately) instead of re-deriving it
        # with a second, independent classifier that could disagree.
        known_type = str(metadata.get("content_type") or "").lower()

        if known_type == "code":
            return self._result(True, "Source code block", "code", 85)

        if known_type == "table":
            if self._is_useful_table(text, metadata):
                return self._result(True, "Useful data table", "table", 75)
            return self._result(False, "Low-value/malformed table", "table", 20)

        if self.patterns["code_block"].search(text):
            return self._result(True, "Source code block (inline)", "code", 85)

        alpha_ratio = self._alpha_ratio(text)

        if alpha_ratio < 0.30:
            if self._is_useful_table(text, metadata):
                return self._result(True, "Useful data table", "table", 75)
            return self._result(False, f"Low alpha ({alpha_ratio:.2%})", "metadata", 20)

        quality_score = self._calculate_quality_score(text, metadata, alpha_ratio)
        return self._result(True, "OK", "main_content", quality_score)

    def _result(self, should_use: bool, reason: str, c_type: str, score: float) -> dict[str, Any]:
        if not should_use:
            self.logger.debug(f"Excluded: {reason}")
        return {
            "should_use": should_use,
            "reason": reason,
            "content_type": c_type,
            "quality_score": score,
        }

    def _calculate_quality_score(self, text: str, metadata: dict[str, Any], alpha_ratio: float) -> float:
        score = 50
        text_len = len(text)

        if 400 <= text_len <= 1200:
            score += 25
        elif 200 <= text_len < 400 or 1200 < text_len <= 2000:
            score += 10

        if alpha_ratio > 0.75:
            score += 15

        if text[0].isupper() and text.strip().endswith((".", "!", "?")):
            score += 10

        header = metadata.get("section_header") or ""
        if len(str(header)) > 3:
            score += 5

        return float(min(100, score))

    def evaluate_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        self.logger.info(f"Evaluating {len(chunks)} chunks...")

        total = len(chunks)
        results = []

        for c in chunks:
            content = c.get("content", "")
            metadata = c.get("metadata", {})
            res = self.evaluate_chunk(content, metadata)
            results.append(res)

        evaluated_chunks = []
        quality_scores = []
        rejection_reasons: dict[str, int] = {}

        for i, eval_res in enumerate(results):
            new_chunk = chunks[i].copy()
            new_chunk["evaluation"] = eval_res
            evaluated_chunks.append(new_chunk)

            if eval_res["should_use"]:
                quality_scores.append(eval_res["quality_score"])
            else:
                reason = eval_res["reason"]
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        accepted = len(quality_scores)
        avg_quality = sum(quality_scores) / accepted if accepted > 0 else 0

        return {
            "total_chunks": total,
            "accepted_chunks": accepted,
            "rejected_chunks": total - accepted,
            "acceptance_rate": (accepted / total * 100) if total > 0 else 0,
            "rejection_reasons": rejection_reasons,
            "average_quality_score": round(avg_quality, 2),
            "evaluated_chunks": evaluated_chunks,
        }

    def filter_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only the chunks that pass evaluation -- actually drops the
        rejected ones, rather than leaving them in with should_use=False."""
        stats = self.evaluate_chunks(chunks)
        return [c for c in stats["evaluated_chunks"] if c["evaluation"]["should_use"]]
