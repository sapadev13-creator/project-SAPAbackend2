"""
Runtime log untuk proses Excel / Excel profile — progress per baris & trait.
Log ditampilkan di konsol (format mirip uvicorn) dan file sapadev/tmp/app.log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.logger_setup import logger
from sapa_api.config import OCEAN_LABELS, OCEAN_TRAITS

_MODE_LABEL = {
    "excel": "Excel",
    "excel_profile": "Excel profile",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _label(mode: str) -> str:
    return _MODE_LABEL.get(mode, mode)


@dataclass
class ExcelRuntimeLog:
    mode: str  # excel | excel_profile
    filename: str = ""
    total_rows: int = 0
    processed_rows: int = 0
    started_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def _tag(self) -> str:
        return f"[{_label(self.mode)}]"

    def start(self, total: int, filename: str = "") -> None:
        self.total_rows = total
        self.filename = filename
        self.started_at = _utc_now()
        logger.info(
            "%s Mulai proses | file=%s | total_baris=%s",
            self._tag,
            filename or "-",
            total,
        )

    def log_row_start(self, row_index: int) -> None:
        current = self.processed_rows + 1
        logger.info(
            "%s Baris %s/%s - memproses prediksi traits (O, C, E, A, N)...",
            self._tag,
            current,
            self.total_rows,
        )

    def log_row(self, row_index: int, pred: dict[str, Any], row_seconds: float) -> None:
        self.processed_rows += 1
        scores = pred.get("prediction_adjusted") or {}
        traits = {t: round(float(scores.get(t, 0)), 2) for t in OCEAN_TRAITS}
        persona_list = pred.get("personality_profile") or []
        persona = persona_list[0] if persona_list else ""

        trait_parts = " ".join(
            f"{t}={traits.get(t, 0):.2f}" for t in OCEAN_TRAITS
        )

        entry = {
            "row_index": row_index,
            "progress": f"{self.processed_rows}/{self.total_rows}",
            "dominant_trait": pred.get("dominant_trait"),
            "text_intent": pred.get("text_intent"),
            "traits": traits,
            "persona": persona,
            "seconds": round(row_seconds, 3),
        }
        self.rows.append(entry)

        logger.info(
            "%s Baris %s/%s selesai | dom=%s | %s | persona=%s | %.2fs",
            self._tag,
            self.processed_rows,
            self.total_rows,
            entry["dominant_trait"],
            trait_parts,
            persona or "-",
            row_seconds,
        )

    def log_chunk(self, chunk_start: int, chunk_end: int) -> None:
        logger.info(
            "%s Batch inferensi model baris %s-%s dari %s",
            self._tag,
            chunk_start,
            chunk_end,
            self.total_rows,
        )

    def finish(self, batch_stats: dict | None = None) -> dict[str, Any]:
        self.finished_at = _utc_now()
        if batch_stats and batch_stats.get("seconds") is not None:
            self.duration_seconds = float(batch_stats["seconds"])
        rps = (
            self.processed_rows / self.duration_seconds
            if self.duration_seconds > 0
            else 0
        )
        logger.info(
            "%s Selesai - %s/%s baris | durasi=%.2fs | %.2f baris/detik",
            self._tag,
            self.processed_rows,
            self.total_rows,
            self.duration_seconds,
            rps,
        )
        return self.to_dict(batch_stats)

    def to_dict(self, processing: dict | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mode": self.mode,
            "filename": self.filename,
            "total_rows": self.total_rows,
            "processed_rows": self.processed_rows,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "trait_labels": {t: OCEAN_LABELS[t] for t in OCEAN_TRAITS},
            "rows": self.rows,
        }
        if processing:
            out["processing"] = processing
        return out
