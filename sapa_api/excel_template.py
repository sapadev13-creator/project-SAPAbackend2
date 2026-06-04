"""
Validasi & template upload Excel untuk prediksi OCEAN.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import HTTPException

EXCEL_TEMPLATE_FILENAME = "sapa_excel_template.xlsx"
REQUIRED_COLUMNS = ("text",)
TEMPLATE_ERROR_MESSAGE = (
    "Template salah. Silahkan unduh template yang sesuai dan isi kolom 'text'."
)
TEMPLATE_DOWNLOAD_PATH = "/predict/excel/template"

EXAMPLE_ROWS = (
    "Saya menikmati acara sosial seperti seminar dan komunitas.",
    "Perasaan yang didengarkan terasa lebih ringan.",
)


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower()


def validate_excel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan struktur template benar. Return DataFrame dengan kolom 'text' standar.
    """
    if df is None or df.empty:
        raise _invalid_template(
            "File Excel kosong atau tidak dapat dibaca.",
            found_columns=[],
        )

    if len(df.columns) == 0:
        raise _invalid_template("File Excel tidak memiliki kolom.", found_columns=[])

    col_map = {_normalize_column_name(c): c for c in df.columns}
    found = list(df.columns.astype(str))

    missing = [c for c in REQUIRED_COLUMNS if c not in col_map]
    if missing:
        raise _invalid_template(
            TEMPLATE_ERROR_MESSAGE,
            found_columns=found,
            missing_columns=missing,
        )

    out = df.copy()
    out["text"] = out[col_map["text"]].astype(str)
    return out


def parse_excel_upload(file_bytes: bytes) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Baca bytes Excel → dataframe valid + daftar (row_index, text)."""
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception as exc:
        raise _invalid_template(
            "File tidak valid atau bukan format Excel (.xlsx).",
            found_columns=[],
        ) from exc

    df = validate_excel_dataframe(df)
    items: list[tuple[int, str]] = []
    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        if text and text.lower() != "nan":
            items.append((int(idx), text))
    return df, items


def build_template_excel_bytes() -> BytesIO:
    """Generate file template kosong + contoh baris."""
    buffer = BytesIO()
    df = pd.DataFrame({"text": list(EXAMPLE_ROWS)})
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Input")
    buffer.seek(0)
    return buffer


def _invalid_template(
    message: str,
    *,
    found_columns: list[str],
    missing_columns: list[str] | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "error": "invalid_template",
            "message": message,
            "template_download": TEMPLATE_DOWNLOAD_PATH,
            "required_columns": list(REQUIRED_COLUMNS),
            "found_columns": found_columns,
            "missing_columns": missing_columns or list(REQUIRED_COLUMNS),
        },
    )
