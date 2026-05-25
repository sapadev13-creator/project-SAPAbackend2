import base64
import re
from io import BytesIO

import pandas as pd


def likert_to_percent(value, scale=5):
    if value is None:
        return 0
    return round((float(value) / scale) * 100, 2)


def build_excel_rows(results):
    rows = []
    for r in results:
        scores = r["prediction_adjusted"]
        rows.append({
            "text": re.sub(r"<.*?>", "", r.get("highlighted_text", "")),
            "O (%)": likert_to_percent(scores["O"]),
            "C (%)": likert_to_percent(scores["C"]),
            "E (%)": likert_to_percent(scores["E"]),
            "A (%)": likert_to_percent(scores["A"]),
            "N (%)": likert_to_percent(scores["N"]),
            "kepribadian": ", ".join(r.get("personality_profile", [])),
            "solusi": r.get("suggestion", ""),
        })
    return pd.DataFrame(rows)


def dataframe_to_excel_bytes(df_detail, profile_summary=None):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_detail.to_excel(writer, index=False, sheet_name="Detail")
        if profile_summary is not None:
            cols = len(df_detail.columns)
            start_row = len(df_detail) + 2

            def pad(values):
                return values + [""] * (cols - len(values))

            avg = profile_summary["average_ocean_likert"]
            summary_rows = [
                pad(["SUMMARY"]),
                pad(["Average O (%)", likert_to_percent(avg["O"])]),
                pad(["Average C (%)", likert_to_percent(avg["C"])]),
                pad(["Average E (%)", likert_to_percent(avg["E"])]),
                pad(["Average A (%)", likert_to_percent(avg["A"])]),
                pad(["Average N (%)", likert_to_percent(avg["N"])]),
                pad(["Dominant Trait", profile_summary["dominant_trait"]]),
                pad(["Conclusion", profile_summary["conclusion"]]),
                pad(["Suggestion", profile_summary["suggestion"]]),
                pad(["Total Text", profile_summary["total_text_analyzed"]]),
            ]
            pd.DataFrame(summary_rows, columns=df_detail.columns).to_excel(
                writer,
                index=False,
                header=False,
                sheet_name="Detail",
                startrow=start_row,
            )

    buffer.seek(0)
    return buffer


def excel_buffer_to_base64(buffer: BytesIO):
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
