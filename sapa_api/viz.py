import base64
from io import BytesIO

import matplotlib.pyplot as plt

from sapa_api.config import OCEAN_COLORS, OCEAN_LABELS, OCEAN_TRAITS
from sapa_api.persona import generate_global_conclusion


def ocean_to_bar_chart(avg_ocean):
    bar_chart = []
    for trait in OCEAN_TRAITS:
        value = avg_ocean.get(trait, 1)
        if value < 1 or value > 5:
            raise ValueError(
                f"Invalid Likert value for {trait}: {value}. Expected 1–5"
            )
        percent = round(((value - 1) / 4) * 100, 1)
        bar_chart.append({
            "trait": trait,
            "label": OCEAN_LABELS[trait],
            "value": percent,
            "raw_likert": round(value, 2),
            "color": OCEAN_COLORS[trait],
        })
    return bar_chart


def aggregate_ocean_profile(results):
    if not results:
        return None

    total = {t: 0.0 for t in OCEAN_TRAITS}
    count = 0
    for r in results:
        ocean = r.get("prediction_adjusted")
        if not ocean:
            continue
        for t in OCEAN_TRAITS:
            total[t] += ocean.get(t, 0)
        count += 1

    if count == 0:
        return None

    avg = {t: round(total[t] / count, 3) for t in OCEAN_TRAITS}
    dominant = max(avg, key=avg.get)
    conclusion, suggestion = generate_global_conclusion(avg, dominant)
    bar_chart = ocean_to_bar_chart(avg)

    return {
        "average_ocean_likert": avg,
        "average_ocean_percent": {
            t: round(((avg[t] - 1) / 4) * 100, 1) for t in OCEAN_TRAITS
        },
        "dominant_trait": dominant,
        "bar_chart": bar_chart,
        "scale_info": {
            "model_scale": "Likert 1–5",
            "visualization": "Bar chart",
            "percentage_formula": "(value - 1) / 4 * 100",
        },
        "conclusion": conclusion,
        "suggestion": suggestion,
        "total_text_analyzed": count,
    }


def generate_ocean_chart(ocean_scores: dict):
    traits = list(OCEAN_LABELS.values())
    values = [ocean_scores[t] for t in OCEAN_TRAITS]
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        values,
        labels=traits,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    plt.title("OCEAN Personality Composition", fontsize=14, fontweight="bold")
    plt.axis("equal")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150, transparent=True)
    plt.close()

    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
