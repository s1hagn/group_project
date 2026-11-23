from base64 import b64encode
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "cleaned_viral_videos.csv")

charts = [
    ("Viral Rate by Category", "viral_rate_by_category.png"),
    ("Viral Rate by Duration Group", "viral_rate_by_duration.png"),
    ("Category and Duration", "viral_rate_heatmap.png"),
    ("Median Subscribers by Category", "median_subscribers_by_category.png"),
    ("Model Confusion Matrices", "ml_confusion_matrices.png"),
    ("ROC Curve Comparison", "ml_roc_comparison.png"),
]

sections = []
for title, filename in charts:
    image = b64encode((root / filename).read_bytes()).decode("ascii")
    sections.append(
        f'<section><h2>{title}</h2>'
        f'<img alt="{title}" src="data:image/png;base64,{image}"></section>'
    )

report = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>YouTube Video Virality Report</title>
<style>
    body {{ max-width: 1000px; margin: 40px auto; padding: 0 20px;
            font-family: Arial, sans-serif; color: #172033; }}
    section {{ margin: 32px 0; }}
    img {{ max-width: 100%; height: auto; }}
</style>
<h1>YouTube Video Virality Report</h1>
<p>Videos processed: {len(df):,} | Viral rate: {df["viral"].mean():.1%}</p>
<p>Source: Synthetic YouTube Video Dataset</p>
{"".join(sections)}
</html>
"""

(root / "virality_report.html").write_text(report, encoding="utf-8")