"""Agent: Report Generator
Assembles a professional downloadable PDF: executive summary, EDA, charts,
insights, recommendations, and ML results.
"""
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak


def generate_report(output_path: str, understanding: str, profile: dict, eda_results: dict,
                     insights: list, recommendations: list, ml_results: dict,
                     chart_images: list) -> str:
    """chart_images: list of (image_path, title) tuples, pre-rendered to PNG."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Data Analysis Report", styles["Title"]))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(Paragraph(understanding or "N/A", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Dataset Overview", styles["Heading1"]))
    story.append(Paragraph(
        f"Rows: {profile.get('n_rows')} | Columns: {profile.get('n_cols')} | "
        f"Duplicate rows: {profile.get('n_duplicates')}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    if insights:
        story.append(Paragraph("Key Business Insights", styles["Heading1"]))
        for insight in insights:
            story.append(Paragraph(f"\u2022 {insight}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    if chart_images:
        story.append(PageBreak())
        story.append(Paragraph("Visualizations", styles["Heading1"]))
        for img_path, title in chart_images:
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Image(img_path, width=5.5 * inch, height=3.3 * inch))
            story.append(Spacer(1, 0.2 * inch))

    if ml_results:
        story.append(PageBreak())
        story.append(Paragraph("Machine Learning Results", styles["Heading1"]))
        for k, v in ml_results.items():
            if k in ("feature_importance", "confusion_matrix", "shap_summary", "cluster_sizes"):
                continue
            story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        fi = ml_results.get("feature_importance")
        if fi:
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph("Top Features:", styles["Heading2"]))
            for item in fi:
                story.append(Paragraph(f"- {item['feature']}: {item['importance']}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    if recommendations:
        story.append(Paragraph("Recommendations", styles["Heading1"]))
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles["Normal"]))

    doc.build(story)
    return output_path
