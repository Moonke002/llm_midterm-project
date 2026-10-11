"""Build the final PDF report from saved experiment results."""

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#168C8C")
GOLD = colors.HexColor("#E6A23C")
INK = colors.HexColor("#243447")
MUTED = colors.HexColor("#64748B")
PALE = colors.HexColor("#EDF4F7")


def percent(value):
    """Format a metric as a percentage."""
    return f"{100 * float(value):.1f}%"


class AccuracyChart(Flowable):
    """Draw grouped accuracy bars using the PDF canvas."""

    def __init__(self, results, width=470, height=190):
        super().__init__()
        self.results = results
        self.width = width
        self.height = height

    def wrap(self, available_width, available_height):
        return min(self.width, available_width), self.height

    def draw(self):
        # Compare accuracy across the three text views
        canvas = self.canv
        left, bottom, plot_width, plot_height = 48, 38, self.width - 62, 126
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(INK)
        canvas.drawString(left, self.height - 13, "Test accuracy")
        for tick in range(0, 101, 20):
            y = bottom + plot_height * tick / 100
            canvas.setStrokeColor(colors.HexColor("#DCE5EB"))
            canvas.setLineWidth(0.5)
            canvas.line(left, y, left + plot_width, y)
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(left - 6, y - 2, f"{tick}%")

        settings = ["input_only", "output_only", "input_plus_output"]
        names = ["Input only", "Output only", "Input + output"]
        model_colors = {"CNN": TEAL, "RNN": GOLD}
        group_width = plot_width / len(settings)
        for group_index, setting in enumerate(settings):
            center = left + group_width * (group_index + 0.5)
            rows = self.results[self.results["setting"] == setting]
            for offset, model_name in [(-10, "CNN"), (10, "RNN")]:
                row = rows[rows["architecture"] == model_name].iloc[0]
                value = 100 * float(row["accuracy"])
                bar_height = plot_height * value / 100
                canvas.setFillColor(model_colors[model_name])
                canvas.roundRect(
                    center + offset - 7, bottom, 14, bar_height,
                    2, stroke=0, fill=1,
                )
                canvas.setFont("Helvetica", 7)
                canvas.setFillColor(INK)
                canvas.drawCentredString(
                    center + offset, bottom + bar_height + 3,
                    f"{value:.1f}%",
                )
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(INK)
            canvas.drawCentredString(center, 19, names[group_index])

        legend_y = 1
        for x_pos, name in [(left + 15, "CNN"), (left + 75, "RNN")]:
            canvas.setFillColor(model_colors[name])
            canvas.rect(x_pos, legend_y, 8, 8, stroke=0, fill=1)
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica", 8)
            canvas.drawString(x_pos + 12, legend_y, name)


def build_styles():
    """Create a compact, consistent report style set."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=27, leading=32, textColor=NAVY, alignment=TA_LEFT,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Deck", parent=styles["Normal"], fontSize=13, leading=19,
        textColor=MUTED, spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=17, leading=21, textColor=NAVY, spaceBefore=2,
        spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="Subsection", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=11.5, leading=14,
        textColor=TEAL, spaceBefore=6, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["BodyText"], fontSize=9.4,
        leading=13.2, textColor=INK, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="SmallNote", parent=styles["BodyText"], fontSize=8,
        leading=10.5, textColor=MUTED, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Callout", parent=styles["BodyText"], fontSize=10,
        leading=14, textColor=NAVY, backColor=PALE, borderColor=TEAL,
        borderWidth=0, borderPadding=8, spaceBefore=5, spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="TableText", parent=styles["BodyText"], fontSize=8.2,
        leading=10.2, textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=8.2, leading=10.2, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="CenterSmall", parent=styles["BodyText"], fontSize=8,
        leading=10, textColor=INK, alignment=TA_CENTER,
    ))
    return styles


def make_table(rows, widths, header=True, font_size=8):
    """Style a report table with restrained alternating rows."""
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0,
                  hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DCE5EB")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    table.setStyle(TableStyle(commands))
    return table


def page_decoration(canvas, document):
    """Draw a simple header and numbered footer."""
    canvas.saveState()
    page_width, page_height = letter
    canvas.setStrokeColor(colors.HexColor("#DCE5EB"))
    canvas.setLineWidth(0.6)
    canvas.line(0.65 * inch, page_height - 0.48 * inch,
                page_width - 0.65 * inch, page_height - 0.48 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, page_height - 0.38 * inch,
                      "LLM authorship attribution | Project report")
    canvas.line(0.65 * inch, 0.46 * inch, page_width - 0.65 * inch, 0.46 * inch)
    canvas.drawString(0.65 * inch, 0.29 * inch, "Brayden Aubry")
    canvas.drawRightString(page_width - 0.65 * inch, 0.29 * inch,
                           f"Page {document.page}")
    canvas.restoreState()


def report_story(data_dir, results_dir, team_leader):
    """Assemble report content using the generated project artifacts."""
    styles = build_styles()
    results = pd.read_csv(results_dir / "classification_results.csv")
    styles_df = pd.read_csv(results_dir / "style_features.csv")
    counts = pd.read_csv(data_dir / "llm_responses.csv")["LLM_name"].value_counts()
    labels = sorted(counts.index)

    total = int(counts.sum())
    prompt_count = total // len(labels)
    test_prompts = int(round(prompt_count * 0.2))
    test_rows = test_prompts * len(labels)
    source_count = pd.read_csv(data_dir / "llm_responses.csv")[
        "source_dataset"
    ].nunique()

    def metric(setting, arch, key):
        row = results[(results["setting"] == setting)
                      & (results["architecture"] == arch)].iloc[0]
        return float(row[key])

    body = styles["BodySmall"]
    small = styles["SmallNote"]
    story = [
        Spacer(1, 0.38 * inch),
        Paragraph("Who Wrote This Response?", styles["CoverTitle"]),
        Paragraph(
            "A CNN and LSTM study of LLM response attribution",
            styles["Deck"],
        ),
        Paragraph(
            f"<b>Team leader:</b> {team_leader}<br/>"
            f"<b>Team member:</b> {team_leader} (individual project)<br/>"
            "<b>Report date:</b> October 11, 2026",
            body,
        ),
        Spacer(1, 0.12 * inch),
        Paragraph("Research questions", styles["Section"]),
        Paragraph(
            "<b>RQ1:</b> Can response text identify the model family? "
            "<b>RQ2:</b> Does the prompt help? "
            "<b>RQ4:</b> What response style differences can we measure? "
            "We did not run RQ3 because filtering left only one source category.",
            body,
        ),
        Paragraph("1. Data and collection", styles["Section"]),
        Paragraph(
            "We used the public Ai2 Dolci DPO Model Response Pool. It contains "
            "responses from several model configurations to shared prompts. "
            "We selected one configuration from each of three "
            "families: Gemma 3 4B, OLMo 2 7B, and Qwen 3 no-reasoning 4B. The "
            "data builder matched records by <font name='Courier'>custom_id</font> "
            "and kept only prompts and source labels identical across all three "
            "configurations.", body,
        ),
        Paragraph(
            "To reduce use of user-originated or potentially sensitive content, "
            "the collected subset keeps only rows labeled "
            "<font name='Courier'>IF_sft_data_verified_permissive</font>. It "
            "excludes WildChat and DaringAnteater source rows and drops rows with "
            "simple email, phone-number, and redacted-name patterns. This is a "
            "basic screening step, not a complete privacy audit. The source "
            "dataset is ODC-BY; the dataset card also advises checking each "
            "model's output terms before redistribution.", body,
        ),
        Paragraph("Dataset schema", styles["Subsection"]),
        make_table([
            ["Required field", "Meaning", "Example"],
            ["LLM_name", "Model-family label", "Gemma / OLMo / Qwen"],
            ["LLM_Input", "Prompt given to the model", "User instruction"],
            ["LLM_output", "Model-generated response", "Assistant answer"],
        ], [1.1 * inch, 2.3 * inch, 2.8 * inch]),
        Spacer(1, 8),
        Paragraph("Sample balance", styles["Subsection"]),
        make_table(
            [["Family", "Model configuration", "Examples"]]
            + [[name, config, str(int(counts[name]))]
               for name, config in [
                   ("Gemma", "gemma3-4b"),
                   ("OLMo", "olmo2-7b"),
                   ("Qwen", "qwen3-no_reasoning-4b"),
               ]],
            [1.2 * inch, 3.7 * inch, 1.3 * inch],
        ),
        Spacer(1, 7),
        Paragraph(
            f"The final CSV contains {total:,} rows from {prompt_count:,} "
            f"matched prompt IDs and {source_count} source category. The data is "
            "saved locally rather than included in the code repository.", small,
        ),
        PageBreak(),
        Paragraph("2. Models and training", styles["Section"]),
        Paragraph(
            "Text was lowercased and split into word and punctuation tokens. "
            "The vocabulary was learned from training data only, with a 12,000 "
            "token cap. Sequences were limited to 128 tokens. For combined text, "
            "the encoder reserved 64 tokens for the prompt and 64 for the "
            "response.", body,
        ),
        make_table([
            ["Component", "CNN", "RNN (LSTM)"],
            ["Embedding", "64 dimensions", "64 dimensions"],
            ["Main layer", "96 filters, kernel size 3", "Bidirectional LSTM, 64 hidden units"],
            ["Pooling / output", "Global max pooling", "Concatenate final directions"],
            ["Regularization", "Dropout 0.3", "Dropout 0.3"],
            ["Optimization", "Adam, learning rate 0.001", "Adam, learning rate 0.001"],
        ], [1.25 * inch, 2.5 * inch, 2.45 * inch]),
        Paragraph(
            f"Prompt IDs were split as 64% train, 16% validation, and 20% test. "
            f"All three model responses to one prompt stayed in the same split. "
            f"The test set has {test_prompts} prompt IDs and {test_rows} rows, "
            f"with {test_prompts} examples per class. Training used four epochs, "
            "batch size 64, seed 42, and the best validation-loss checkpoint. "
            "We report accuracy and macro-F1. The 3-class chance reference is "
            f"{percent(1/len(labels))} accuracy.", body,
        ),
        PageBreak(),
        Paragraph("3. Results", styles["Section"]),
        Paragraph(
            "Table 2 compares the two required architectures across all three "
            "input settings. The output-only CNN is the strongest required "
            "model. Both prompt-only models are at chance, as expected because "
            "each test prompt appears once for every family.", body,
        ),
        AccuracyChart(results),
        make_table([
            ["Input view", "CNN accuracy", "CNN macro-F1", "LSTM accuracy", "LSTM macro-F1"],
        ] + [
            [
                {"input_only": "Input only", "output_only": "Output only",
                 "input_plus_output": "Input + output"}[setting],
                percent(metric(setting, "CNN", "accuracy")),
                percent(metric(setting, "CNN", "macro_f1")),
                percent(metric(setting, "RNN", "accuracy")),
                percent(metric(setting, "RNN", "macro_f1")),
            ]
            for setting in ["input_only", "output_only", "input_plus_output"]
        ], [1.35 * inch, 1.15 * inch, 1.2 * inch, 1.15 * inch, 1.35 * inch]),
        Spacer(1, 9),
        Paragraph("Output-only confusion matrices", styles["Subsection"]),
        Paragraph(
            "Rows are true families; columns are predicted families. "
            "The CNN correctly classified 77 Gemma, 71 OLMo, and 63 Qwen test "
            "responses. The LSTM was more likely to confuse OLMo and Qwen.", body,
        ),
    ]

    matrix_tables = []
    for architecture in ["CNN", "RNN"]:
        matrix = pd.read_csv(
            results_dir / f"confusion_{architecture}.csv", index_col=0
        )
        table_rows = [[f"{architecture}"] + list(matrix.columns)]
        table_rows += [[str(index)] + [str(int(value)) for value in row]
                       for index, row in matrix.iterrows()]
        matrix_tables.append(make_table(
            table_rows, [0.7 * inch] * 4
        ))
    side_by_side = Table([[matrix_tables[0], matrix_tables[1]]],
                         colWidths=[3.0 * inch, 3.0 * inch])
    side_by_side.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([side_by_side, Spacer(1, 5), Paragraph(
        "The combined view improved over prompt-only CNN and LSTM, but "
        "remained below output-only performance. The fixed 128-token budget "
        "split between prompt and response reduces the response context in "
        "that comparison.", small
    ), PageBreak()])

    story.extend([
        Paragraph("4. Style analysis and discussion", styles["Section"]),
        Paragraph(
            "RQ4 was explored with four simple response measurements. Average "
            "length and formatting differed across the families. These summaries "
            "are descriptive; they do not establish that a particular feature "
            "caused the CNN or LSTM predictions.", body,
        ),
        make_table([
            ["Family", "Words", "Lexical diversity", "Punctuation rate", "Lines"],
        ] + [[
            str(row["LLM_name"]),
            f"{row['word_count']:.1f}",
            f"{row['lexical_diversity']:.3f}",
            f"{row['punctuation_rate']:.3f}",
            f"{row['line_count']:.1f}",
        ] for _, row in styles_df.iterrows()],
            [1.0 * inch, 1.05 * inch, 1.45 * inch, 1.25 * inch, 1.0 * inch],
        ),
        Paragraph(
            "Qwen responses were longest on average (381 words) and had the "
            "highest mean line count (35.1). OLMo responses were shortest "
            "(224 words) and had the highest lexical-diversity ratio (0.627). "
            "Gemma responses had the highest punctuation rate (0.090). The "
            "output-only CNN was the best classifier, which is consistent with "
            "response length, vocabulary, and formatting carrying some signal. "
            "A feature-removal experiment would be needed to attribute the "
            "performance to any one signal.", body,
        ),
        Paragraph("5. Scope, limitations, and lessons", styles["Section"]),
        Paragraph(
            "<b>RQ1 and RQ2:</b> Completed with both required neural models and "
            "all three input views. The prompt-only result is a useful control: "
            "with identical prompts grouped across splits, the prompt does not "
            "explain the model label. The response itself carries the measured "
            "signal, though best accuracy is moderate at 58.6%.", body,
        ),
        Paragraph(
            "<b>RQ3:</b> Not run. The privacy and source-license filter left one "
            "safe source category, so this dataset cannot support a valid "
            "cross-source holdout. RQ3 is extra credit in the assignment, not a "
            "required research question. The source category is not itself a "
            "verified semantic task label.", body,
        ),
        Paragraph(
            "<b>Limitations:</b> Results cover one configuration from each family, "
            "600 prompts per family, one random seed, and one train/validation/test "
            "split. The personal-information checks are heuristic, not a complete "
            "manual review. The 128-token cap truncates long texts. Model-family "
            "attribution here should not be generalized to every prompt, version, "
            "or deployment of these model families.", body,
        ),
        Paragraph(
            "<b>Lessons:</b> Balanced examples and prompt-grouped splitting matter "
            "for fair comparison. Input-only performance at chance helps rule out "
            "prompt leakage. The CNN outperformed the LSTM on response-only text "
            "in this run, but repeated seeds and feature ablations are needed for "
            "stronger conclusions.", body,
        ),
        Paragraph("References and data use", styles["Subsection"]),
        Paragraph(
            "[1] Ai2, <i>Dolci DPO Model Response Pool</i>, public dataset card. "
            "<link href='https://huggingface.co/datasets/allenai/Dolci-DPO-Model-Response-Pool' color='#168C8C'>"
            "Open dataset card</link>. Accessed October 11, 2026. Dataset license: "
            "ODC-BY; check the model-specific output terms before redistribution.",
            small,
        ),
        Paragraph(
            "[2] Team Olmo et al. (2025), <i>Olmo 3</i>, technical report, "
            "arXiv:2512.13961.", small,
        ),
    ])
    return story


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--team-leader", default="Brayden Aubry")
    args = parser.parse_args()
    project = Path(args.project_dir)
    output = project / "output" / "pdf" / "llm_authorship_project_report.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=letter,
        rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.66 * inch, bottomMargin=0.62 * inch,
        title="Who Wrote This Response?",
        author=args.team_leader,
        subject="LLM response attribution using CNN and LSTM",
    )
    document.build(
        report_story(project / "data", project / "results", args.team_leader),
        onFirstPage=page_decoration, onLaterPages=page_decoration,
    )
    print(f"Wrote report: {output.resolve()}")


if __name__ == "__main__":
    main()
