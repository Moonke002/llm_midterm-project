"""Build the final PDF report from saved experiment results."""

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BLUE = colors.HexColor("#2F5496")
INK = colors.HexColor("#202020")
MUTED = colors.HexColor("#666666")
PALE = colors.HexColor("#EAF0F8")


def percent(value):
    """Format a metric as a percentage."""
    return f"{100 * float(value):.1f}%"


class AccuracyChart(Flowable):
    """Draw grouped accuracy bars using the PDF canvas."""

    def __init__(self, results, width=470, height=160):
        super().__init__()
        self.results = results
        self.width = width
        self.height = height

    def wrap(self, available_width, available_height):
        return min(self.width, available_width), self.height

    def draw(self):
        # Compare accuracy across the three text views
        canvas = self.canv
        left, bottom, plot_width, plot_height = 48, 32, self.width - 62, 102
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
        model_colors = {"CNN": BLUE, "RNN": colors.HexColor("#8FAADC")}
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


class WordCountChart(Flowable):
    """Draw average response length by model family."""

    def __init__(self, styles, width=470, height=110):
        super().__init__()
        self.styles = styles
        self.width = width
        self.height = height

    def wrap(self, available_width, available_height):
        return min(self.width, available_width), self.height

    def draw(self):
        canvas = self.canv
        left, bottom, plot_width = 78, 28, self.width - 160
        max_value = max(float(row["word_count"]) for row in self.styles)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(INK)
        canvas.drawString(left, self.height - 12, "Average response length")
        for row_index, row in enumerate(self.styles):
            y = bottom + (2 - row_index) * 22
            name = str(row["LLM_name"])
            value = float(row["word_count"])
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(INK)
            canvas.drawRightString(left - 8, y + 4, name)
            canvas.setFillColor(PALE)
            canvas.rect(left, y, plot_width, 15, stroke=0, fill=1)
            canvas.setFillColor(BLUE)
            canvas.rect(left, y, plot_width * value / max_value, 15,
                        stroke=0, fill=1)
            canvas.setFillColor(INK)
            canvas.drawString(left + plot_width * value / max_value + 6,
                              y + 4, f"{value:.0f} words")


def build_styles():
    """Set the report fonts and paragraph spacing."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=15, leading=18, textColor=INK, alignment=TA_CENTER,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="Deck", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=12.5, leading=15, textColor=INK, alignment=TA_CENTER,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=12.5, leading=15, textColor=BLUE, spaceBefore=7,
        spaceAfter=2, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="Subsection", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=10.5, leading=13,
        textColor=BLUE, spaceBefore=5, spaceAfter=3, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=10, leading=12.5, textColor=INK, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="SmallNote", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.5, leading=10.5, textColor=MUTED, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="TableText", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.4, leading=10.4, textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=8.4, leading=10.4, textColor=BLUE,
    ))
    styles.add(ParagraphStyle(
        name="CenterSmall", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.5, leading=10.5, textColor=INK, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.5, leading=10.5, textColor=MUTED, alignment=TA_CENTER,
        spaceAfter=5,
    ))
    return styles


def make_table(rows, widths, header=True, font_size=8):
    """Format a small table with readable spacing."""
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0,
                  hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D2E2")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), PALE),
            ("TEXTCOLOR", (0, 0), (-1, 0), BLUE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    table.setStyle(TableStyle(commands))
    return table


def page_decoration(canvas, document):
    """Add the course label used in the reference footer."""
    canvas.saveState()
    page_width, _ = letter
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(page_width / 2, 0.31 * inch,
                             "CMPSC 448 - Project Report")
    canvas.restoreState()


def report_story(data_dir, results_dir, team_leader):
    """Write the project findings in a short report format."""
    styles = build_styles()
    results = pd.read_csv(results_dir / "classification_results.csv")
    features = pd.read_csv(results_dir / "style_features.csv")
    data = pd.read_csv(data_dir / "llm_responses.csv")
    counts = data["LLM_name"].value_counts()
    labels = sorted(counts.index)

    total = len(data)
    prompt_count = data["custom_id"].nunique()
    source_count = data["source_dataset"].nunique()
    test_prompts = int(round(prompt_count * 0.2))
    test_rows = test_prompts * len(labels)
    body = styles["BodySmall"]

    def metric(setting, architecture, column):
        row = results[
            (results["setting"] == setting)
            & (results["architecture"] == architecture)
        ].iloc[0]
        return float(row[column])

    def paragraph(text, style=body):
        return Paragraph(text, style)

    def lead(label, text):
        return Paragraph(f"<b>{label}:</b> {text}", body)

    data_steps = ListFlowable(
        [
            ListItem(paragraph("Keep prompts that appear for all three model configurations.")),
            ListItem(paragraph("Apply the source and text checks, then select 600 shared prompts.")),
            ListItem(paragraph("Keep all responses to one prompt in the same data split.")),
        ],
        bulletType="1",
        bulletFormat="%s.",
        leftIndent=18,
        bulletDedent=14,
        bulletFontName="Helvetica",
        bulletFontSize=10,
        bulletColor=INK,
        spaceAfter=3,
    )

    story = [
        Spacer(1, 0.12 * inch),
        Paragraph("CMPSC 448 - Project Report", styles["CoverTitle"]),
        Paragraph("LLM Authorship Classification", styles["Deck"]),
        paragraph(
            f"October 11, 2026<br/>Team leader: {team_leader}<br/>"
            f"Team member: {team_leader} (individual project)",
            styles["CenterSmall"],
        ),
        Spacer(1, 5),
        Paragraph("1. Problem", styles["Section"]),
        paragraph(
            "For this project, I tested whether a classifier could tell which "
            "model family wrote a response. I used Gemma, OLMo, and Qwen outputs. "
            "I compared the prompt, the response, and both together."
        ),
        Paragraph("2. Method", styles["Section"]),
        paragraph(
            "I used the public Ai2 Dolci DPO Model Response Pool. It has model "
            "responses to shared prompts. I selected Gemma 3 4B, OLMo 2 7B, and "
            "Qwen 3 no-reasoning 4B so each prompt had an answer from all three."
        ),
        paragraph(
            "<b>Data fields:</b> Each row has "
            "<font name='Courier'>LLM_name</font> for the model family, "
            "<font name='Courier'>LLM_Input</font> for the prompt, and "
            "<font name='Courier'>LLM_output</font> for the response. The CSV "
            "also records the prompt ID, model configuration, and source."
        ),
        paragraph(
            f"The final file has {total:,} rows: 600 responses from each family "
            f"for {prompt_count:,} matched prompts. I kept rows labeled "
            "<font name='Courier'>IF_sft_data_verified_permissive</font>, left "
            "out WildChat and DaringAnteater rows, and removed simple email, "
            "phone-number, and redacted-name matches. These checks are basic "
            "filters, not a manual privacy review."
        ),
        paragraph(
            "The dataset card lists ODC-BY and says model-specific output terms "
            "may apply. Check those terms before sharing the collected responses. "
            "<link href='https://huggingface.co/datasets/allenai/Dolci-DPO-Model-Response-Pool' color='#2F5496'>"
            "Dataset card</link>.",
            styles["SmallNote"],
        ),
        make_table(
            [
                ["Field", "What it contains"],
                ["LLM_name", "Gemma, OLMo, or Qwen"],
                ["LLM_Input", "The prompt sent to the model"],
                ["LLM_output", "The model's response"],
            ],
            [1.45 * inch, 4.75 * inch],
        ),
        Spacer(1, 4),
        paragraph(
            f"I used {data['source_dataset'].nunique()} source category after "
            "filtering. The data CSV is not included in the repository; the "
            "preparation script rebuilds it."
        ),
        data_steps,
        Paragraph("3. Models and training", styles["Section"]),
        paragraph(
            f"I split prompt IDs into 64% training, 16% validation, and 20% "
            f"testing. The test set has {test_prompts} prompts and {test_rows} "
            "responses. This kept answers to the same prompt out of different "
            "splits."
        ),
        paragraph(
            "I lowercased the text and split it into words and punctuation. "
            "The vocabulary came from training text only and held up to 12,000 "
            "tokens. Both models used 64-dimension embeddings, dropout of 0.3, "
            "Adam with a 0.001 learning rate, and four training epochs. Inputs "
            "were limited to 128 tokens. For prompt plus response, I reserved "
            "64 tokens for each. I used seed 42."
        ),
        make_table(
            [
                ["Model", "Main layer", "Output"],
                ["CNN", "96 filters, kernel size 3", "Max pooling"],
                ["RNN", "Bidirectional LSTM, 64 hidden units", "Join both directions"],
            ],
            [0.9 * inch, 3.35 * inch, 1.95 * inch],
        ),
        paragraph(
            "The best checkpoint was the one with the lowest validation loss. "
            "I report accuracy and macro-F1. With three balanced families, "
            f"chance accuracy is {percent(1 / len(labels))}."
        ),
        Paragraph("4. Question 1: Can a response identify its model family?", styles["Section"]),
        lead(
            "Question",
            "Can a CNN or LSTM identify the model from the response alone?",
        ),
        lead(
            "Final understanding",
            f"Yes, in this test. The response-only CNN reached "
            f"{percent(metric('output_only', 'CNN', 'accuracy'))} accuracy and "
            f"{percent(metric('output_only', 'CNN', 'macro_f1'))} macro-F1. "
            f"The LSTM reached {percent(metric('output_only', 'RNN', 'accuracy'))} "
            f"accuracy and {percent(metric('output_only', 'RNN', 'macro_f1'))} "
            "macro-F1.",
        ),
        AccuracyChart(results),
        paragraph(
            "Figure 1. Test accuracy for both models and all three input settings.",
            styles["Caption"],
        ),
        make_table(
            [
                ["Input", "CNN accuracy", "CNN macro-F1", "LSTM accuracy", "LSTM macro-F1"],
            ]
            + [
                [
                    {
                        "input_only": "Prompt",
                        "output_only": "Response",
                        "input_plus_output": "Prompt + response",
                    }[setting],
                    percent(metric(setting, "CNN", "accuracy")),
                    percent(metric(setting, "CNN", "macro_f1")),
                    percent(metric(setting, "RNN", "accuracy")),
                    percent(metric(setting, "RNN", "macro_f1")),
                ]
                for setting in ["input_only", "output_only", "input_plus_output"]
            ],
            [1.2 * inch, 1.2 * inch, 1.25 * inch, 1.2 * inch, 1.35 * inch],
        ),
        PageBreak(),
        Paragraph("5. Question 2: Does the prompt help?", styles["Section"]),
        lead(
            "Question",
            "Can the prompt identify the model before a response is seen? "
            "Does it improve the result when added to the response?",
        ),
        lead(
            "Final understanding",
            "The prompt alone scored 33.3% for both models, the three-class "
            "chance level. Adding the prompt did not improve the response-only "
            "scores. The combined CNN reached 46.9% accuracy; the combined "
            "LSTM reached 35.3%. Since all three models answered each selected "
            "prompt, the prompt by itself did not reveal the label.",
        ),
        paragraph(
            "The combined model saw only 64 response tokens because its "
            "128-token limit was split evenly with the prompt. That may help "
            "explain why the combined setting did worse than response-only."
        ),
        Paragraph("Output-only confusion matrices", styles["Subsection"]),
    ]

    matrix_tables = []
    for architecture in ["CNN", "RNN"]:
        matrix = pd.read_csv(
            results_dir / f"confusion_{architecture}.csv", index_col=0
        )
        rows = [[architecture] + list(matrix.columns)]
        rows.extend(
            [[str(index)] + [str(int(value)) for value in values]
             for index, values in matrix.iterrows()]
        )
        matrix_tables.append(make_table(rows, [0.72 * inch] * 4))
    matrices = Table(
        [[matrix_tables[0], matrix_tables[1]]],
        colWidths=[3.05 * inch, 3.05 * inch],
    )
    matrices.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        matrices,
        paragraph(
            "Figure 2. Rows show the actual family; columns show the prediction.",
            styles["Caption"],
        ),
        Paragraph("6. Question 3: Does it work on a new task?", styles["Section"]),
        lead(
            "Question",
            "Would a model trained on one task type identify a model on another?",
        ),
        lead(
            "Final understanding",
            "I did not run this test. After filtering, the dataset had one shared "
            "source category, so there was no separate category to hold out. "
            "This leaves cross-task generalization unanswered.",
        ),
        Paragraph("7. Question 4: What response features differ?", styles["Section"]),
        lead(
            "Question",
            "Do response length, vocabulary, punctuation, or formatting differ?",
        ),
        lead(
            "Final understanding",
            "The measurements show some differences. They do not prove that any "
            "one feature caused the classifiers' predictions.",
        ),
        make_table(
            [
                ["Family", "Words", "Word variety", "Punctuation", "Lines"],
            ]
            + [
                [
                    str(row["LLM_name"]),
                    f"{row['word_count']:.1f}",
                    f"{row['lexical_diversity']:.3f}",
                    f"{row['punctuation_rate']:.3f}",
                    f"{row['line_count']:.1f}",
                ]
                for _, row in features.iterrows()
            ],
            [0.95 * inch, 1.0 * inch, 1.2 * inch, 1.45 * inch, 1.0 * inch],
        ),
        WordCountChart(features.to_dict("records")),
        paragraph(
            "Figure 3. Mean response length in words.",
            styles["Caption"],
        ),
        paragraph(
            "Word variety is the number of unique lowercase words divided by "
            "the total words. Punctuation is punctuation characters divided "
            "by all response characters. Lines are counted from line breaks."
        ),
        paragraph(
            "Qwen averaged 381 words; OLMo averaged 224 and had the highest "
            "word-variety ratio. Gemma had the highest punctuation rate. These "
            "averages do not describe every response."
        ),
        Paragraph("8. Challenges and how I addressed them", styles["Section"]),
        paragraph(
            "The source included rows from several collections, so I used the "
            "verified permissive category and simple text checks for the dataset. "
            "Those checks cannot catch every sensitive detail. I also matched "
            "prompts across model configurations and kept each prompt group in "
            "one split to reduce leakage."
        ),
        paragraph(
            "Long responses had to be cut to fit the 128-token input. In the "
            "combined setting, half the space went to the prompt. A larger input "
            "limit or a separate experiment with more response tokens could "
            "change those results."
        ),
        Paragraph("9. Key lessons learned", styles["Section"]),
        ListFlowable(
            [
                ListItem(paragraph("The response text was more useful than the prompt for identifying the model.")),
                ListItem(paragraph("Grouping by prompt gave a fairer test because the same question did not appear in training and testing.")),
                ListItem(paragraph("The CNN did better than the LSTM on response-only text in this run.")),
                ListItem(paragraph("One split and one configuration per family are not enough to generalize to every model version.")),
            ],
            bulletType="bullet",
            leftIndent=18,
            bulletDedent=9,
            bulletFontName="Helvetica",
            bulletFontSize=8,
            bulletColor=BLUE,
            spaceAfter=3,
        ),
        Paragraph("10. Conclusion", styles["Section"]),
        paragraph(
            "The response-only CNN identified the model family better than "
            "chance, but its accuracy was 58.6%. The prompt alone did not help, "
            "and adding it lowered performance in this setup. The results apply "
            "to these three configurations and this test split."
        ),
        Paragraph("References", styles["Section"]),
        paragraph(
            "Ai2, <i>Dolci DPO Model Response Pool</i>, dataset card. "
            "<link href='https://huggingface.co/datasets/allenai/Dolci-DPO-Model-Response-Pool' color='#2F5496'>"
            "Open the dataset card</link>. Accessed October 11, 2026. "
            "The card lists ODC-BY and notes that model-specific output terms "
            "may apply.",
            styles["SmallNote"],
        ),
        paragraph(
            "Team Olmo et al. (2025), <i>Olmo 3</i>, technical report, "
            "arXiv:2512.13961.",
            styles["SmallNote"],
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
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        title="LLM Authorship Classification",
        author=args.team_leader,
        subject="CNN and LSTM report on model-family classification",
    )
    document.build(
        report_story(project / "data", project / "results", args.team_leader),
        onFirstPage=page_decoration, onLaterPages=page_decoration,
    )
    print(f"Wrote report: {output.resolve()}")


if __name__ == "__main__":
    main()
