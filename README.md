# Which LLM Wrote This?

## Data and collection

`prepare_data.py` selects shared prompt/response IDs from the public
[Ai2 Dolci DPO Model Response Pool](https://huggingface.co/datasets/allenai/Dolci-DPO-Model-Response-Pool).
It uses the `gemma3-4b`, `olmo2-7b`, and
`qwen3-no_reasoning-4b` configurations. Each selected prompt has a response
from all three configurations, so each family has the same number of examples.
The script keeps rows labeled `IF_sft_data_verified_permissive`, excludes
WildChat variants, and drops rows with simple email, phone, or redacted-name
patterns.
It writes the curated subset locally; the dataset is
not bundled with this code.

The upstream dataset has an ODC-BY license and notes that each model's outputs
may have separate terms. Review the dataset card and model terms before sharing
the collected CSV or publishing model outputs. Cite the dataset and its
technical report in your project submission.

## Setup

Use Python 3.10 or newer. Install packages from this folder:

```bash
python -m pip install -r requirements.txt
```

## Build the dataset

Run from this folder:

```bash
python prepare_data.py
```

Defaults collect 600 common prompts per model (1,800 balanced rows) after
deterministic sampling from shared prompt IDs. If too few
shared prompts are found, increase the scan limit:

```bash
python prepare_data.py --max-scan 100000 --per-model 600
```

Output: `data/llm_responses.csv` with `LLM_name`, `LLM_Input`, `LLM_output`,
`custom_id`, model configuration, and source dataset.

## Train and evaluate

```bash
python train.py
```

Each prompt ID stays entirely in train, validation, or test to prevent
responses to the same prompt leaking across splits. The test split is 20% of
prompt IDs; 20% of the remaining IDs are used for validation. Vocabulary is
learned from training text only. CNN and bidirectional LSTM models use the
same tokenization, 64-dimensional embeddings, Adam optimizer, and four epochs.

The script reports accuracy and macro-F1 for all six input/architecture
combinations. It also saves output-only confusion matrices, class-level
reports, and response style summaries. The style summaries are descriptive
and do not prove which features caused a prediction.

Results are written to `results/`:

- `classification_results.csv`
- `classification_reports.md`
- `confusion_CNN.csv` and `confusion_RNN.csv`
- `style_features.csv`

## Build the report

After data preparation and training finish, generate the PDF report:

```bash
python make_report.py --team-leader "Brayden Aubry"
```

The PDF is saved in `output/pdf/`. Edit `--team-leader` if the submission's
leader is different; add other team members to the report source if needed.

## Citation

Team Olmo et al. (2025), *Olmo 3*, technical report and
[Dolci DPO Model Response Pool](https://huggingface.co/datasets/allenai/Dolci-DPO-Model-Response-Pool).
Add the actual date collected, selected configurations, row counts, filtering,
split method, hyperparameters, and your team names to the final report.
