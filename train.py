"""Train simple CNN and LSTM models for LLM attribution."""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
PAD = 0
UNK = 1
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def set_seed(seed):
    """Set random seeds for repeatable experiments."""
    # Set seeds before splitting and training
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text):
    """Split text into lowercase words and punctuation."""
    return TOKEN_RE.findall(str(text).lower())


def make_vocabulary(texts, max_words):
    """Build a compact vocabulary from training text only."""
    # Build the vocabulary from training text
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    words = [word for word, _ in counts.most_common(max_words - 2)]
    return {word: index + 2 for index, word in enumerate(words)}


def encode(text, vocabulary, max_len):
    """Convert one text into a fixed-length token sequence."""
    ids = [vocabulary.get(word, UNK) for word in tokenize(text)[:max_len]]
    return ids + [PAD] * (max_len - len(ids))


def encode_pair(prompt, response, vocabulary, max_len):
    """Reserve equal sequence space for prompt and response."""
    half = max_len // 2
    tokens = tokenize(prompt)[:half] + tokenize(response)[:max_len - half]
    ids = [vocabulary.get(word, UNK) for word in tokens]
    return ids + [PAD] * (max_len - len(ids))


class TextRows(Dataset):
    """Store encoded examples and integer class labels."""

    # Convert rows into tensors for training

    def __init__(self, frame, text_column, vocabulary, label_map, max_len):
        if text_column == "combined_text":
            self.texts = [
                encode_pair(prompt, response, vocabulary, max_len)
                for prompt, response in zip(
                    frame["LLM_Input"], frame["LLM_output"]
                )
            ]
        else:
            self.texts = [
                encode(value, vocabulary, max_len)
                for value in frame[text_column]
            ]
        self.labels = [label_map[value] for value in frame["LLM_name"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(self.texts[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class CNN(nn.Module):
    """Use token embeddings and convolution for text classification."""

    # Learn nearby word patterns with filters

    def __init__(self, vocab_size, class_count, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 64, padding_idx=padding_idx
        )
        self.convolution = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(96, class_count)

    def forward(self, tokens):
        values = self.embedding(tokens).transpose(1, 2)
        values = torch.relu(self.convolution(values))
        values = self.pool(values).squeeze(-1)
        return self.output(self.dropout(values))


class RNN(nn.Module):
    """Use a bidirectional LSTM for text classification."""

    # Read each sequence in both directions

    def __init__(self, vocab_size, class_count, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 64, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            64, 64, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(128, class_count)

    def forward(self, tokens):
        values = self.embedding(tokens)
        _, (hidden, _) = self.lstm(values)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(self.dropout(pooled))


def run_epoch(model, loader, loss_fn, optimizer, device):
    """Run one training or evaluation pass over batches."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions = []
    actual = []

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(tokens)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        actual.extend(labels.cpu().tolist())

    return total_loss / len(actual), predictions, actual


def split_by_prompt(frame):
    """Keep responses to each prompt inside one data split."""
    # Keep each prompt inside one data split
    prompt_ids = frame["custom_id"].drop_duplicates().to_numpy()
    train_ids, test_ids = train_test_split(
        prompt_ids, test_size=0.2, random_state=SEED
    )
    train_ids, val_ids = train_test_split(
        train_ids, test_size=0.2, random_state=SEED
    )
    train = frame[frame["custom_id"].isin(train_ids)].copy()
    valid = frame[frame["custom_id"].isin(val_ids)].copy()
    test = frame[frame["custom_id"].isin(test_ids)].copy()
    return train, valid, test


def train_neural(train, valid, test, text_column, architecture, labels,
                 epochs, batch_size, max_len, max_words, device):
    """Train one neural architecture and score its held-out split."""
    # Keep the epoch with lowest validation loss
    vocabulary = make_vocabulary(train[text_column], max_words)
    label_map = {name: index for index, name in enumerate(labels)}
    train_data = TextRows(
        train, text_column, vocabulary, label_map, max_len
    )
    valid_data = TextRows(
        valid, text_column, vocabulary, label_map, max_len
    )
    test_data = TextRows(test, text_column, vocabulary, label_map, max_len)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model_class = CNN if architecture == "CNN" else RNN
    model = model_class(len(vocabulary) + 2, len(labels), PAD).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, _, _ = run_epoch(
            model, valid_loader, loss_fn, None, device
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    _, predictions, actual = run_epoch(
        model, test_loader, loss_fn, None, device
    )
    return predictions, actual


def save_confusion_matrix(actual, predicted, labels, path, title):
    """Save a labeled confusion matrix as a CSV file."""
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(actual, predicted):
        matrix[true_label, pred_label] += 1
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis(
        "true_model"
    ).to_csv(path)


def describe_style(frame, output_dir):
    """Summarize simple response style features by model."""
    # Measure length, word variety, punctuation, and lines
    result = frame.copy()
    result["word_count"] = result["LLM_output"].map(
        lambda text: len(str(text).split())
    )
    result["lexical_diversity"] = result["LLM_output"].map(
        lambda text: len(set(str(text).lower().split()))
        / max(1, len(str(text).split()))
    )
    result["punctuation_rate"] = result["LLM_output"].map(
        lambda text: sum(not char.isalnum() and not char.isspace()
                         for char in str(text))
        / max(1, len(str(text)))
    )
    result["line_count"] = result["LLM_output"].map(
        lambda text: str(text).count("\n") + 1
    )
    features = [
        "word_count", "lexical_diversity", "punctuation_rate", "line_count"
    ]
    summary = result.groupby("LLM_name")[features].mean().round(3)
    summary.to_csv(output_dir / "style_features.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Compare both models across three text views
    parser.add_argument("--data", default="data/llm_responses.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-words", type=int, default=12000)
    args = parser.parse_args()

    set_seed(SEED)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data).dropna(
        subset=["custom_id", "LLM_name", "LLM_Input", "LLM_output"]
    )
    labels = sorted(frame["LLM_name"].unique())
    if len(labels) < 3:
        raise ValueError("The dataset must contain three model families.")
    train, valid, test = split_by_prompt(frame)
    settings = {
        "input_only": "LLM_Input",
        "output_only": "LLM_output",
    }
    frame["combined_text"] = (
        "Prompt: " + frame["LLM_Input"].astype(str)
        + " Response: " + frame["LLM_output"].astype(str)
    )
    train["combined_text"] = (
        "Prompt: " + train["LLM_Input"].astype(str)
        + " Response: " + train["LLM_output"].astype(str)
    )
    valid["combined_text"] = (
        "Prompt: " + valid["LLM_Input"].astype(str)
        + " Response: " + valid["LLM_output"].astype(str)
    )
    test["combined_text"] = (
        "Prompt: " + test["LLM_Input"].astype(str)
        + " Response: " + test["LLM_output"].astype(str)
    )
    settings["input_plus_output"] = "combined_text"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_rows = []
    report_rows = []

    for setting, text_column in settings.items():
        for architecture in ("CNN", "RNN"):
            predictions, actual = train_neural(
                train, valid, test, text_column, architecture, labels,
                args.epochs, args.batch_size, args.max_len,
                args.max_words, device,
            )
            accuracy = accuracy_score(actual, predictions)
            macro_f1 = f1_score(actual, predictions, average="macro")
            result_rows.append(
                {
                    "setting": setting,
                    "architecture": architecture,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                }
            )
            report_rows.append(
                f"## {setting} / {architecture}\n\n"
                + classification_report(
                    actual, predictions, labels=range(len(labels)),
                    target_names=labels, zero_division=0,
                )
            )
            if setting == "output_only":
                save_confusion_matrix(
                    actual, predictions, labels,
                    output_dir / f"confusion_{architecture}.csv",
                    f"Output-only {architecture}",
                )
            print(
                f"{setting} {architecture}: "
                f"accuracy={accuracy:.3f}, macro-F1={macro_f1:.3f}"
            )

    pd.DataFrame(result_rows).to_csv(
        output_dir / "classification_results.csv", index=False
    )
    (output_dir / "classification_reports.md").write_text(
        "\n\n".join(report_rows), encoding="utf-8"
    )
    describe_style(frame, output_dir)
    print(f"Results saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
