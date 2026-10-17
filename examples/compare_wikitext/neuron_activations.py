"""Correlate each neuron's MAGIC scores with its activation on the training
chunks under the final model, and with the change in that activation from the
pretrained model.

python examples/compare_wikitext/neuron_activations.py \
    --config examples/compare_wikitext/neurons.yaml \
    --model runs/compare_wikitext/random/retrained/base
"""

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.config.config_io import read_config
from bergson.data import load_scores_loss_signed
from bergson.utils.neuron import neuron_activations


def chunk_activations(model, tokenizer, texts, neurons, device):
    """Mean, max and fraction-positive activation of each neuron per chunk."""
    stats = np.zeros((3, len(neurons), len(texts)))
    with ExitStack() as stack, torch.no_grad():
        acts = [stack.enter_context(neuron_activations(model, n)) for n in neurons]
        for i, text in enumerate(texts):
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            model(input_ids=ids)
            for j, a in enumerate(acts):
                x = a.pop()[0].float().cpu()
                stats[:, j, i] = [x.mean(), x.max(), (x > 0).float().mean()]
    return stats


def load(path, device):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    return model.to(device).eval()


def lds(run_path: Path) -> float:
    with open(run_path / "validation.csv") as f:
        rows = list(csv.DictReader(f))
    diffs = [float(r["diff"]) for r in rows]
    sums = [float(r["score_sum"]) for r in rows]
    return float(spearmanr(diffs, sums).statistic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/compare_wikitext/neurons.yaml")
    parser.add_argument(
        "--model", default="runs/compare_wikitext/random/retrained/base"
    )
    parser.add_argument("--pretrained", default="gpt2")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="runs/compare_wikitext/neurons/summary.csv")
    args = parser.parse_args()

    steps = [s["magic"] for s in read_config(args.config)["steps"]]
    neurons = [s["query"]["neuron"] for s in steps]
    data, query = steps[0]["data"], steps[0]["query"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = load_dataset(data["dataset"], split=data["split"])["text"]
    queries = load_dataset(query["dataset"], split=query["split"])["text"]
    stats, query_acts = {}, {}
    for name, path in [("final", args.model), ("pretrained", args.pretrained)]:
        model = load(path, args.device)
        stats[name] = chunk_activations(model, tokenizer, texts, neurons, args.device)
        query_acts[name] = chunk_activations(
            model, tokenizer, queries, neurons, args.device
        )[0].mean(1)

    k = round(0.01 * len(texts))
    header = [
        "neuron",
        "lds",
        "spearman_mean",
        "spearman_max",
        "spearman_frac",
        "spearman_abs_mean",
        "spearman_delta_mean",
        "spearman_pre_final",
        "top1pct_mean_act",
        "all_mean_act",
        "query_act_pre",
        "query_act_final",
    ]
    rows = []
    for j, (step, neuron) in enumerate(zip(steps, neurons)):
        run_path = Path(step["run_path"])
        scores, _ = load_scores_loss_signed(str(run_path / "scores"))
        # Loss-signed: the most negative scores raise the activation most.
        influence = -scores.double().reshape(-1).numpy()
        mean, peak, frac = stats["final"][:, j]
        pre = stats["pretrained"][0, j]
        top = np.argsort(-influence)[:k]
        rows.append(
            [
                neuron,
                lds(run_path),
                spearmanr(influence, mean).statistic,
                spearmanr(influence, peak).statistic,
                spearmanr(influence, frac).statistic,
                spearmanr(np.abs(influence), mean).statistic,
                spearmanr(influence, mean - pre).statistic,
                spearmanr(pre, mean).statistic,
                mean[top].mean(),
                mean.mean(),
                query_acts["pretrained"][j],
                query_acts["final"][j],
            ]
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    for name, s in stats.items():
        np.save(Path(args.out).with_name(f"activations_{name}.npy"), s)

    print(" ".join(f"{h:>19}" for h in header))
    for row in rows:
        print(f"{row[0]:>19} " + " ".join(f"{v:19.4f}" for v in row[1:]))
    means = np.array([r[1:] for r in rows]).mean(0)
    print(f"{'mean':>19} " + " ".join(f"{v:19.4f}" for v in means))


if __name__ == "__main__":
    main()
