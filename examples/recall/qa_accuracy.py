"""Measure how well a recall-experiment model answers its QA queries.

Interprets the recall numbers: the model is trained only on statements, so it
may store the facts without being able to produce answers in Q/A format
(cf. Krasheninnikov et al., 2023, where QA pairs for other entities are
co-trained to teach the format). Reports greedy exact-match accuracy and the
mean per-token log-prob of the gold answer, for both the trained model and
its base model as a control.

Example:
    python -m examples.recall.qa_accuracy runs/recall/train/hf_model \
        --query_dataset runs/recall/data/query --base_model HuggingFaceTB/SmolLM2-135M
"""

from dataclasses import dataclass

import torch
from simple_parsing import field, parse
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.data import load_data_string


@dataclass
class QaAccuracyConfig:
    """Config for the QA accuracy check."""

    model: str = field(positional=True)
    """Trained model to evaluate."""

    query_dataset: str = ""
    """Query dataset with input_ids/labels (see prepare_recall_data.py)."""

    base_model: str = ""
    """Optional untrained control model for comparison."""

    max_new_tokens: int = 16


@torch.inference_mode()
def evaluate(model_name: str, query_ds, tokenizer, max_new_tokens: int):
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    exact, logprobs = [], []
    for row in query_ds:
        input_ids = torch.tensor([row["input_ids"]]).cuda()
        labels = torch.tensor([row["labels"]]).cuda()

        # Mean answer-token log-prob under teacher forcing
        logits = model(input_ids).logits[:, :-1]
        targets = labels[:, 1:]
        mask = targets != -100
        token_logprobs = torch.log_softmax(logits.float(), dim=-1)
        gathered = token_logprobs.gather(-1, targets.clamp_min(0).unsqueeze(-1))
        logprobs.append((gathered.squeeze(-1)[mask]).mean().item())

        # Greedy generation from the prompt (everything before the answer)
        prompt_len = int((labels == -100).sum())
        out = model.generate(
            input_ids[:, :prompt_len],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        exact.append(completion.strip().startswith(row["answer"].strip()))

    del model
    return sum(exact) / len(exact), sum(logprobs) / len(logprobs)


def main():
    run_cfg = parse(QaAccuracyConfig)
    assert run_cfg.query_dataset, "--query_dataset must be provided"

    query_ds = load_data_string(run_cfg.query_dataset)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model)

    acc, lp = evaluate(run_cfg.model, query_ds, tokenizer, run_cfg.max_new_tokens)
    print(f"{run_cfg.model}: exact-match {acc:.3f}, mean answer log-prob {lp:.3f}")

    if run_cfg.base_model:
        acc, lp = evaluate(
            run_cfg.base_model, query_ds, tokenizer, run_cfg.max_new_tokens
        )
        print(
            f"{run_cfg.base_model} (control): exact-match {acc:.3f}, "
            f"mean answer log-prob {lp:.3f}"
        )


if __name__ == "__main__":
    main()
