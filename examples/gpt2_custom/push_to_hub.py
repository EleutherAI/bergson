"""Upload the custom GPT-2 to the Hub with its modeling code.

Usage:
    python examples/gpt2_custom/push_to_hub.py [--repo EleutherAI/gpt2-custom] [--public]

Publishes stock ``gpt2`` weights wrapped in ``GPT2CustomLMHeadModel`` with
``arch_mod="none"`` plus the modeling module (``trust_remote_code``). Variants
load from the same repo by overriding the config at load time:

    AutoModelForCausalLM.from_pretrained(repo, arch_mod="qk_norm",
                                         trust_remote_code=True)

``arch_mod="none"`` is bit-identical to stock ``gpt2`` (tested at both toy and
124M scale), so this repo is a drop-in base for the architecture-modification
experiments and their control row.
"""

import argparse

from transformers import GPT2LMHeadModel

from bergson.models.modeling_gpt2_custom import GPT2CustomConfig, GPT2CustomLMHeadModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="EleutherAI/gpt2-custom")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    stock = GPT2LMHeadModel.from_pretrained("gpt2")
    cfg = GPT2CustomConfig(arch_mod="none", **stock.config.to_diff_dict())
    model = GPT2CustomLMHeadModel(cfg)
    model.load_state_dict(stock.state_dict(), strict=True)

    GPT2CustomConfig.register_for_auto_class()
    GPT2CustomLMHeadModel.register_for_auto_class("AutoModelForCausalLM")
    model.push_to_hub(args.repo, private=not args.public)
    print(f"pushed {args.repo} (private={not args.public})")


if __name__ == "__main__":
    main()
