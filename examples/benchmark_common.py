from dataclasses import dataclass
from datetime import datetime

DEFAULT_DATASET = "EleutherAI/SmolLM2-135M-10B"

@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    params: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "pythia-14m": ModelSpec("pythia-14m", "EleutherAI/pythia-14m", 14_000_000),
    "pythia-70m": ModelSpec("pythia-70m", "EleutherAI/pythia-70m", 70_000_000),
    "pythia-160m": ModelSpec("pythia-160m", "EleutherAI/pythia-160m", 160_000_000),
    "pythia-1b": ModelSpec("pythia-1b", "EleutherAI/pythia-1b", 1_000_000_000),
    "pythia-6.9b": ModelSpec("pythia-6.9b", "EleutherAI/pythia-6.9b", 6_900_000_000),
    "pythia-12b": ModelSpec("pythia-12b", "EleutherAI/pythia-12b", 12_000_000_000),
}

def timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000_000:
        value = tokens / 1_000_000_000
        suffix = "B"
    elif tokens >= 1_000_000:
        value = tokens / 1_000_000
        suffix = "M"
    elif tokens >= 1_000:
        value = tokens / 1_000
        suffix = "K"
    else:
        return str(tokens)
    if value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.2f}{suffix}"


def parse_tokens(value: str) -> int:
    text = value.strip().lower().replace(",", "")
    if text.endswith("tokens"):
        text = text[:-6]
    if not text:
        raise ValueError("empty token spec")

    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    unit = 1
    if text[-1] in suffixes:
        unit = suffixes[text[-1]]
        text = text[:-1]
    number = float(text)
    return int(number * unit)
