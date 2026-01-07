from dataclasses import dataclass

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