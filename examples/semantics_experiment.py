import subprocess
from pathlib import Path

import torch
from datasets import load_dataset

from bergson import load_gradient_dataset

dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# Build Bergson index
run_path = Path("runs/math-500/gemma")
cmd = [
    "bergson",
    "build",
    str(run_path),
    "--model",
    "google/gemma-3-4b-it",
    "--dataset",
    "HuggingFaceH4/MATH-500",
    "--drop_columns",
    "False",
    "--split",
    "test",
    "--prompt_column",
    "problem",
    "--completion_column",
    "answer",
]
print(" ".join(cmd))

if not run_path.exists():
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

# Check whether items with the same subject value have a greater cosine similarity score
# Than items from dissimilar subjects

gradient_ds = load_gradient_dataset(run_path, structured=False)

subjects = gradient_ds["subject"]

# Compute cosine similarity between all items' gradients
gradients = torch.tensor(gradient_ds["gradients"], device="cuda")
gradients /= gradients.norm(dim=1, keepdim=True)
similarities = gradients @ gradients.T


# Check whether items with the same subject value have a greater cosine similarity score
# Than items from dissimilar subjects
intra_subject_similarities = []
inter_subject_similarities = []

for i in range(len(gradients)):
    for j in range(i + 1, len(gradients)):
        if subjects[i] == subjects[j]:
            intra_subject_similarities.append(similarities[i, j])
        else:
            inter_subject_similarities.append(similarities[i, j])


mean_intra_subject_similarity = torch.mean(torch.tensor(intra_subject_similarities))
mean_inter_subject_similarity = torch.mean(torch.tensor(inter_subject_similarities))
print(f"Intra-subject similarity mean: {mean_intra_subject_similarity}")
print(f"Inter-subject similarity mean: {mean_inter_subject_similarity}")

breakpoint()
