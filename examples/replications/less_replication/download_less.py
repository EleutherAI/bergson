import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_less(rank: int = 0):
    final_path = Path("runs/less/extracted_data")

    # Skip if already extracted or not main process
    if (final_path / "data").exists() or rank != 0:
        return final_path

    # Download the zip file
    path = hf_hub_download(
        repo_id="princeton-nlp/less_data", filename="less-data.zip", repo_type="dataset"
    )

    # Unzip it
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(final_path)

    return final_path


if __name__ == "__main__":
    download_less()
