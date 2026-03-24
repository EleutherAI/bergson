import csv
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Generator

from transformers import AutoTokenizer

from bergson import Attributor, FaissConfig
from bergson.config import IndexConfig, QueryConfig
from bergson.data import load_data_string
from bergson.utils.utils import setup_reproducibility
from bergson.utils.worker_utils import setup_model_and_peft


@contextmanager
def csv_recorder(path: str) -> Generator[Callable | None, None, None]:
    """Open a CSV file for appending query results, or yield None if no path given.

    Yields a callable ``record(row)`` that writes a row and flushes immediately,
    so data survives interrupted sessions.
    """
    if not path:
        yield None
        return

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Check whether headers are needed before opening in append mode,
    # because tell() is unreliable in append mode on some platforms.
    needs_header = not file_path.exists() or file_path.stat().st_size == 0

    with open(file_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if needs_header:
            writer.writerow(["query", "direction", "result", "result_index", "score"])
            csv_file.flush()

        def record(row: list[Any]) -> None:
            writer.writerow(row)
            csv_file.flush()

        yield record


def query(
    query_cfg: QueryConfig,
):
    """
    Run an interactive CLI session that queries a pre-built gradient index.

    Parameters
    ----------
    cfg : QueryConfig
        Configuration describing the index path, HF model to load, and dataset field
        used to print the retrieved documents.
    """
    index_cfg = IndexConfig.load_json(Path(query_cfg.index) / "index_config.json")

    if index_cfg.debug:
        setup_reproducibility()

    # Load a different model than the one the index was built for, e.g.
    # a different checkpoint.
    if query_cfg.model:
        query_index_cfg = IndexConfig(
            **{k: v for k, v in asdict(index_cfg).items() if k != "model"},
            model=query_cfg.model,
        )
        tokenizer = AutoTokenizer.from_pretrained(query_cfg.model)
        model, target_modules = setup_model_and_peft(
            query_index_cfg, device_map_auto=query_cfg.device_map_auto
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(index_cfg.model)
        model, target_modules = setup_model_and_peft(
            index_cfg, device_map_auto=query_cfg.device_map_auto
        )

    ds = load_data_string(
        index_cfg.data.dataset,
        index_cfg.data.split,
        index_cfg.data.subset,
        index_cfg.data.data_args,
    )

    faiss_cfg = FaissConfig() if query_cfg.faiss else None
    attr = Attributor(Path(query_cfg.index), device="cuda", faiss_cfg=faiss_cfg)

    # Get the device of the first model parameter for multi-GPU setups
    model_device = next(model.parameters()).device

    with csv_recorder(query_cfg.record) as record:
        while True:
            query = input("Enter your query: ")
            if query.lower() == "exit":
                break

            # Tokenize the query
            inputs = tokenizer(query, return_tensors="pt").to(model_device)
            x = inputs["input_ids"]

            # Retrieve both the highest and lowest influence samples
            with attr.trace(
                model.base_model,
                query_cfg.top_k,
                modules=target_modules,
                reverse=False,
            ) as top_result:
                model(x, labels=x).loss.backward()
                model.zero_grad()

            with attr.trace(
                model.base_model,
                query_cfg.top_k,
                modules=target_modules,
                reverse=True,
            ) as bottom_result:
                model(x, labels=x).loss.backward()
                model.zero_grad()

            for direction, result in [("Top", top_result), ("Bottom", bottom_result)]:
                print(f"\n{direction} {query_cfg.top_k} results for '{query}':")
                for i, (d, idx) in enumerate(
                    zip(result.scores.squeeze(), result.indices.squeeze())
                ):
                    if idx.item() == -1:
                        print("Found invalid result, skipping")
                        continue

                    idx_int = int(idx.item())
                    score = d.item()
                    text = str(ds[idx_int][query_cfg.text_field])  # type: ignore[arg-type]
                    print(text[:2000])
                    if len(text) > 2000:
                        print(". . .")

                    print(f"{i + 1}: (distance: {score:.4f})")

                    if record is not None:
                        record([query, direction, text, idx_int, score])
