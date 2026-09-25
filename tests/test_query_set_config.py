"""One query spec for every pipeline, and the shim that reads the old fields.

TODO Lucia Quirke delete 12/2026: the migration tests go with the shim.
"""

import warnings

import pytest

from bergson.build import build_query
from bergson.cli.commands import Ekfac, Trackstar, Validate
from bergson.config.config import IndexConfig, PreprocessConfig, QuerySetConfig
from bergson.config.validation import migrate_query_config

QUERY = {"dataset": "EleutherAI/bergson-wikitext-512-chunks", "split": "test[0:2]"}
INDEX = {"run_path": "runs/x", "model": "gpt2", "data": {"dataset": "d"}}
HESSIAN = {"method": "kfac"}


def _query(cmd) -> QuerySetConfig:
    for name in (
        "hessian_pipeline_cfg",
        "approx_unrolling_cfg",
        "trackstar_cfg",
        "trak_cfg",
    ):
        if hasattr(cmd, name):
            return getattr(cmd, name).query
    return cmd.query


def test_old_fields_become_one_query_config():
    old = {"query": dict(QUERY), "query_aggregation": "none"}
    with pytest.warns(FutureWarning):
        new = migrate_query_config(old, legacy_key="query_aggregation", default="mean")
    assert new == {"query": {"data": QUERY, "aggregation": "none"}}


def test_missing_aggregation_takes_the_pipelines_old_default():
    with pytest.warns(FutureWarning):
        new = migrate_query_config(
            {"query": dict(QUERY)}, legacy_key="query_method", default="none"
        )
    assert new["query"]["aggregation"] == "none"


def test_new_style_query_passes_through_unchanged():
    obj = {"query": {"data": QUERY, "aggregation": "sum", "path": ""}, "other": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            migrate_query_config(obj, legacy_key="query_method", default="none") == obj
        )
        assert migrate_query_config(
            {"other": 1}, legacy_key="query_method", default="none"
        ) == {"other": 1}


def test_mixing_old_and_new_fields_is_an_error():
    with pytest.raises(ValueError, match="legacy"):
        migrate_query_config(
            {"query": {"data": QUERY}, "query_method": "mean"},
            legacy_key="query_method",
            default="none",
        )


@pytest.mark.parametrize(
    "cls, payload, legacy_key, old_default",
    [
        (
            Validate,
            {"run_path": "runs/x", "model": "gpt2", "scores": "s"},
            "query_method",
            "none",
        ),
        (
            Ekfac,
            {
                "index_cfg": INDEX,
                "hessian_cfg": HESSIAN,
                "score_cfg": {},
                "preprocess_cfg": {},
                "hessian_pipeline_cfg": {},
            },
            "query_aggregation",
            "mean",
        ),
        (Trackstar, {"index_cfg": INDEX, "trackstar_cfg": {}}, None, "none"),
    ],
)
def test_every_pipeline_reads_old_and_new_yaml(cls, payload, legacy_key, old_default):
    sub = next(
        (
            k
            for k in (
                "hessian_pipeline_cfg",
                "approx_unrolling_cfg",
                "trackstar_cfg",
                "trak_cfg",
            )
            if k in payload
        ),
        None,
    )

    def with_query(block: dict) -> dict:
        obj = {k: (dict(v) if isinstance(v, dict) else v) for k, v in payload.items()}
        (obj[sub] if sub else obj).update(block)
        return obj

    old = {"query": dict(QUERY)}
    if legacy_key:
        old[legacy_key] = "sum"
    with pytest.warns(FutureWarning):
        cmd = cls.from_dict(with_query(old), drop_extra_fields=False)
    q = _query(cmd)
    assert q.data.dataset == QUERY["dataset"] and q.data.split == QUERY["split"]
    assert q.aggregation == ("sum" if legacy_key else old_default)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cmd = cls.from_dict(
            with_query(
                {"query": {"data": QUERY, "aggregation": "sum", "path": "runs/q"}}
            ),
            drop_extra_fields=False,
        )
    q = _query(cmd)
    assert (q.data.split, q.aggregation, q.path) == (QUERY["split"], "sum", "runs/q")


def test_trackstar_old_yaml_takes_aggregation_from_preprocess():
    obj = {
        "index_cfg": INDEX,
        "trackstar_cfg": {
            "query": dict(QUERY),
            "preprocess_cfg": {"aggregation": "mean"},
        },
    }
    with pytest.warns(FutureWarning):
        cmd = Trackstar.from_dict(obj, drop_extra_fields=False)
    assert cmd.trackstar_cfg.query.aggregation == "mean"


def test_build_query_returns_an_existing_index_without_building(tmp_path):
    existing = tmp_path / "query"
    existing.mkdir()
    cfg = IndexConfig(run_path=str(tmp_path / "unused"), model="gpt2")
    query = QuerySetConfig(path=str(existing))
    assert build_query(cfg, query, PreprocessConfig()) == str(existing)
    assert not (tmp_path / "unused").exists()
    with pytest.raises(FileNotFoundError):
        build_query(
            cfg, QuerySetConfig(path=str(tmp_path / "missing")), PreprocessConfig()
        )
