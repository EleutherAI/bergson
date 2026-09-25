"""Per-epoch shuffling of the MAGIC training stream."""

from datasets import Dataset

from bergson.magic.cli import shuffled_epochs

N = 12


def _ds() -> Dataset:
    return Dataset.from_dict({"doc_ids": list(range(N))})


def _order(ds: Dataset) -> list[int]:
    return list(ds["doc_ids"])


def test_each_epoch_is_shuffled_independently():
    out = _order(shuffled_epochs(_ds(), seed=0, num_epochs=3))

    epochs = [out[i * N : (i + 1) * N] for i in range(3)]
    assert len(out) == 3 * N
    for epoch in epochs:
        assert sorted(epoch) == list(range(N))
    assert epochs[0] != epochs[1]
    assert epochs[1] != epochs[2]


def test_deterministic_for_a_given_seed():
    a = _order(shuffled_epochs(_ds(), seed=7, num_epochs=3))
    b = _order(shuffled_epochs(_ds(), seed=7, num_epochs=3))
    assert a == b

    c = _order(shuffled_epochs(_ds(), seed=8, num_epochs=3))
    assert a != c
