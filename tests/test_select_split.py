from datasets import Dataset, DatasetDict

from bergson.data import load_data_string


def _saved(tmp_path, ds):
    path = str(tmp_path / "ds")
    ds.save_to_disk(path)
    return path


def test_saved_dataset_takes_split_slices(tmp_path):
    path = _saved(tmp_path, Dataset.from_dict({"x": list(range(10))}))

    assert load_data_string(path, "train")["x"] == list(range(10))
    assert load_data_string(path, "train[:3]")["x"] == [0, 1, 2]
    assert load_data_string(path, "train[50%:60%]")["x"] == [5]
    assert load_data_string(path, "train[8:]+train[:1]")["x"] == [8, 9, 0]


def test_saved_dataset_dict_takes_split_slices(tmp_path):
    path = _saved(
        tmp_path,
        DatasetDict(
            train=Dataset.from_dict({"x": list(range(10))}),
            test=Dataset.from_dict({"x": list(range(100, 104))}),
        ),
    )

    assert load_data_string(path, "test")["x"] == [100, 101, 102, 103]
    assert load_data_string(path, "test[:2]")["x"] == [100, 101]
    assert load_data_string(path, "train[:1]+test[-1:]")["x"] == [0, 103]


def test_loaded_split_is_sliced_once(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("x\n" + "\n".join(str(i) for i in range(8)) + "\n")

    assert load_data_string(str(path), "train[:50%]")["x"] == [0, 1, 2, 3]
