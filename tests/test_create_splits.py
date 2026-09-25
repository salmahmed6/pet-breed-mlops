from pathlib import Path

import yaml

from scripts.create_splits import create_splits


def write_annotations(root: Path) -> None:
    """Create a small synthetic Oxford-IIIT Pet-style annotation set."""
    annotations = root / "annotations"
    annotations.mkdir(parents=True)

    trainval = "\n".join(f"cat_{index} 1 1" for index in range(10))
    test = "\n".join(f"dog_{index} 2 1" for index in range(4))

    (annotations / "trainval.txt").write_text(
        trainval + "\n",
        encoding="utf-8",
    )
    (annotations / "test.txt").write_text(
        test + "\n",
        encoding="utf-8",
    )


def load_split_file(path: Path) -> dict:
    """Load generated split YAML."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_create_splits_is_deterministic(tmp_path: Path) -> None:
    """The same seed and input must produce identical splits."""
    dataset_root = tmp_path / "dataset"
    write_annotations(dataset_root)

    output_one = tmp_path / "splits_one.yaml"
    output_two = tmp_path / "splits_two.yaml"

    create_splits(
        dataset_root,
        output_one,
        seed=42,
        validation_size=0.2,
    )

    create_splits(
        dataset_root,
        output_two,
        seed=42,
        validation_size=0.2,
    )

    assert output_one.read_text(encoding="utf-8") == output_two.read_text(encoding="utf-8")


def test_create_splits_preserves_test_and_has_no_overlap(
    tmp_path: Path,
) -> None:
    """Train, validation, and test must be disjoint."""
    dataset_root = tmp_path / "dataset"
    write_annotations(dataset_root)

    output = tmp_path / "splits.yaml"

    create_splits(
        dataset_root,
        output,
        seed=42,
        validation_size=0.2,
    )

    splits = load_split_file(output)

    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])

    assert len(train) == 8
    assert len(validation) == 2
    assert len(test) == 4

    assert not train & validation
    assert not train & test
    assert not validation & test

    assert train | validation == {f"cat_{index}.jpg" for index in range(10)}

    assert test == {f"dog_{index}.jpg" for index in range(4)}
