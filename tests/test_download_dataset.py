from pathlib import Path

from scripts.download_dataset import (
    EXPECTED_CLASSES,
    EXPECTED_IMAGES,
    EXPECTED_TEST,
    EXPECTED_TRAINVAL,
    read_split,
)


def test_oxford_pet_expected_constants() -> None:
    assert EXPECTED_IMAGES == 7_349
    assert EXPECTED_CLASSES == 37
    assert EXPECTED_TRAINVAL == 3_680
    assert EXPECTED_TEST == 3_669


def test_read_split(tmp_path: Path) -> None:
    split = tmp_path / "trainval.txt"
    split.write_text("cat_1 1 1\ndog_2 37 1\n", encoding="utf-8")

    names, labels = read_split(split)

    assert names == {"cat_1.jpg", "dog_2.jpg"}
    assert labels == {1, 37}
