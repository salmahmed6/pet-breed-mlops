from scripts.download_dataset import EXPECTED_CLASSES, EXPECTED_IMAGES


def test_oxford_pet_expected_constants() -> None:
    assert EXPECTED_IMAGES == 7_349
    assert EXPECTED_CLASSES == 37