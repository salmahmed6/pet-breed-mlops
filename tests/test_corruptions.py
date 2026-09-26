from pathlib import Path

import pytest
from PIL import Image, ImageChops

from pet_breed_mlops.corruptions import (
    CORRUPTION_TYPES,
    SEVERITIES,
    apply_corruption,
)


@pytest.fixture
def test_image() -> Image.Image:
    image = Image.new("RGB", (128, 96))

    pixels = image.load()

    for x in range(image.width):
        for y in range(image.height):
            pixels[x, y] = (
                x % 256,
                y % 256,
                (x + y) % 256,
            )

    return image


def test_all_corruptions_and_severities_are_supported(
    test_image: Image.Image,
) -> None:
    for corruption in CORRUPTION_TYPES:
        for severity in SEVERITIES:
            result = apply_corruption(
                test_image,
                corruption,
                severity,
            )

            assert isinstance(result, Image.Image)


def test_corruption_does_not_modify_original(
    test_image: Image.Image,
) -> None:
    original = test_image.copy()

    apply_corruption(
        test_image,
        "gaussian_blur",
        2,
    )

    difference = ImageChops.difference(
        test_image,
        original,
    )

    assert difference.getbbox() is None


def test_corruptions_are_deterministic(
    test_image: Image.Image,
) -> None:
    first = apply_corruption(
        test_image,
        "gaussian_blur",
        2,
    )

    second = apply_corruption(
        test_image,
        "gaussian_blur",
        2,
    )

    difference = ImageChops.difference(first, second)

    assert difference.getbbox() is None


def test_downscale_upscale_keeps_original_dimensions(
    test_image: Image.Image,
) -> None:
    result = apply_corruption(
        test_image,
        "downscale_upscale",
        2,
    )

    assert result.size == test_image.size


def test_invalid_severity_is_rejected(
    test_image: Image.Image,
) -> None:
    with pytest.raises(ValueError, match="Severity"):
        apply_corruption(
            test_image,
            "gaussian_blur",
            4,
        )


def test_invalid_corruption_is_rejected(
    test_image: Image.Image,
) -> None:
    with pytest.raises(ValueError, match="Unknown corruption"):
        apply_corruption(
            test_image,
            "not_a_real_corruption",
            1,
        )


def test_jpeg_corruption_returns_rgb_image(
    test_image: Image.Image,
) -> None:
    result = apply_corruption(
        test_image,
        "jpeg_quality",
        3,
    )

    assert result.mode == "RGB"


def test_corruption_module_is_committed() -> None:
    assert Path("src/pet_breed_mlops/corruptions.py").is_file()
