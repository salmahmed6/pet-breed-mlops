"""Deterministic image corruption functions for Track C."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter

CORRUPTION_TYPES = (
    "gaussian_blur",
    "brightness_up",
    "brightness_down",
    "jpeg_quality",
    "downscale_upscale",
    "motion_blur",
)

SEVERITIES = (1, 2, 3)


def gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    """Apply deterministic Gaussian blur."""
    radius = {1: 1.0, 2: 2.0, 3: 3.0}[severity]
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def brightness_up(image: Image.Image, severity: int) -> Image.Image:
    """Increase brightness deterministically."""
    factor = {1: 1.15, 2: 1.35, 3: 1.60}[severity]
    return ImageEnhance.Brightness(image).enhance(factor)


def brightness_down(image: Image.Image, severity: int) -> Image.Image:
    """Decrease brightness deterministically."""
    factor = {1: 0.85, 2: 0.65, 3: 0.45}[severity]
    return ImageEnhance.Brightness(image).enhance(factor)


def jpeg_quality(image: Image.Image, severity: int) -> Image.Image:
    """Apply deterministic JPEG compression."""
    quality = {1: 70, 2: 50, 3: 30}[severity]

    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
    )
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        converted: Image.Image = compressed.convert("RGB")
        result: Image.Image = converted.copy()
        return result


def downscale_upscale(image: Image.Image, severity: int) -> Image.Image:
    """Downscale and restore the image to its original dimensions."""
    target = {1: 128, 2: 96, 3: 64}[severity]

    original_size = image.size

    reduced = image.resize(
        (target, target),
        Image.Resampling.BILINEAR,
    )

    return reduced.resize(
        original_size,
        Image.Resampling.BILINEAR,
    )


def motion_blur(image: Image.Image, severity: int) -> Image.Image:
    """Apply deterministic horizontal motion blur."""
    radius = {1: 3, 2: 5, 3: 9}[severity]

    source = image.convert("RGB")
    width, height = source.size

    result = Image.new("RGB", (width, height))

    source_pixels = source.load()
    result_pixels = result.load()

    if source_pixels is None or result_pixels is None:
        raise RuntimeError("Unable to access image pixels.")

    for y in range(height):
        for x in range(width):
            total_r = 0
            total_g = 0
            total_b = 0
            count = 0

            for offset in range(-radius, radius + 1):
                source_x = x + offset

                if 0 <= source_x < width:
                    pixel = source_pixels[source_x, y]

                    if not isinstance(pixel, tuple) or len(pixel) != 3:
                        raise RuntimeError("Expected an RGB pixel.")

                    red, green, blue = pixel
                    total_r += red
                    total_g += green
                    total_b += blue
                    count += 1

            result_pixels[x, y] = (
                total_r // count,
                total_g // count,
                total_b // count,
            )

    return result


def apply_corruption(
    image: Image.Image,
    corruption: str,
    severity: int,
) -> Image.Image:
    """Apply one supported corruption at a supported severity."""
    if severity not in SEVERITIES:
        raise ValueError(f"Severity must be one of {SEVERITIES}, got {severity}.")

    functions = {
        "gaussian_blur": gaussian_blur,
        "brightness_up": brightness_up,
        "brightness_down": brightness_down,
        "jpeg_quality": jpeg_quality,
        "downscale_upscale": downscale_upscale,
        "motion_blur": motion_blur,
    }

    if corruption not in functions:
        raise ValueError(
            f"Unknown corruption '{corruption}'. Supported values: {CORRUPTION_TYPES}."
        )

    return functions[corruption](image.copy(), severity)
