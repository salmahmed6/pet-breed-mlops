# Dataset

This document will describe the Oxford-IIIT Pet dataset, dataset acquisition,
fixed train/validation/test splits, the 37-class label mapping, manifest format,
validation rules, and corruption generation process.

Detailed dataset documentation will be completed as the data pipeline is implemented.

# Dataset

## Oxford-IIIT Pet

The final project Track C uses the Oxford-IIIT Pet dataset.

The project handbook specifies:

- 7,349 images
- 37 cat and dog breeds
- roughly 200 images per class
- approximately 800 MB

The handbook specifies using the torchvision download mechanism:

`torchvision.datasets.OxfordIIITPet(download=True)`

### Dataset source

Official Oxford VGG dataset:

https://www.robots.ox.ac.uk/~vgg/data/pets/

Torchvision dataset implementation:

https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html

### Local location

Raw dataset:

`data/raw/`

The raw dataset is not committed directly to Git. It will be tracked through DVC.

### Reproducibility

Dataset acquisition is implemented by:

`scripts/download_dataset.py`

The script:

1. Downloads the `trainval` split.
2. Downloads the `test` split.
3. Validates the expected total image count.
4. Validates the expected 37 classes.
5. Does not download during validation.

The test set must remain protected from training and tuning workflows.