# Dataset

## Oxford-IIIT Pet

The final project Track C uses the Oxford-IIIT Pet dataset.

The project handbook specifies:

- 7,349 images
- 37 cat and dog breeds
- roughly 200 images per class
- approximately 800 MB

The official Oxford VGG page provides the dataset as `images.tar.gz` and `annotations.tar.gz`. The project uses the torchvision acquisition mechanism required by the handbook. Torchvision pins the two official resources by MD5 and extracts them under `oxford-iiit-pet`.

### Dataset source

Official Oxford VGG dataset:

https://www.robots.ox.ac.uk/~vgg/data/pets/

Torchvision dataset implementation:

https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html

### Expected split sizes

The official annotation files contain:

- `trainval.txt`: 3,680 images
- `test.txt`: 3,669 images
- total: 7,349 images

The test split must remain protected from training and tuning workflows. The later split-management stage will carve validation from trainval using a fixed seed.

### Local location

The raw dataset is expected under:

`data/raw/oxford-iiit-pet/`

Raw data is not committed directly to Git. It will be tracked through DVC in the dedicated DVC issue.

### Acquisition and validation

Dataset acquisition and validation are implemented by:

`scripts/download_dataset.py`

Download and validate from a clean environment:

```powershell
python .\scripts\download_dataset.py --download
```

Validate an existing local copy without downloading:

```powershell
python .\scripts\download_dataset.py
```

Validation checks:

1. `trainval.txt` contains exactly 3,680 images.
2. `test.txt` contains exactly 3,669 images.
3. Trainval and test image IDs do not overlap.
4. Exactly 37 class labels are present.
5. The image directory contains exactly the 7,349 images referenced by the official split files.
6. Missing or extra images cause validation to fail.

The script does not commit or upload raw image data to Git.
