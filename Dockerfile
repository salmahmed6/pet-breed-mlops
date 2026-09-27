FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY docker/constraints-cpu.txt ./docker/constraints-cpu.txt
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
       --constraint docker/constraints-cpu.txt \
       "torch==2.6.0+cpu" "torchvision==0.21.0+cpu" \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
       --constraint docker/constraints-cpu.txt .

CMD ["python", "-c", "import pet_breed_mlops; print('pet-breed-mlops CPU image ready')"]
