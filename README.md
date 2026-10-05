# taxi-train-image

Образ для Runpod: unsloth + torch (CUDA 12.8) без установки на каждом Pod'е. Собирается GitHub Actions в
`ghcr.io/itiligator/taxi-train-image:cu128`. Код обучения — в приватном репо `itiligator/taxi-parser`.

Pod: image `ghcr.io/itiligator/taxi-train-image:cu128`, SSH включён; дальше залить архив с данными и
`python3 train.py …` (без venv и без `setup.sh`).
