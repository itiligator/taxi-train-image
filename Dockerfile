# Образ для дообучения парсера тредов taxi на арендованной GPU (Runpod): unsloth уже стоит, torch — под CUDA 12.8,
# поэтому работает и на хостах с драйвером 12.x, и на 13.x. Данных и скриптов в образе нет — они приходят архивом.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install --no-cache-dir --break-system-packages -U uv \
 && uv pip install --system --break-system-packages --no-cache --torch-backend cu128 unsloth \
 && python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), "torch не под CUDA 12.8"
import trl, peft, transformers, datasets
print("trl", trl.__version__, "peft", peft.__version__, "transformers", transformers.__version__)
PY
