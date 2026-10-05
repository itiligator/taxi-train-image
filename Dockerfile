# Образ для дообучения парсера тредов taxi на арендованной GPU (Runpod): unsloth уже стоит, torch — под CUDA 12.8,
# поэтому работает и на хостах с драйвером 12.x, и на 13.x. Данных и скриптов в образе нет — они приходят архивом.
#
# Версии важны для скорости: на базовых torch 2.8 / xformers 0.0.33 обучение шло в ~6 раз медленнее (3.5 с/шаг
# вместо 0.55, память 11 ГБ вместо 6) — unsloth не грузит быстрые расширения. Рабочая связка: torch 2.11 cu128,
# xformers 0.0.35, triton 3.6 (проверено на RTX 4090: 60 шагов за 34 с).
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install --no-cache-dir --break-system-packages -U uv \
 && uv pip install --system --break-system-packages --no-cache --torch-backend cu128 \
      unsloth "torch>=2.11,<2.12" torchvision torchaudio "xformers==0.0.35" \
 && python3 - <<'PY'
import torch, xformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "xformers", xformers.__version__)
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), "torch не под CUDA 12.8"
assert torch.__version__.startswith("2.11"), "нужен torch 2.11"
assert xformers.__version__ == "0.0.35", "нужен xformers 0.0.35"
import trl, peft, transformers, datasets
print("trl", trl.__version__, "peft", peft.__version__, "transformers", transformers.__version__)
PY
