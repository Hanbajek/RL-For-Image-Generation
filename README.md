# BK-SDM LoRA + RL Training & Image Generation

Este repositorio contiene un conjunto de herramientas para entrenar adaptadores **LoRA** sobre el modelo de difusión BK-SDM utilizando una mezcla de **entrenamiento supervisado (MSE)** y **aprendizaje por refuerzo (REINFORCE)** con tres evaluadores: **CLIPScore**, **ImageReward** y **Qwen-VL**.  
También incluye scripts para **generar imágenes** usando los LoRA entrenados.

---

## 🌟 Características principales

- Generación de embeddings de texto usando el tokenizer y text encoder del modelo base.
- Entrenamiento de LoRA con:
  - MSE (predicción de ruido)
  - RL con ventaja normalizada
  - Mezcla de recompensas (CLIP + ImageReward + Qwen-VL)
- Generación de imágenes usando cualquier LoRA entrenado.
- Compatible con GPU (CUDA) y CPU para preprocesamiento.

---

## 📂 Estructura del repositorio

```
.
├── Checkpoints/                    
│   └── pytorch_lora_weights.safetensors
│
├── Embeds/                        
│   ├── embed_prompts_local.py
│   ├── requirements2.txt
│   └── prompts.txt
│
├── Generation/                     
│   ├── gen_BKSDM.py
│   └── gen_with_lora.py
│
├── Requirement.txt                 
│
└── train RL BASELINE.py             

```

---

## 🧪 Requisitos e instalación

Se recomienda usar dos entornos: uno para embeddings y otro para entrenamiento/generación.

### 🔹 Entorno para embeddings:

```bash
conda create -n env_text python=3.10 -y
conda activate env_text
pip install torch transformers tokenizers safetensors
```

### 🔹 Entorno para entrenamiento y generación:

```bash
conda create -n env_diffusers python=3.10 -y
conda activate env_diffusers
pip install torch torchvision diffusers transformers accelerate peft image-reward qwen-vl-utils safetensors pillow tqdm
```

---

## 🚀 Uso

### 1) Generar embeddings:

```bash
python src/rl/embed_prompts_local.py   --model_dir "./bk2m"   --prompts_file "artifacts/topset/prompts_modificado.txt"   --out "artifacts/topset/embeds3.pt"
```

### 2) Entrenar LoRA con RL:

```bash
python src/rl/train_lora_stage1_reinforce_3eval.py   --topset "artifacts/topset"   --embeds "artifacts/topset/embeds3.pt"   --out_dir "artifacts/samples_lora_rewardMix"
```

### 3) Generar imágenes con LoRA:

```bash
python src/generator/gen_with_lora.py   --model_dir "./bk2m"   --lora "artifacts/samples_lora_rewardMix/fold1"   --prompts "artifacts/topset/prompts_modificado.txt"   --out_dir "artifacts/samples_lora_rewardMix/png_fold1"
```

---

## 📜 Licencia

Este proyecto puede licenciarse bajo MIT, Apache-2.0 o GPL-3.0.
