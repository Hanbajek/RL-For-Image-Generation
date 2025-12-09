import os
import sys
import argparse
import random
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

from diffusers import StableDiffusionPipeline
from ImageReward import load as load_image_reward
from transformers import (
    CLIPModel,
    CLIPProcessor,
)

# =========================
# Dataset imágenes + prompts
# =========================
class TxtImgDataset(Dataset):
    def __init__(self, folder):
        folder = Path(folder)
        self.img_dir = folder / "images3"
        self.files = sorted(list(self.img_dir.glob("*.png")))
        pfile = folder / "prompts_modificado.txt"
        self.prompts = [l.strip() for l in open(pfile, encoding="utf-8") if l.strip()]
        assert len(self.files) == len(self.prompts), "Mismatch images/prompts en topset"
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB").resize((512, 512))
        return self.prompts[idx], self.to_tensor(img)


# =========================
# Carga robusta de embeddings
# =========================
def load_text_embeds_and_prompts(embeds_path: str, topset_dir: str):
    """
    Devuelve (text_embeds, prompt_list)
      - text_embeds: torch.Tensor [N, seq, dim]
      - prompt_list: list[str]
    """
    emb = torch.load(embeds_path, map_location="cpu")

    if isinstance(emb, dict):
        if "text_embeds" in emb and isinstance(emb["text_embeds"], torch.Tensor):
            text_embeds = emb["text_embeds"]
        elif "embeds" in emb and isinstance(emb["embeds"], torch.Tensor):
            text_embeds = emb["embeds"]
        else:
            vals = [v for v in emb.values() if isinstance(v, torch.Tensor)]
            if not vals:
                raise ValueError(f"No encontré tensor en {embeds_path}. Claves: {list(emb.keys())}")
            text_embeds = vals[0]
        prompt_list = emb.get("prompts", None)
    else:
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"Embeddings inválidos en {embeds_path}: tipo {type(emb)}")
        text_embeds = emb
        prompt_list = None

    if prompt_list is None:
        pfile = Path(topset_dir) / "prompts.txt"
        if pfile.exists():
            with open(pfile, "r", encoding="utf-8") as f:
                prompt_list = [l.strip() for l in f if l.strip()]
        else:
            prompt_list = []

    if not isinstance(text_embeds, torch.Tensor):
        raise TypeError(f"El archivo {embeds_path} no contiene un tensor válido")

    return text_embeds, prompt_list


# =========================
# Inyección LoRA con PEFT/diffusers (robusta)
# =========================
def inject_lora_peft(pipe, rank=8, adapter_name="rl_lora"):
    """
    Inyecta LoRA usando PEFT si está disponible;
    si no, usa LoRAConfig de diffusers (cuando existe).
    Compatible con diffusers recientes y peft 0.11.x.
    """
    peft_cfg = None

    # 1) Intentar LoRAConfig de diffusers
    diff_err = None
    try:
        from diffusers.models.lora import LoRAConfig as DiffLoRAConfig
        peft_cfg = DiffLoRAConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            use_scale=False,
        )
        print(" Usando diffusers.models.lora.LoRAConfig")
    except Exception as e_diff:
        diff_err = e_diff
        # 2) Fallback: PEFT puro
        try:
            from peft import LoraConfig as PeftLoraConfig
            peft_cfg = PeftLoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.0,
                bias="none",
            )
            print(" Usando peft.LoraConfig (fallback)")
        except Exception as e_peft:
            raise RuntimeError(
                f"No pude construir config LoRA ni con diffusers ni con peft.\n"
                f"diffusers_error={diff_err}\npeft_error={e_peft}"
            )

    # 3) add_adapter: posibles firmas
    attached = False
    try:
        pipe.unet.add_adapter(peft_cfg, adapter_name=adapter_name)
        attached = True
    except TypeError:
        try:
            pipe.unet.add_adapter(adapter_name=adapter_name, peft_config=peft_cfg)
            attached = True
        except Exception as e:
            raise RuntimeError(f"Fallo al añadir adapter LoRA: {e}")

    if not attached:
        raise RuntimeError("No se pudo adjuntar el adapter LoRA (firmas incompatibles).")

    pipe.set_adapters(adapter_name)
    print(f"Inyección LoRA (adapter='{adapter_name}', r={rank}) activada correctamente.")
    return "peft"


# =========================
# Evaluación en validación (usa CLIP + ImageReward)
# =========================
def evaluate_on_validation(
    pipe,
    dl_val,
    rm,
    clip_model,
    clip_processor,
    device,
    w_clip,
    w_ir,
    max_batches=3,
):
    pipe_was_train = pipe.unet.training
    pipe.unet.eval()
    rm.eval()
    clip_model.eval()

    all_ir = []
    all_clip = []
    all_mix = []

    with torch.no_grad():
        batches_done = 0
        for batch_prompts, _ in dl_val:
            if isinstance(batch_prompts, tuple):
                batch_prompts = list(batch_prompts)

            # 1) GENERAR imágenes con el pipeline
            images = pipe(
                batch_prompts,
                num_inference_steps=20,
                guidance_scale=5.0,
            ).images  # lista PIL

            # 2) ImageReward
            scores_ir = rm.score(batch_prompts, images)
            scores_ir_t = torch.tensor(scores_ir, dtype=torch.float32, device=device)
            ir_val = float(scores_ir_t.mean().item())
            all_ir.append(ir_val)

            # 3) CLIPScore
            clip_inputs = clip_processor(
                text=batch_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            out = clip_model(**clip_inputs)
            img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            cos_sim = (img_emb * txt_emb).sum(dim=-1)
            clip_val = float(cos_sim.mean().item())
            all_clip.append(clip_val)

            # 4) Mezcla normalizada de 2 rewards
            clip_norm = (cos_sim.mean() + 1.0) / 2.0            # ~ [0,1]
            ir_norm = (torch.tanh(scores_ir_t.mean() / 10.0) + 1.0) / 2.0

            total_w = w_clip + w_ir
            if total_w <= 0:
                total_w = 1.0
            wc = w_clip / total_w
            wi = w_ir / total_w

            mix_val = (wc * clip_norm + wi * ir_norm).item()
            all_mix.append(mix_val)

            batches_done += 1
            if batches_done >= max_batches:
                break

    if pipe_was_train:
        pipe.unet.train()

    mean_ir = sum(all_ir) / len(all_ir) if all_ir else 0.0
    mean_clip = sum(all_clip) / len(all_clip) if all_clip else 0.0
    mean_mix = sum(all_mix) / len(all_mix) if all_mix else 0.0

    return mean_ir, mean_clip, mean_mix


# =========================
# Entrenamiento de UN FOLD (BASELINE + 2 evaluadores)
# =========================
def train_one_fold(
    fold_id,
    args,
    train_indices,
    val_indices,
    ds_full,
    text_embeds,
    prompt_list,
    rm,
    clip_model,
    clip_processor,
    device,
):
    print(f"\n==============================")
    print(f"FOLD {fold_id+1} / {args.k_folds}")
    print("==============================\n")

    # 1) Pipeline 
    print("Cargando modelo base BK-SDM para este fold...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    print("Pipeline cargado.\n")

    if hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()
    pipe.unet.train()
    DTYPE = pipe.unet.dtype

    # 2) Inyección LoRA
    print(" Inyectando LoRA...")
    adapter_name = f"rl_lora_fold{fold_id+1}"
    mode = inject_lora_peft(pipe, rank=args.rank, adapter_name=adapter_name)
    print(f"Inyección LoRA completada (modo: {mode}).\n")

    # 3) Dataloaders de este fold
    ds_train = Subset(ds_full, train_indices)
    ds_val = Subset(ds_full, val_indices)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=min(args.batch, 4), shuffle=False)

    print(f"Fold {fold_id+1}: train={len(ds_train)}, val={len(ds_val)}\n")

    # 4) Embeddings + mapa de prompts
    prompt_to_idx = {p: i for i, p in enumerate(prompt_list)} if prompt_list else {}

    # 5) Optimizador
    print("Configurando optimizador...")
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        params,
        lr=args.lr,
        eps=1e-6,
        weight_decay=0.01,
    )
    print(" Optimizador listo.\n")

    to_pil = T.ToPILImage()

    # Baseline para advantage 
    baseline_reward = 0.0
    baseline_init = False

    pbar = tqdm(range(args.steps), desc=f"Fold {fold_id+1} entrenando LoRA (REINFORCE)")
    it = iter(dl_train)

    last_train_mse = 0.0
    last_train_rmix = 0.0
    last_val_ir = 0.0
    last_val_clip = 0.0
    last_val_mix = 0.0

    for step in pbar:
        try:
            batch_prompts, imgs = next(it)
        except StopIteration:
            it = iter(dl_train)
            batch_prompts, imgs = next(it)

        imgs = imgs.to(device, dtype=DTYPE)

        # --- latentes ---
        with torch.no_grad():
            latents = pipe.vae.encode(imgs * 2 - 1).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            latents = latents.to(DTYPE)

        # --- ruido ---
        noise = torch.randn_like(latents, dtype=latents.dtype)
        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        noisy = pipe.scheduler.add_noise(latents, noise, timesteps)

        # --- condición de texto a partir de embeddings ---
        if prompt_to_idx:
            idxs = [prompt_to_idx.get(p, 0) for p in batch_prompts]
            cond = text_embeds[idxs]
        else:
            cond = text_embeds[:imgs.shape[0]]

        cond = cond.to(device=device, dtype=DTYPE)

        # --- forward UNet (política actual) ---
        noise_pred = pipe.unet(noisy, timesteps, encoder_hidden_states=cond).sample

        # --- MSE base ---
        mse = F.mse_loss(noise_pred, noise)

        # =========================
        # BASELINE: log-probs + reward + baseline
        # =========================
        loss_rl = torch.tensor(0.0, device=device)
        r_ir_val = 0.0
        r_clip_val = 0.0
        r_mix_val = 0.0

        if step % args.rl_interval == 0:
            B = noise.shape[0]

            # log-prob aproximada de la política actual
            eps = noise.view(B, -1)
            eps_pred = noise_pred.view(B, -1)
            logp_new = - ((eps - eps_pred) ** 2).mean(dim=1)  # [B]

            # Subconjunto para RL
            B_rl = min(B, args.rl_batch)
            lat_prev = latents[:B_rl]

            with torch.no_grad():
                # Decode imágenes para evaluadores
                img_dec = pipe.vae.decode(lat_prev / pipe.vae.config.scaling_factor).sample
                img_dec = (img_dec / 2 + 0.5).clamp(0, 1)
                img_dec = img_dec.to(torch.float32).cpu()

                pil_imgs = [to_pil(img_dec[i]) for i in range(B_rl)]
                prompts_preview = list(batch_prompts[:B_rl])

                # ===== ImageReward =====
                scores_ir = rm.score(prompts_preview, pil_imgs)
                scores_ir_t = torch.tensor(scores_ir, dtype=torch.float32, device=device)
                ir_norm = (torch.tanh(scores_ir_t / 10.0) + 1.0) / 2.0  # [0,1]

                # ===== CLIP =====
                clip_inputs = clip_processor(
                    text=prompts_preview,
                    images=pil_imgs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(device)
                out_clip = clip_model(**clip_inputs)
                img_emb = out_clip.image_embeds / out_clip.image_embeds.norm(dim=-1, keepdim=True)
                txt_emb = out_clip.text_embeds / out_clip.text_embeds.norm(dim=-1, keepdim=True)

                n_clip = min(img_emb.size(0), txt_emb.size(0))
                img_emb = img_emb[:n_clip]
                txt_emb = txt_emb[:n_clip]
                cos_sim = (img_emb * txt_emb).sum(dim=-1)  # [n_clip]
                clip_norm = (cos_sim + 1.0) / 2.0  # [0,1]

                # ===== Alinear tamaños =====
                a = clip_norm.reshape(-1)
                b = ir_norm.reshape(-1)
                m = min(a.numel(), b.numel(), B_rl)

                if m == 0:
                    reward_mix_raw = torch.zeros(1, device=device)
                else:
                    a = a[:m]
                    b = b[:m]

                    total_w = args.w_clip + args.w_ir
                    if total_w <= 0:
                        total_w = 1.0
                    w_clip = args.w_clip / total_w
                    w_ir = args.w_ir / total_w

                    reward_mix_raw = w_clip * a + w_ir * b  # [m]

                # Para logging
                r_ir_val = float(scores_ir_t.mean().item())
                r_clip_val = float(cos_sim.mean().item())
                r_mix_val = float(reward_mix_raw.mean().item())

            # Advantage con baseline EMA
            if not baseline_init:
                baseline_reward = r_mix_val
                baseline_init = True

            A = reward_mix_raw - baseline_reward  # [m]
            baseline_reward = 0.9 * baseline_reward + 0.1 * r_mix_val

            # Escalar advantage para que el gradiente RL no sea diminuto
            A_scaled = args.reward_scale * A

            # RL BASELINE: -E[advantage * logp]
            m = reward_mix_raw.shape[0]
            logp_new_rl = logp_new[:m]
            loss_rl = - (A_scaled.detach() * logp_new_rl).mean()

        # =========================
        #  Loss total: MSE + RL
        # =========================
        loss = mse + args.lambda_rl * loss_rl

        if not torch.isfinite(loss):
            pbar.set_postfix({
                "loss": "nan/inf",
                "mse": float(mse.detach().cpu().item()),
                "R_mix": r_mix_val,
            })
            optim.zero_grad(set_to_none=True)
            continue

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in pipe.unet.parameters() if p.requires_grad),
            max_norm=1.0
        )
        optim.step()

        last_train_mse = float(mse.detach().cpu().item())
        last_train_rmix = r_mix_val

        pbar.set_postfix({
            "loss": float(loss.detach().cpu().item()),
            "mse": last_train_mse,
            "R_mix": r_mix_val,
        })

        if step % 20 == 0:
            print(
                f"[fold {fold_id+1} | step {step:5d}] "
                f"mse={last_train_mse:.4f} | "
                f"R_ir={r_ir_val:.4f} | "
                f"R_clip={r_clip_val:.4f} | "
                f"R_mix={r_mix_val:.4f}",
                flush=True,
            )

        # VALIDACIÓN PERIÓDICA
        if (step > 0 and step % args.val_interval == 0) or (step == args.steps - 1):
            print("\n Evaluando en validación (fold", fold_id+1, ")...")
            val_ir, val_clip, val_mix = evaluate_on_validation(
                pipe,
                dl_val,
                rm,
                clip_model,
                clip_processor,
                device,
                args.w_clip,
                args.w_ir,
                max_batches=args.max_val_batches,
            )
            last_val_ir = val_ir
            last_val_clip = val_clip
            last_val_mix = val_mix
            print(
                f"    VAL → R_ir={val_ir:.4f} | "
                f"R_clip={val_clip:.4f} | "
                f"R_mix≈{val_mix:.4f}\n",
                flush=True,
            )

    # Guardar LoRA de este fold
    fold_out_dir = Path(args.out_dir) / f"fold{fold_id+1}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    saved = False
    try:
        pipe.save_lora_weights(fold_out_dir)
        print(f" LoRA (fold {fold_id+1}) guardado con pipe.save_lora_weights() en {fold_out_dir}")
        saved = True
    except TypeError:
        pass
    except Exception as e:
        print(f"ℹ save_lora_weights() falló (fold {fold_id+1}): {e}")

    if not saved:
        try:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file

            state = get_peft_model_state_dict(pipe.unet, adapter_name=adapter_name)
            state = {k: v.detach().cpu() for k, v in state.items()}
            out_path = fold_out_dir / "pytorch_lora_weights.safetensors"
            save_file(state, out_path)
            print(f" LoRA (fold {fold_id+1}) guardado en {out_path} (formato safetensors)")
            saved = True
        except Exception as e:
            print(f" No pude guardar LoRA con PEFT (fold {fold_id+1}): {e}")

    if not saved:
        try:
            if hasattr(pipe.unet, "peft_config") and adapter_name in pipe.unet.peft_config and "default" not in pipe.unet.peft_config:
                pipe.unet.peft_config["default"] = pipe.unet.peft_config[adapter_name]
            pipe.unet.save_attn_procs(
                fold_out_dir,
                weight_name="pytorch_lora_weights.safetensors"
            )
            print(f" LoRA (fold {fold_id+1}) guardado con save_attn_procs() en {fold_out_dir}")
            saved = True
        except Exception as e:
            print(f" save_attn_procs() también falló (fold {fold_id+1}): {e}")

    return {
        "train_mse": last_train_mse,
        "train_rmix": last_train_rmix,
        "val_ir": last_val_ir,
        "val_clip": last_val_clip,
        "val_mix": last_val_mix,
    }


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str,
                    default=r"D:/bk2m_rl_starter/bk2m")

    ap.add_argument("--topset", required=True)
    ap.add_argument("--embeds", type=str, default="artifacts/topset/embeds.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=4000)

    #  rank por defecto
    ap.add_argument("--rank", type=int, default=512)

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=12)

    # pesos de la mezcla de reward
    ap.add_argument("--w_clip", type=float, default=0.7)
    ap.add_argument("--w_ir", type=float, default=0.3)

    # hiperparámetros RL (BASELINE)
    ap.add_argument("--lambda_rl", type=float, default=3.0,
                    help="Peso del término RL en la loss total")
    ap.add_argument("--reward_scale", type=float, default=20.0,
                    help="Factor para escalar el advantage (evitar gradientes minúsculos)")
    ap.add_argument("--rl_interval", type=int, default=1,
                    help="Cada cuántos steps aplicar RL (reward + log-probs)")
    ap.add_argument("--rl_batch", type=int, default=12,
                    help="Cuántas imágenes del batch usar para RL (<= batch)")

    # validación
    ap.add_argument("--val_interval", type=int, default=400)
    ap.add_argument("--max_val_batches", type=int, default=3)

    # k-fold
    ap.add_argument("--k_folds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    print("\n==============================")
    print("ENTRENAMIENTO LoRA CON REINFORCE (BK-SDM + CLIP + ImageReward)")
    print("==============================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Device: {device}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    torch.cuda.empty_cache()

    # Dataset completo
    print(" Cargando dataset completo...")
    ds_full = TxtImgDataset(args.topset)
    n_total = len(ds_full)
    print(f"Dataset total: {n_total} muestras.\n")

    # Embeddings y prompts
    print(" Cargando embeddings de texto...")
    text_embeds, prompt_list = load_text_embeds_and_prompts(args.embeds, args.topset)
    print(f"Embeddings cargados: shape={tuple(text_embeds.shape)}, dtype={text_embeds.dtype}\n")

    # Evaluadores compartidos
    print(" Cargando ImageReward (compartido)...")
    rm = load_image_reward(name="ImageReward-v1.0", device=device)
    rm.eval()
    print(" ImageReward cargado.\n")

    print(" Cargando CLIP (openai/clip-vit-base-patch32, compartido)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print(" CLIP cargado.\n")

    # Preparar índices para k-fold
    indices = list(range(n_total))
    random.seed(args.seed)
    random.shuffle(indices)

    k = max(1, args.k_folds)
    fold_sizes = [n_total // k] * k
    for i in range(n_total % k):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(indices[start:end])
        start = end

    all_results = []

    if k == 1:
        n_train = int(0.8 * n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        res = train_one_fold(
            fold_id=0,
            args=args,
            train_indices=train_idx,
            val_indices=val_idx,
            ds_full=ds_full,
            text_embeds=text_embeds,
            prompt_list=prompt_list,
            rm=rm,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
        )
        all_results.append(res)
    else:
        for fold_id in range(k):
            val_idx = folds[fold_id]
            train_idx = []
            for j in range(k):
                if j != fold_id:
                    train_idx.extend(folds[j])

            res = train_one_fold(
                fold_id=fold_id,
                args=args,
                train_indices=train_idx,
                val_indices=val_idx,
                ds_full=ds_full,
                text_embeds=text_embeds,
                prompt_list=prompt_list,
                rm=rm,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
            )
            all_results.append(res)

    if len(all_results) > 1:
        avg = {}
        for key in all_results[0].keys():
            avg[key] = sum(r[key] for r in all_results) / len(all_results)
        print("\n==============================")
        print(" PROMEDIO ENTRE FOLDS")
        print("==============================")
        for kname, v in avg.items():
            print(f"  {kname}: {v:.4f}")
    else:
        print("\nResultados (único split):", all_results[0])


if __name__ == "__main__":
    main()
