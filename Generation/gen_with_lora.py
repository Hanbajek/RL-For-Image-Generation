# src/generator/gen_with_lora.py
import argparse, os, json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

def load_prompts(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No encontré el archivo de prompts: {p}")
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"El archivo de prompts está vacío: {p}")
    return lines

def robust_load_lora(pipe, lora_dir, adapter_name="rl_lora"):
    lora_dir = str(Path(lora_dir).resolve())
    # 1) Intentar API moderna de diffusers
    try:
        pipe.load_lora_weights(lora_dir, adapter_name=adapter_name)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(adapter_name)
        print(f" LoRA cargado con pipe.load_lora_weights('{lora_dir}', adapter='{adapter_name}')")
        return
    except Exception as e1:
        print(f"ℹ load_lora_weights falló: {e1}")

    # 2) Intentar cargar attn_procs directo al UNet (legacy)
    try:
        pipe.unet.load_attn_procs(lora_dir)
        print(f" LoRA cargado con unet.load_attn_procs('{lora_dir}') (modo legacy)")
        return
    except Exception as e2:
        print(f"ℹ unet.load_attn_procs falló: {e2}")

    # 3) Intentar PEFT (muy legacy)
    try:
        from peft import PeftModel
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
        print(f" LoRA envuelto con PEFT desde '{lora_dir}'")
        return
    except Exception as e3:
        print(f" No pude cargar LoRA por ninguna vía. Último error: {e3}")
        raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Carpeta local del modelo Diffusers (con model_index.json)")
    ap.add_argument("--lora", required=True, help="Carpeta con los pesos LoRA (.safetensors)")
    ap.add_argument("--prompts", required=True, help="Archivo de prompts (uno por línea)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Dispositivo: {device}")

    model_dir = str(Path(args.model_dir).resolve())
    if not (Path(model_dir)/"model_index.json").exists():
        raise FileNotFoundError(f"Falta model_index.json en {model_dir} (¿ruta correcta?)")

    # Cargar pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    # Cargar LoRA
    robust_load_lora(pipe, args.lora, adapter_name="rl_lora")

    # Crear out_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar prompts
    prompts = load_prompts(args.prompts)
    print(f" {len(prompts)} prompts cargados de: {args.prompts}")

    # Generador de seeds
    generator = torch.Generator(device=device).manual_seed(args.seed)

    meta = []
    count = 0
    for i, ptxt in enumerate(prompts):
        for k in range(args.num_images):
            # Llamado estándar a diffusers
            with torch.no_grad():
                out = pipe(
                    ptxt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator
                )

            if not hasattr(out, "images") or len(out.images) == 0:
                print(f" No se generó imagen para prompt[{i}] '{ptxt}'. Revisa steps/guidance.")
                continue

            img = out.images[0]
            # Guardar PNG
            fname = f"p{i:03d}_k{k:02d}.png"
            fpath = out_dir / fname
            img.save(fpath)
            count += 1

            meta.append({
                "prompt": ptxt,
                "path": str(fpath),
                "steps": args.steps,
                "guidance": args.guidance,
                "seed": args.seed
            })
            print(f" Guardado: {fpath}")

    # Guardar meta
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f" Listo: {count} imagen(es). Meta en: {meta_path}")

if __name__ == "__main__":
    main()
