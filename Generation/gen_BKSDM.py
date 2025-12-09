from typing import List
import os, argparse, json
from tqdm import tqdm

def generate_images(prompts: List[str], out_dir: str, per_prompt: int = 2, steps: int = 12, guidance: float = 2.5):
    from diffusers import StableDiffusionPipeline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "nota-ai/bk-sdm-small-2m",
        torch_dtype=torch.float16 if device=='cuda' else torch.float32
    ).to(device)
    pipe.enable_attention_slicing()
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for i, p in enumerate(tqdm(prompts, desc="BK-SDM-Small-2M")):
        for j in range(per_prompt):
            img = pipe(p, num_inference_steps=steps, guidance_scale=guidance).images[0]
            fp = os.path.join(out_dir, f"bk2m_{i:04d}_{j:02d}.png")
            img.save(fp)
            meta.append({'prompt': p, 'path': fp})
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_prompt', type=int, default=2)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--guidance', type=float, default=2.5)
    args = ap.parse_args()
    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = [l.strip() for l in f if l.strip()]
    generate_images(prompts, args.out_dir, args.per_prompt, args.steps, args.guidance)

if __name__ == '__main__':
    main()
