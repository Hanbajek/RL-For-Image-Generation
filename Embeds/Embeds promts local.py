# src/rl/embed_prompts_local.py
import argparse, torch
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=r"C:\Users\estudiantes\Desktop\bk2m_rl_starter\bk2m")
    ap.add_argument("--prompts_file", default="artifacts/topset/prompts_modificado.txt")
    ap.add_argument("--out", default="artifacts/topset/embeds3.pt")
    ap.add_argument("--device", default="cpu" \
    "")  # usa "cuda" si quieres+
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    tok = CLIPTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    txt = CLIPTextModel.from_pretrained(model_dir / "text_encoder", local_files_only=True)
    txt.to(args.device)

    prompts = [l.strip() for l in open(args.prompts_file, encoding="utf-8") if l.strip()]

    enc = tok(
        prompts,
        max_length=tok.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        out = txt(enc.input_ids.to(args.device))
        # hidden states que consume el UNet como encoder_hidden_states
        prompt_embeds = out[0].to("cpu")  # guardamos en CPU para compatibilidad

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": prompts, "embeds": prompt_embeds}, args.out)
    print(f"Guardado embeddings en {args.out} | shape={tuple(prompt_embeds.shape)}")

if __name__ == "__main__":
    main()
