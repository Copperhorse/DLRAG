"""QLoRA fine-tuning of MiniCPM5-2B on the book + repository data (run on Colab, NOT on the dev machine).

    python train_qlora.py --data train.jsonl --chat-template chat_template.jinja

Pipeline: 4-bit NF4 base -> LoRA adapters -> TRL SFTTrainer -> save adapter -> merge -> GGUF (Q8_0).
Uses Unsloth when available (faster, less VRAM); ``--no-unsloth`` uses plain transformers + peft +
bitsandbytes. The chat template is the one extracted from the Sharp GGUF so training matches inference.

Fairness notes (see README.md): the base must be the HF weights the GGUF arms were made from
(`openbmb/MiniCPM5-2B`); the thinking mode is OFF in training formatting, as in the evaluation runs.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--data", type=Path, default=Path("train.jsonl"))
    ap.add_argument("--chat-template", type=Path, default=Path("chat_template.jinja"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--quant", default="q8_0", help="GGUF quantization of the merged model")
    ap.add_argument("--no-unsloth", action="store_true", help="plain transformers + peft + bitsandbytes")
    ap.add_argument("--max-steps", type=int, default=-1, help="debug: stop after N steps")
    ap.add_argument("--skip-gguf", action="store_true")
    return ap.parse_args()


def load_model_unsloth(args):
    from unsloth import FastLanguageModel  # must be imported before transformers/trl/peft

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model, max_seq_length=args.max_seq_len, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=TARGET_MODULES, use_gradient_checkpointing="unsloth", random_state=args.seed)
    return model, tokenizer


def load_model_plain(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    return model, tokenizer


def build_dataset(args, tokenizer):
    from datasets import Dataset

    rows = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]

    def to_text(r):
        if r["kind"] == "doc":
            return r["text"] + (tokenizer.eos_token or "")
        # qa -> the Sharp chat template with thinking off, exactly as at inference time
        return tokenizer.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False)

    texts = [to_text(r) for r in rows]
    print(f"{len(texts)} training examples; first QA example as the model will see it:\n"
          + next(t for r, t in zip(rows, texts) if r["kind"] == "qa")[:800])
    return Dataset.from_dict({"text": texts})


def main() -> None:
    args = parse_args()
    model, tokenizer = (load_model_plain if args.no_unsloth else load_model_unsloth)(args)
    tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch
    from trl import SFTConfig, SFTTrainer

    dataset = build_dataset(args, tokenizer)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg_kwargs = dict(
        output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
        optim="adamw_8bit", bf16=bf16, fp16=not bf16, logging_steps=10, save_strategy="no",
        seed=args.seed, report_to="none", dataset_text_field="text", packing=False)
    # TRL renamed max_seq_length -> max_length in newer releases.
    params = inspect.signature(SFTConfig.__init__).parameters
    cfg_kwargs["max_length" if "max_length" in params else "max_seq_length"] = args.max_seq_len
    trainer_kwargs = dict(model=model, train_dataset=dataset, args=SFTConfig(**cfg_kwargs))
    key = "processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else "tokenizer"
    trainer_kwargs[key] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to {adapter_dir}")
    if args.skip_gguf:
        return

    if args.no_unsloth:
        merged_dir = args.out / "merged-16bit"
        from peft import AutoPeftModelForCausalLM
        merged = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir), torch_dtype=torch.float16, trust_remote_code=True)
        merged.merge_and_unload().save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Merged fp16 model in {merged_dir}. Convert with llama.cpp:\n"
              f"  python convert_hf_to_gguf.py {merged_dir} --outtype {args.quant} --outfile {args.out}/model-{args.quant}.gguf")
    else:
        model.save_pretrained_gguf(str(args.out / "gguf"), tokenizer, quantization_method=args.quant)
        print(f"GGUF ({args.quant}) written under {args.out / 'gguf'}")


if __name__ == "__main__":
    main()
