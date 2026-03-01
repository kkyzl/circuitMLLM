#!/usr/bin/env python3
"""
AnalogSeeker NSC-SFT Training Pipeline
=======================================
Fine-tunes Qwen3-VL-8B-Instruct on text-only SFT data with:
  - Frozen vision encoder (model.visual)
  - LoRA on language model
  - NSC-SFT objective: CE + lambda_kl * KL(p_policy || p_ref)
  - Reference distribution via LoRA adapter toggling (eval + no_grad)

Paper: "AnalogSeeker" (arXiv 2508.10409)
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="AnalogSeeker NSC-SFT Training")

    # Model / data
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--data_path", type=str, default="./analog_data/sft.jsonl")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--attn_implementation", type=str, default="eager",
                    choices=["eager", "sdpa"],
                    help="Attention implementation (flash disabled for KL stability)")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"])

    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=1,
                    help="Total target epochs (default 1 for staged execution)")
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_kl", type=float, default=0.1,
                    help="Weight for KL divergence in NSC-SFT objective")

    # Split / Eval
    p.add_argument("--val_split_ratio", type=float, default=0.05)
    p.add_argument("--evals_per_epoch", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=None,
                    help="Explicit eval step count (overrides evals_per_epoch)")
    p.add_argument("--format_check_samples", type=int, default=64)

    # Early stopping (optional, off by default)
    p.add_argument("--early_stop_epsilon", type=float, default=0.005)
    p.add_argument("--early_stop_patience", type=int, default=0,
                    help="0 = disabled (default); set to 2+ for unattended mode")
    p.add_argument("--early_stop_metric", type=str, default="total",
                    choices=["total", "ce"])
    p.add_argument("--min_epochs", type=int, default=1)

    # Resume
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to checkpoint dir (e.g., output/checkpoints/epoch-1)")

    # Pilot run
    p.add_argument("--pilot_steps", type=int, default=0,
                    help="If >0, run N steps then exit with timing report")

    # External eval
    p.add_argument("--run_external_eval", action="store_true", default=False)
    p.add_argument("--eval_at_checkpoints", action="store_true", default=False)

    # WandB
    p.add_argument("--report_to", type=str, default="wandb",
                    choices=["wandb", "none"])
    p.add_argument("--wandb_project", type=str, default="analogseeker_sft")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default=None,
                    help="Path to DeepSpeed config JSON (e.g., ds_zero3_config.json). "
                         "Enables DeepSpeed ZeRO training.")
    p.add_argument("--local_rank", type=int, default=-1,
                    help="Local rank for distributed training (set by DeepSpeed launcher)")
    p.add_argument("--fp16", action="store_true", default=False,
                    help="Use FP16 instead of BF16 (required for RTX 3090)")
    p.add_argument("--kl_chunk_size", type=int, default=512,
                    help="Chunk size along sequence dim for KL computation "
                         "(reduces peak memory; 0=no chunking)")

    # System
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are an expert in analog circuit design."


class SFTDataset(Dataset):
    """Loads sft.jsonl and tokenizes with ChatML template."""

    def __init__(self, samples: list, processor, max_seq_length: int):
        self.samples = samples
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        response = sample["response"]

        # Build ChatML conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Apply chat template — full conversation (no generation prompt)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Build labels: mask everything except assistant content
        # Tokenize system + user portion to find where assistant starts
        messages_no_assistant = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        prefix_text = self.tokenizer.apply_chat_template(
            messages_no_assistant, tokenize=False, add_generation_prompt=True
        )
        prefix_enc = self.tokenizer(
            prefix_text, truncation=False, padding=False, return_tensors=None
        )
        prefix_len = len(prefix_enc["input_ids"])

        # Labels: -100 for prefix, token_ids for assistant content
        labels = [-100] * prefix_len + input_ids[prefix_len:]
        # Ensure same length after truncation
        labels = labels[: len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_jsonl(path: str) -> list:
    """Load all samples from a JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at line {line_num}")
                continue
            if "prompt" not in obj or "response" not in obj:
                logger.warning(f"Missing prompt/response at line {line_num}")
                continue
            samples.append(obj)
    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples


def make_collate_fn(pad_token_id: int):
    """Create a collate function with the correct pad token id."""

    def collate_fn(batch):
        """Right-pad batch to max length in batch."""
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            # Right-pad for causal LM
            input_ids_list.append(
                F.pad(item["input_ids"], (0, pad_len), value=pad_token_id)
            )
            attention_mask_list.append(
                F.pad(item["attention_mask"], (0, pad_len), value=0)
            )
            labels_list.append(
                F.pad(item["labels"], (0, pad_len), value=-100)
            )

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Train / Val Split
# ---------------------------------------------------------------------------
def get_or_create_split(all_samples, output_dir, val_split_ratio, seed):
    """Deterministic 95/5 split. Reuse saved indices if they exist."""
    splits_dir = Path(output_dir) / "splits"
    train_idx_path = splits_dir / "train_indices.json"
    val_idx_path = splits_dir / "val_indices.json"
    split_config_path = splits_dir / "split_config.json"

    n = len(all_samples)

    if train_idx_path.exists() and val_idx_path.exists():
        logger.info(f"Loading existing split from {splits_dir}")
        with open(train_idx_path) as f:
            train_indices = json.load(f)
        with open(val_idx_path) as f:
            val_indices = json.load(f)
        logger.info(f"Loaded split: {len(train_indices)} train, {len(val_indices)} val")
        return train_indices, val_indices

    logger.info(f"Creating new {1 - val_split_ratio:.0%}/{val_split_ratio:.0%} split with seed={seed}")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    split_point = int(n * (1 - val_split_ratio))
    train_indices = sorted(indices[:split_point].tolist())
    val_indices = sorted(indices[split_point:].tolist())

    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(train_idx_path, "w") as f:
        json.dump(train_indices, f)
    with open(val_idx_path, "w") as f:
        json.dump(val_indices, f)
    with open(split_config_path, "w") as f:
        json.dump({
            "seed": seed,
            "train_ratio": 1 - val_split_ratio,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
        }, f, indent=2)

    logger.info(f"Split saved: {len(train_indices)} train, {len(val_indices)} val")
    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def load_model_and_processor(args):
    """Load Qwen3-VL with frozen vision encoder and LoRA on LLM."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    logger.info(f"Loading model: {args.model_name}")
    logger.info(f"Attention implementation: {args.attn_implementation}")

    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    use_deepspeed = getattr(args, "deepspeed", None) is not None

    if use_deepspeed:
        # ZeRO-3: load to CPU; deepspeed.initialize() handles sharding later
        logger.info("Loading model to CPU for DeepSpeed ZeRO-3 sharding")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map=None,
            low_cpu_mem_usage=True,
        )
    else:
        # Legacy mode: device_map="auto"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(args.model_name)

    # Freeze vision encoder
    frozen_count = 0
    for name, param in model.named_parameters():
        if name.startswith("model.visual") or name.startswith("visual"):
            param.requires_grad = False
            frozen_count += 1
    logger.info(f"Frozen {frozen_count} vision encoder parameters")

    # Verify all vision params are frozen
    for name, param in model.named_parameters():
        if "visual" in name:
            assert not param.requires_grad, f"Vision param {name} not frozen!"

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Log parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    return model, processor


# ---------------------------------------------------------------------------
# DeepSpeed helper
# ---------------------------------------------------------------------------
def get_base_model(model):
    """Unwrap DeepSpeed engine to access the PeftModel underneath.

    Chain: DeepSpeedEngine -> PeftModel -> Qwen3VLForConditionalGeneration
    Without DeepSpeed, model IS the PeftModel directly.
    """
    if hasattr(model, "module"):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Reference forward context manager
# ---------------------------------------------------------------------------
@contextmanager
def reference_mode(model, use_deepspeed=False):
    """Context manager for reference forward: eval() + disable LoRA adapters + no_grad."""
    base = get_base_model(model) if use_deepspeed else model
    was_training = base.training
    base.eval()
    try:
        with base.disable_adapter():
            with torch.no_grad():
                yield
    finally:
        if was_training:
            base.train()


# ---------------------------------------------------------------------------
# NSC-SFT Loss
# ---------------------------------------------------------------------------
def _compute_kl_chunked(shift_logits, shift_ref_logits, mask, mask_sum,
                         chunk_size):
    """Compute KL(p_policy || p_ref) in chunks along the sequence dimension.

    At seq=8192, vocab=152064, FP16:
    - Full tensor: [1, 8191, 152064] = 2.49 GB
    - Chunk of 512: [1, 512, 152064] = 0.16 GB
    - Peak with intermediates (log_p, log_q, p, diff): ~0.6 GB per chunk

    This reduces peak memory from ~15-20 GB to ~1-2 GB.
    """
    seq_len = shift_logits.size(1)
    kl_sum = torch.tensor(0.0, device=shift_logits.device, dtype=torch.float32)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        chunk_mask = mask[:, start:end]
        if chunk_mask.sum() == 0:
            continue

        chunk_logits = shift_logits[:, start:end, :]
        chunk_ref_logits = shift_ref_logits[:, start:end, :]

        log_p = F.log_softmax(chunk_logits, dim=-1)
        log_q = F.log_softmax(chunk_ref_logits, dim=-1)
        p = log_p.exp()
        kl_per_token = (p * (log_p - log_q)).sum(dim=-1)
        kl_sum = kl_sum + (kl_per_token * chunk_mask).sum().float()

        del log_p, log_q, p, kl_per_token

    return kl_sum / mask_sum.float()


def compute_nsc_sft_loss(policy_logits, ref_logits, labels, lambda_kl,
                         kl_chunk_size=0):
    """
    NSC-SFT: L = CE + lambda_kl * KL(p_policy || p_ref)

    Both losses are computed only on assistant tokens (labels != -100).
    KL is full-vocabulary: KL(p||q) = sum_v p(v) * (log p(v) - log q(v))

    When kl_chunk_size > 0, KL is computed in chunks along the sequence
    dimension to reduce peak memory from O(seq * vocab) to O(chunk * vocab).
    CE remains unchunked because F.cross_entropy is already memory-efficient.
    """
    # Shift for causal LM: predict next token
    shift_logits = policy_logits[..., :-1, :].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # CE loss (masked via ignore_index=-100)
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    # KL loss on assistant tokens only
    mask = (shift_labels != -100).float()  # [batch, seq_len]
    mask_sum = mask.sum()

    if mask_sum > 0 and lambda_kl > 0:
        if kl_chunk_size > 0:
            kl_loss = _compute_kl_chunked(
                shift_logits, shift_ref_logits, mask, mask_sum, kl_chunk_size
            )
        else:
            # Original unchunked path (for small sequences / backward compat)
            log_p = F.log_softmax(shift_logits, dim=-1)
            log_q = F.log_softmax(shift_ref_logits, dim=-1)
            p = log_p.exp()
            kl_per_token = (p * (log_p - log_q)).sum(dim=-1)
            kl_loss = (kl_per_token * mask).sum() / mask_sum
    else:
        kl_loss = torch.tensor(0.0, device=policy_logits.device)

    total_loss = ce_loss + lambda_kl * kl_loss

    return total_loss, ce_loss, kl_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_dataloader, lambda_kl, device, use_deepspeed=False,
             kl_chunk_size=0):
    """Run validation and return average CE, KL, total losses."""
    base = get_base_model(model) if use_deepspeed else model
    base.eval()

    total_ce = 0.0
    total_kl = 0.0
    total_loss_sum = 0.0
    n_batches = 0

    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Assert no image data in batch
        assert "pixel_values" not in batch, "Batch contains pixel_values — text-only training expected"
        assert "image_grid_thw" not in batch, "Batch contains image_grid_thw — text-only training expected"

        # Reference forward (LoRA disabled, eval mode)
        with base.disable_adapter():
            ref_out = model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_out.logits

        # Policy forward (LoRA enabled, still in eval mode for val)
        policy_out = model(input_ids=input_ids, attention_mask=attention_mask)
        policy_logits = policy_out.logits

        loss, ce, kl = compute_nsc_sft_loss(
            policy_logits, ref_logits, labels, lambda_kl,
            kl_chunk_size=kl_chunk_size,
        )

        total_ce += ce.item()
        total_kl += kl.item()
        total_loss_sum += loss.item()
        n_batches += 1

    base.train()

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    # In distributed mode, average metrics across all ranks
    if use_deepspeed and torch.distributed.is_initialized():
        metrics = torch.tensor(
            [total_ce, total_kl, total_loss_sum, float(n_batches)],
            device=device,
        )
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        total_ce, total_kl, total_loss_sum, n_batches = metrics.tolist()

    return (
        total_ce / n_batches,
        total_kl / n_batches,
        total_loss_sum / n_batches,
    )


def format_compliance_check(model, processor, val_samples, n_samples, device,
                            max_new_tokens=2048, use_deepspeed=False):
    """Check what fraction of generated outputs contain <answer>...</answer>."""
    base = get_base_model(model) if use_deepspeed else model
    base.eval()
    tokenizer = processor.tokenizer

    subset = val_samples[:n_samples]
    has_answer = 0

    for sample in subset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            # All ranks must call generate() for ZeRO-3 parameter gathering
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
        if re.search(r"<answer>.*?</answer>", generated, re.DOTALL):
            has_answer += 1

    rate = has_answer / len(subset) if subset else 0.0
    base.train()
    return rate


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                    best_val_loss, seed, wandb_run_id, save_dir,
                    use_deepspeed=False):
    """Save full checkpoint for resumable training."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if use_deepspeed:
        # LoRA adapter weights (portable, for inference)
        # stage3_gather_16bit_weights_on_model_save=true gathers sharded params
        base_model = get_base_model(model)
        base_model.save_pretrained(str(save_dir))

        # DeepSpeed engine state (sharded optimizer, scheduler, etc.)
        ds_ckpt_dir = save_dir / "deepspeed_state"
        model.save_checkpoint(str(ds_ckpt_dir), tag="latest")
    else:
        # Legacy mode
        model.save_pretrained(str(save_dir))
        torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")

    # RNG states
    rng_states = {
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    torch.save(rng_states, save_dir / "rng_states.pt")

    # Training state metadata
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "seed": seed,
        "wandb_run_id": wandb_run_id,
        "use_deepspeed": use_deepspeed,
    }
    with open(save_dir / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)

    logger.info(f"Checkpoint saved to {save_dir}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, device,
                    use_deepspeed=False):
    """Load checkpoint and restore all state."""
    checkpoint_dir = Path(checkpoint_dir)
    logger.info(f"Resuming from checkpoint: {checkpoint_dir}")

    if use_deepspeed:
        # Try DeepSpeed engine checkpoint first
        ds_ckpt_dir = checkpoint_dir / "deepspeed_state"
        if ds_ckpt_dir.exists():
            _, client_state = model.load_checkpoint(
                str(ds_ckpt_dir), tag="latest"
            )
            logger.info("Loaded DeepSpeed engine checkpoint (optimizer + scheduler + params)")
        else:
            # Fallback: load LoRA adapter weights only (e.g., from legacy checkpoint)
            from safetensors.torch import load_file
            adapter_path = checkpoint_dir / "adapter_model.safetensors"
            if adapter_path.exists():
                adapter_state = load_file(str(adapter_path))
                base_model = get_base_model(model)
                incompatible = base_model.load_state_dict(adapter_state, strict=False)
                logger.info(f"Loaded LoRA adapter weights only ({len(adapter_state)} tensors)")
                if incompatible.unexpected_keys:
                    logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
            else:
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")
    else:
        # Legacy mode
        from safetensors.torch import load_file
        adapter_path = checkpoint_dir / "adapter_model.safetensors"
        if adapter_path.exists():
            adapter_state = load_file(str(adapter_path))
            incompatible = model.load_state_dict(adapter_state, strict=False)
            logger.info(f"Loaded LoRA adapter weights ({len(adapter_state)} tensors)")
            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
        else:
            raise FileNotFoundError(f"Adapter weights not found at {adapter_path}")

        # Optimizer
        opt_path = checkpoint_dir / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))
            logger.info("Loaded optimizer state")

        # Scheduler
        sched_path = checkpoint_dir / "scheduler.pt"
        if scheduler is not None and sched_path.exists():
            scheduler.load_state_dict(torch.load(sched_path, map_location=device, weights_only=True))
            logger.info("Loaded scheduler state")
        elif scheduler is None:
            logger.info("Scheduler will be re-initialized after resume")

    # RNG states (same for both modes)
    rng_path = checkpoint_dir / "rng_states.pt"
    if rng_path.exists():
        rng_states = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.random.set_rng_state(rng_states["torch"])
        if rng_states["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_states["cuda"])
        np.random.set_state(rng_states["numpy"])
        random.setstate(rng_states["python"])
        logger.info("Restored RNG states")

    # Training state
    state_path = checkpoint_dir / "training_state.json"
    with open(state_path) as f:
        training_state = json.load(f)

    logger.info(f"Resumed at epoch={training_state['epoch']}, "
                f"global_step={training_state['global_step']}, "
                f"best_val_loss={training_state['best_val_loss']}")

    return training_state


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopper:
    """Tracks val loss and determines when to stop."""

    def __init__(self, patience: int, epsilon: float, min_epochs: int, metric: str):
        self.patience = patience
        self.epsilon = epsilon
        self.min_epochs = min_epochs
        self.metric = metric  # "total" or "ce"
        self.best_loss = float("inf")
        self.counter = 0
        self.enabled = patience > 0

    def check(self, val_ce, val_total, current_epoch) -> bool:
        """Returns True if training should stop."""
        if not self.enabled:
            return False
        if current_epoch < self.min_epochs:
            return False

        val_loss = val_total if self.metric == "total" else val_ce

        # Check relative improvement
        if self.best_loss == float("inf"):
            self.best_loss = val_loss
            return False

        relative_improvement = (self.best_loss - val_loss) / abs(self.best_loss) if self.best_loss != 0 else 0

        if relative_improvement >= self.epsilon:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"Early stopping: no improvement ({relative_improvement:.4f} < {self.epsilon}), "
                        f"patience {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.counter} evals without improvement")
            return True

        return False


# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------
def init_wandb(args, wandb_run_id=None):
    """Initialize WandB. Returns run_id or None."""
    if args.report_to != "wandb":
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, disabling wandb logging")
        return None

    init_kwargs = {
        "project": args.wandb_project,
        "config": vars(args),
    }

    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name

    if wandb_run_id:
        init_kwargs["id"] = wandb_run_id
        init_kwargs["resume"] = "must"

    run = wandb.init(**init_kwargs)
    logger.info(f"WandB initialized: run_id={run.id}")
    return run.id


def log_wandb(metrics: dict):
    """Log metrics to WandB if available."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- FP16/BF16 mutual exclusivity ---
    if args.fp16 and args.bf16:
        args.bf16 = False
        logger.warning("Both --fp16 and --bf16 set; disabling bf16 in favor of fp16")

    # --- Distributed / DeepSpeed initialization ---
    use_deepspeed = args.deepspeed is not None

    if use_deepspeed:
        import deepspeed
        deepspeed.init_distributed()
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        is_main_process = (args.local_rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        args.local_rank = 0

    # Restrict logging to main process
    if not is_main_process:
        logger.setLevel(logging.WARNING)

    set_seed(args.seed)

    # Create output dir and save config (main process only)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save training config
        config_path = Path(args.output_dir) / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Config saved to {config_path}")

        # Save git hash if available
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            with open(Path(args.output_dir) / "git_hash.txt", "w") as f:
                f.write(git_hash)
            logger.info(f"Git hash: {git_hash}")
        except Exception:
            pass

    # Barrier: ensure output dir exists before other ranks proceed
    if use_deepspeed:
        torch.distributed.barrier()

    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(device.index or 0).total_mem / 1e9:.1f} GB")

    # Load data
    all_samples = load_jsonl(args.data_path)

    # Train/val split — main process creates split files, others wait
    if is_main_process:
        train_indices, val_indices = get_or_create_split(
            all_samples, args.output_dir, args.val_split_ratio, args.seed
        )
    if use_deepspeed:
        torch.distributed.barrier()
    if not is_main_process:
        train_indices, val_indices = get_or_create_split(
            all_samples, args.output_dir, args.val_split_ratio, args.seed
        )
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    # Load model
    model, processor = load_model_and_processor(args)

    # Create datasets
    train_dataset = SFTDataset(train_samples, processor, args.max_seq_length)
    val_dataset = SFTDataset(val_samples, processor, args.max_seq_length)

    # Collate with correct pad token
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
        logger.warning(f"pad_token_id is None, using eos_token_id={pad_token_id}")
    collate_fn = make_collate_fn(pad_token_id)

    # Distributed samplers for DeepSpeed multi-GPU
    if use_deepspeed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
            seed=args.seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Steps computation
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    # Resume state
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    wandb_run_id = None

    if use_deepspeed:
        # --- DeepSpeed engine initialization ---
        # DeepSpeed manages optimizer, scheduler, gradient accumulation, and clipping.
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=args.deepspeed,
        )
        model = model_engine
        logger.info(f"DeepSpeed engine initialized (ZeRO stage "
                     f"{model_engine.zero_optimization_stage()})")

        # Resume from checkpoint
        if args.resume_from_checkpoint:
            training_state = load_checkpoint(
                model, optimizer, scheduler, args.resume_from_checkpoint, device,
                use_deepspeed=True,
            )
            start_epoch = training_state["epoch"]
            global_step = training_state["global_step"]
            best_val_loss = training_state["best_val_loss"]
            wandb_run_id = training_state.get("wandb_run_id")

            if start_epoch >= args.num_train_epochs:
                logger.info(f"Already completed {start_epoch} epochs, target is "
                            f"{args.num_train_epochs}. Nothing to do. Exiting.")
                return

    else:
        # --- Legacy mode: manual optimizer + scheduler ---
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Resume from checkpoint (before scheduler creation)
        if args.resume_from_checkpoint:
            training_state = load_checkpoint(
                model, optimizer, None, args.resume_from_checkpoint, device,
                use_deepspeed=False,
            )
            start_epoch = training_state["epoch"]
            global_step = training_state["global_step"]
            best_val_loss = training_state["best_val_loss"]
            wandb_run_id = training_state.get("wandb_run_id")

            if start_epoch >= args.num_train_epochs:
                logger.info(f"Already completed {start_epoch} epochs, target is "
                            f"{args.num_train_epochs}. Nothing to do. Exiting.")
                return

        # Scheduler — created after resume so total_steps reflects actual range
        remaining_epochs = args.num_train_epochs - start_epoch
        total_steps = steps_per_epoch * remaining_epochs

        if args.resume_from_checkpoint:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps, 1), eta_min=0
            )
            logger.info(f"Scheduler initialized for {remaining_epochs} remaining epochs "
                         f"({total_steps} steps). Cosine shape may differ from single run.")
        else:
            warmup_steps = int(total_steps * args.warmup_ratio)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8 / args.learning_rate if args.learning_rate > 0 else 1e-8,
                total_iters=warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=0
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

    # Init WandB (main process only)
    if is_main_process:
        wandb_run_id = init_wandb(args, wandb_run_id=wandb_run_id)

    # Eval schedule
    if args.eval_steps is not None:
        eval_every_steps = args.eval_steps
    else:
        eval_every_steps = max(steps_per_epoch // args.evals_per_epoch, 1)
    logger.info(f"Eval every {eval_every_steps} optimizer steps")

    # Early stopping
    early_stopper = EarlyStopper(
        patience=args.early_stop_patience,
        epsilon=args.early_stop_epsilon,
        min_epochs=args.min_epochs,
        metric=args.early_stop_metric,
    )

    # Log setup summary
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Attention: {args.attn_implementation}")
    logger.info(f"  Precision: {'FP16' if args.fp16 else 'BF16' if args.bf16 else 'FP32'}")
    logger.info(f"  DeepSpeed: {'ZeRO-3' if use_deepspeed else 'disabled'}")
    logger.info(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    logger.info(f"  Lambda KL: {args.lambda_kl}")
    logger.info(f"  KL chunk size: {args.kl_chunk_size}")
    logger.info(f"  Train samples: {len(train_samples)}")
    logger.info(f"  Val samples: {len(val_samples)}")
    logger.info(f"  Batch size: {args.per_device_train_batch_size}")
    logger.info(f"  Grad accum: {args.gradient_accumulation_steps}")
    if use_deepspeed:
        world_size = torch.distributed.get_world_size()
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size}")
    else:
        logger.info(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Epochs: {start_epoch} -> {args.num_train_epochs}")
    logger.info(f"  Early stopping: {'enabled (patience=' + str(args.early_stop_patience) + ')' if args.early_stop_patience > 0 else 'disabled (staged mode)'}")
    logger.info(f"  Max seq length: {args.max_seq_length}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    base_model = get_base_model(model)
    base_model.train()
    if not use_deepspeed:
        optimizer.zero_grad()

    micro_step = 0  # counts individual forward/backward passes
    tokens_processed = 0
    training_start_time = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        logger.info(f"--- Epoch {epoch + 1}/{args.num_train_epochs} ---")
        epoch_start_time = time.time()
        tokens_processed = 0  # reset per epoch for accurate throughput

        if use_deepspeed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Assert no image data
            assert "pixel_values" not in batch, "Batch contains pixel_values — text-only training expected"
            assert "image_grid_thw" not in batch, "Batch contains image_grid_thw — text-only training expected"

            # Reference forward: eval() + LoRA disabled + no_grad
            with reference_mode(model, use_deepspeed=use_deepspeed):
                ref_out = model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_out.logits.detach()

            # Policy forward: train() + LoRA enabled
            base_model.train()
            policy_out = model(input_ids=input_ids, attention_mask=attention_mask)
            policy_logits = policy_out.logits

            # NSC-SFT loss
            total_loss, ce_loss, kl_loss = compute_nsc_sft_loss(
                policy_logits, ref_logits, labels, args.lambda_kl,
                kl_chunk_size=args.kl_chunk_size,
            )

            # Backward + step
            if use_deepspeed:
                # DeepSpeed handles loss scaling, gradient accumulation, clipping
                model.backward(total_loss)
                model.step()
            else:
                scaled_loss = total_loss / args.gradient_accumulation_steps
                scaled_loss.backward()

            # Track tokens
            n_tokens = (attention_mask.sum()).item()
            tokens_processed += n_tokens
            micro_step += 1

            # Detect optimizer step boundary
            if use_deepspeed:
                is_optimizer_step = model.is_gradient_accumulation_boundary()
            else:
                is_optimizer_step = (micro_step % args.gradient_accumulation_steps == 0)

            if is_optimizer_step:
                if not use_deepspeed:
                    # Legacy: manual gradient clipping + optimizer step
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                else:
                    # DeepSpeed: get grad norm from engine
                    grad_norm = model.get_global_grad_norm()
                    if grad_norm is None:
                        grad_norm = 0.0

                global_step += 1

                # Compute elapsed time and throughput
                elapsed = time.time() - epoch_start_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

                # Current LR
                if use_deepspeed:
                    lrs = model.get_lr()
                    current_lr = lrs[0] if lrs else 0.0
                else:
                    current_lr = scheduler.get_last_lr()[0]

                # Fractional epoch
                frac_epoch = epoch + (batch_idx + 1) / len(train_dataloader)

                # Log
                if global_step % 10 == 0 or global_step == 1:
                    logger.info(
                        f"Step {global_step} | epoch {frac_epoch:.2f} | "
                        f"loss={total_loss.item():.4f} ce={ce_loss.item():.4f} "
                        f"kl={kl_loss.item():.4f} | lr={current_lr:.2e} | "
                        f"grad_norm={grad_norm:.3f} | tok/s={tokens_per_sec:.0f}"
                    )

                # WandB logging (main process only)
                if is_main_process:
                    log_wandb({
                        "train/loss": total_loss.item(),
                        "train/ce_loss": ce_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": frac_epoch,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/global_step": global_step,
                    })

                # Pilot run: exit after N steps
                if args.pilot_steps > 0 and global_step >= args.pilot_steps:
                    pilot_metrics = {
                        "steps": global_step,
                        "elapsed_sec": elapsed,
                        "tokens_processed": tokens_processed,
                        "tokens_per_sec": tokens_per_sec,
                        "estimated_time_per_epoch_hr": (steps_per_epoch / global_step) * elapsed / 3600,
                    }
                    if is_main_process:
                        pilot_path = Path(args.output_dir) / "pilot_metrics.json"
                        with open(pilot_path, "w") as f:
                            json.dump(pilot_metrics, f, indent=2)
                    logger.info(f"Pilot run complete after {global_step} steps.")
                    logger.info(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
                    logger.info(f"  Est. time/epoch: {pilot_metrics['estimated_time_per_epoch_hr']:.1f} hr")
                    return

                # Eval at boundary
                if global_step % eval_every_steps == 0:
                    logger.info(f"--- Eval at step {global_step} (epoch {frac_epoch:.2f}) ---")

                    val_ce, val_kl, val_total = evaluate(
                        model, val_dataloader, args.lambda_kl, device,
                        use_deepspeed=use_deepspeed,
                        kl_chunk_size=args.kl_chunk_size,
                    )
                    logger.info(f"Val loss: total={val_total:.4f} ce={val_ce:.4f} kl={val_kl:.4f}")

                    if is_main_process:
                        log_wandb({
                            "val/ce_loss": val_ce,
                            "val/kl_loss": val_kl,
                            "val/total_loss": val_total,
                            "val/step": global_step,
                        })

                    # Format compliance at epoch boundaries
                    is_epoch_boundary = abs(frac_epoch - round(frac_epoch)) < 0.01
                    if is_epoch_boundary and args.format_check_samples > 0:
                        format_rate = format_compliance_check(
                            model, processor, val_samples,
                            args.format_check_samples, device,
                            use_deepspeed=use_deepspeed,
                        )
                        if is_main_process:
                            logger.info(f"Format compliance (answer tag rate): {format_rate:.2%}")
                            log_wandb({"val/answer_format_rate": format_rate})
                        base_model.train()

                    # Determine completed epoch for checkpoint metadata.
                    completed_epoch = epoch  # 0-indexed epoch currently in progress

                    # Checkpoint: best
                    if val_total < best_val_loss:
                        best_val_loss = val_total
                        save_checkpoint(
                            model, optimizer, scheduler, completed_epoch, global_step,
                            best_val_loss, args.seed, wandb_run_id,
                            Path(args.output_dir) / "checkpoints" / "best",
                            use_deepspeed=use_deepspeed,
                        )

                    # Checkpoint: last
                    save_checkpoint(
                        model, optimizer, scheduler, completed_epoch, global_step,
                        best_val_loss, args.seed, wandb_run_id,
                        Path(args.output_dir) / "checkpoints" / "last",
                        use_deepspeed=use_deepspeed,
                    )

                    # Epoch-boundary checkpoint (designed for resume)
                    if is_epoch_boundary:
                        epoch_num = round(frac_epoch)
                        save_checkpoint(
                            model, optimizer, scheduler, epoch_num, global_step,
                            best_val_loss, args.seed, wandb_run_id,
                            Path(args.output_dir) / "checkpoints" / f"epoch-{epoch_num}",
                            use_deepspeed=use_deepspeed,
                        )

                    # Early stopping check
                    if early_stopper.check(val_ce, val_total, frac_epoch):
                        logger.info("Early stopping triggered. Loading best checkpoint.")
                        break

                    base_model.train()

        else:
            # Loop completed without break (no early stopping)
            continue
        # Early stopping break propagated from inner loop
        break

    # Final log
    total_time = time.time() - training_start_time
    logger.info(f"Training complete. Total steps: {global_step}, total time: {total_time/3600:.1f}h")
    logger.info(f"Best val loss: {best_val_loss:.4f}")

    # External eval (stub) — main process only
    if is_main_process and args.run_external_eval:
        logger.warning("External eval requested but no eval scripts found. Skipping.")
        eval_results_path = Path(args.output_dir) / "external_eval_results.json"
        with open(eval_results_path, "w") as f:
            json.dump({"status": "no_eval_scripts_found"}, f, indent=2)

    # Finish WandB
    if is_main_process:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

    logger.info("Done.")


if __name__ == "__main__":
    main()
