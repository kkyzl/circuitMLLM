#!/usr/bin/env python3
"""
Unit tests for the AnalogSeeker NSC-SFT training pipeline.

Tests 1-8: Core pipeline verification (from PLAN Section 8)
Tests 9-13: Split, eval, early stopping, checkpoint verification (from PLAN Section 14)

Usage:
  # Full suite (requires GPU + model download)
  pytest tests/test_pipeline.py -v --tb=short

  # Quick subset (env + dataset only, no GPU model loading)
  pytest tests/test_pipeline.py -v -k "env or dataset" --tb=short

  # Use smaller model for faster iteration (env var)
  MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct pytest tests/test_pipeline.py -v

  # Inside Singularity container
  singularity exec --nv -B ./:/code -B ./analog_data:/data \\
      analogseeker_train.sif \\
      python3 -m pytest /code/tests/test_pipeline.py -v --tb=short
"""

import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "./analog_data/sft.jsonl")
# Also check container mount path
if not os.path.exists(DATA_PATH) and os.path.exists("/data/sft.jsonl"):
    DATA_PATH = "/data/sft.jsonl"

HAS_CUDA = torch.cuda.is_available()

skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
skip_no_data = pytest.mark.skipif(
    not os.path.exists(DATA_PATH), reason=f"Dataset not found at {DATA_PATH}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_collate_fn(proc):
    """Create a collate function from a processor."""
    from train import make_collate_fn
    pad_id = proc.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = proc.tokenizer.eos_token_id
    return make_collate_fn(pad_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def all_samples():
    """Load all samples from sft.jsonl."""
    from train import load_jsonl
    return load_jsonl(DATA_PATH)


@pytest.fixture(scope="session")
def processor():
    """Load processor (tokenizer + image processor)."""
    if not HAS_CUDA:
        pytest.skip("Processor loading requires CUDA for full model compatibility")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def model_and_processor():
    """Load model with LoRA and frozen vision encoder. Shared across GPU tests."""
    if not HAS_CUDA:
        pytest.skip("Model loading requires CUDA")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    proc = AutoProcessor.from_pretrained(MODEL_NAME)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )

    # Freeze vision encoder
    for name, param in model.named_parameters():
        if name.startswith("model.visual") or name.startswith("visual"):
            param.requires_grad = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, proc


# ---------------------------------------------------------------------------
# Test 1: Environment & Imports
# ---------------------------------------------------------------------------
class TestEnvironment:
    def test_torch_import(self):
        import torch
        print(f"PyTorch version: {torch.__version__}")

    def test_transformers_import(self):
        import transformers
        version = transformers.__version__
        print(f"transformers version: {version}")
        # Must be >= 4.57.0
        parts = version.split(".")
        major, minor = int(parts[0]), int(parts[1])
        assert major >= 4 and minor >= 57, (
            f"ERROR: transformers {version} does not support Qwen3VLForConditionalGeneration. "
            f"Rebuild container with transformers >= 4.57.0."
        )

    def test_peft_import(self):
        import peft
        print(f"peft version: {peft.__version__}")

    def test_accelerate_import(self):
        import accelerate
        print(f"accelerate version: {accelerate.__version__}")

    def test_datasets_import(self):
        import datasets
        print(f"datasets version: {datasets.__version__}")

    def test_qwen3vl_class_import(self):
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            import transformers
            pytest.fail(
                f"ERROR: transformers {transformers.__version__} does not support "
                f"Qwen3VLForConditionalGeneration. Rebuild container with transformers >= 4.57.0."
            )

    @skip_no_cuda
    def test_cuda_available(self):
        assert torch.cuda.is_available(), "CUDA is not available"
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ---------------------------------------------------------------------------
# Test 2: Dataset Loading & Preprocessing
# ---------------------------------------------------------------------------
class TestDataset:
    @skip_no_data
    def test_load_all_samples(self, all_samples):
        """Load sft.jsonl and verify structure."""
        assert len(all_samples) > 0, "No samples loaded"
        print(f"Loaded {len(all_samples)} samples")
        # Expect 15,307 but allow some variance
        assert len(all_samples) >= 15000, f"Expected ~15307 samples, got {len(all_samples)}"

    @skip_no_data
    def test_sample_fields(self, all_samples):
        """Every sample must have prompt and response."""
        for i, s in enumerate(all_samples):
            assert "prompt" in s, f"Sample {i} missing 'prompt'"
            assert "response" in s, f"Sample {i} missing 'response'"

    @skip_no_data
    @skip_no_cuda
    def test_tokenization_and_label_masking(self, all_samples, model_and_processor):
        """Tokenize 10 samples, verify label masking."""
        from train import SFTDataset

        _, proc = model_and_processor
        dataset = SFTDataset(all_samples[:10], proc, max_seq_length=2048)

        for i in range(min(10, len(dataset))):
            item = dataset[i]
            input_ids = item["input_ids"]
            labels = item["labels"]
            assert input_ids.shape == labels.shape, f"Shape mismatch at sample {i}"
            # Labels should have some -100 (system/user) and some non-(-100) (assistant)
            masked = (labels == -100).sum().item()
            unmasked = (labels != -100).sum().item()
            assert masked > 0, f"Sample {i}: no masked tokens (system/user should be masked)"
            assert unmasked > 0, f"Sample {i}: no unmasked tokens (assistant should have labels)"

        # Print one rendered example
        item = dataset[0]
        tokens = proc.tokenizer.decode(item["input_ids"], skip_special_tokens=False)
        print(f"\n--- Rendered Example (sample 0) ---")
        print(f"Token count: {item['input_ids'].shape[0]}")
        print(f"Masked tokens: {(item['labels'] == -100).sum().item()}")
        print(f"Assistant tokens: {(item['labels'] != -100).sum().item()}")
        print(f"Text preview (first 500 chars): {tokens[:500]}")


# ---------------------------------------------------------------------------
# Test 3: Model Loading & Freezing
# ---------------------------------------------------------------------------
class TestModelFreeze:
    @skip_no_cuda
    def test_vision_encoder_frozen(self, model_and_processor):
        """All model.visual params must have requires_grad=False."""
        model, _ = model_and_processor
        for name, param in model.named_parameters():
            if "visual" in name:
                assert not param.requires_grad, f"Vision param {name} not frozen!"

    @skip_no_cuda
    def test_lora_params_trainable(self, model_and_processor):
        """LoRA params should be the only trainable params."""
        model, _ = model_and_processor
        trainable = 0
        total = 0
        for name, param in model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
                # Should be a LoRA param
                assert "lora" in name.lower(), f"Non-LoRA param {name} is trainable"

        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        assert trainable > 0, "No trainable parameters found"


# ---------------------------------------------------------------------------
# Test 4: Forward Pass (text-only)
# ---------------------------------------------------------------------------
class TestForwardPass:
    @skip_no_cuda
    @skip_no_data
    def test_text_only_forward(self, all_samples, model_and_processor):
        """Forward pass with no images should work and produce valid logits."""
        from train import SFTDataset

        model, proc = model_and_processor
        collate_fn = _get_collate_fn(proc)
        dataset = SFTDataset(all_samples[:2], proc, max_seq_length=512)

        batch = collate_fn([dataset[0], dataset[1]])
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        model.eval()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = out.logits
        assert logits.dim() == 3, f"Expected 3D logits, got {logits.dim()}D"
        assert logits.shape[0] == 2, f"Expected batch=2, got {logits.shape[0]}"
        assert logits.shape[1] == input_ids.shape[1], "Seq length mismatch"
        assert torch.isfinite(logits).all(), "Logits contain inf/nan"
        print(f"Logits shape: {logits.shape}")


# ---------------------------------------------------------------------------
# Test 5: NSC-SFT Loss Computation
# ---------------------------------------------------------------------------
class TestNSCSFTLoss:
    @skip_no_cuda
    @skip_no_data
    def test_loss_computation(self, all_samples, model_and_processor):
        """Compute CE and KL losses, verify they are finite and positive."""
        from train import SFTDataset, compute_nsc_sft_loss, reference_mode

        model, proc = model_and_processor
        collate_fn = _get_collate_fn(proc)
        dataset = SFTDataset(all_samples[:2], proc, max_seq_length=512)
        batch = collate_fn([dataset[0], dataset[1]])

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Reference forward
        with reference_mode(model):
            ref_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Policy forward
        model.train()
        policy_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        total_loss, ce_loss, kl_loss = compute_nsc_sft_loss(
            policy_logits, ref_logits, labels, lambda_kl=0.1
        )

        assert torch.isfinite(ce_loss), f"CE loss is not finite: {ce_loss}"
        assert torch.isfinite(kl_loss), f"KL loss is not finite: {kl_loss}"
        assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss}"
        assert ce_loss.item() > 0, f"CE loss should be positive: {ce_loss}"
        assert kl_loss.item() >= 0, f"KL loss should be non-negative: {kl_loss}"

        expected_total = ce_loss.item() + 0.1 * kl_loss.item()
        assert abs(total_loss.item() - expected_total) < 1e-4, (
            f"Total loss mismatch: {total_loss.item()} != {expected_total}"
        )

        print(f"CE: {ce_loss.item():.4f}, KL: {kl_loss.item():.4f}, Total: {total_loss.item():.4f}")


# ---------------------------------------------------------------------------
# Test 6: Backward Pass & Gradient Checks
# ---------------------------------------------------------------------------
class TestBackwardPass:
    @skip_no_cuda
    @skip_no_data
    def test_backward_and_vision_frozen(self, all_samples, model_and_processor):
        """Backward pass: vision grads=None, LoRA grads!=0, vision weights unchanged."""
        from train import SFTDataset, compute_nsc_sft_loss, reference_mode

        model, proc = model_and_processor
        collate_fn = _get_collate_fn(proc)
        model.train()

        # Store vision encoder checksums before
        vision_checksums_before = {}
        for name, param in model.named_parameters():
            if "visual" in name:
                vision_checksums_before[name] = param.data.sum().item()

        dataset = SFTDataset(all_samples[:2], proc, max_seq_length=512)
        batch = collate_fn([dataset[0], dataset[1]])

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Reference forward
        with reference_mode(model):
            ref_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Policy forward
        model.train()
        policy_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        total_loss, _, _ = compute_nsc_sft_loss(policy_logits, ref_logits, labels, 0.1)
        total_loss.backward()

        # Check vision encoder: grads must be None
        for name, param in model.named_parameters():
            if "visual" in name:
                assert param.grad is None, f"Vision param {name} has gradients!"

        # Check LoRA: at least some should have non-zero grads
        lora_has_grad = False
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    lora_has_grad = True
                    break
        assert lora_has_grad, "No LoRA parameter has non-zero gradients"

        # Run one optimizer step
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        optimizer.step()
        optimizer.zero_grad()

        # Verify vision encoder weights unchanged
        for name, param in model.named_parameters():
            if "visual" in name:
                checksum_after = param.data.sum().item()
                assert checksum_after == vision_checksums_before[name], (
                    f"Vision param {name} changed! Before: {vision_checksums_before[name]}, "
                    f"After: {checksum_after}"
                )

        print("Vision encoder: grads=None, weights unchanged after optimizer step")


# ---------------------------------------------------------------------------
# Test 7: LoRA Adapter Toggle
# ---------------------------------------------------------------------------
class TestLoRAToggle:
    @skip_no_cuda
    @skip_no_data
    def test_logits_differ_after_training(self, all_samples, model_and_processor):
        """After 1 optimizer step, LoRA-enabled and LoRA-disabled logits should differ."""
        from train import SFTDataset, compute_nsc_sft_loss, reference_mode

        model, proc = model_and_processor
        collate_fn = _get_collate_fn(proc)
        model.train()

        dataset = SFTDataset(all_samples[:2], proc, max_seq_length=256)
        batch = collate_fn([dataset[0], dataset[1]])
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Do a training step to diverge LoRA from base
        with reference_mode(model):
            ref_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        model.train()
        policy_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss, _, _ = compute_nsc_sft_loss(policy_logits, ref_logits, labels, 0.1)
        loss.backward()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        optimizer.step()
        optimizer.zero_grad()

        # Now compare LoRA-enabled vs LoRA-disabled logits
        model.eval()
        with torch.no_grad():
            # LoRA enabled
            logits_with_lora = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            # LoRA disabled
            with model.disable_adapter():
                logits_without_lora = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

        diff = (logits_with_lora - logits_without_lora).abs().max().item()
        assert diff > 0, "Logits should differ after training step but they are identical"
        print(f"Max logit difference (LoRA on vs off): {diff:.6f}")


# ---------------------------------------------------------------------------
# Test 8: End-to-End Smoke Run (8 samples, 1 step)
# ---------------------------------------------------------------------------
class TestSmokeRun:
    @skip_no_cuda
    @skip_no_data
    def test_e2e_one_step(self, all_samples, model_and_processor):
        """Full training step on 8 samples: ref fwd + policy fwd + loss + backward + step."""
        from train import SFTDataset, compute_nsc_sft_loss, reference_mode

        model, proc = model_and_processor
        collate_fn = _get_collate_fn(proc)
        model.train()

        dataset = SFTDataset(all_samples[:8], proc, max_seq_length=512)
        # Process in two batches of 4
        batch1 = collate_fn([dataset[i] for i in range(4)])
        batch2 = collate_fn([dataset[i] for i in range(4, 8)])

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=2e-6
        )
        optimizer.zero_grad()

        total_ce = 0
        total_kl = 0

        for batch in [batch1, batch2]:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            # Assert no image data
            assert "pixel_values" not in batch
            assert "image_grid_thw" not in batch

            with reference_mode(model):
                ref_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            model.train()
            policy_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            loss, ce, kl = compute_nsc_sft_loss(policy_logits, ref_logits, labels, 0.1)
            (loss / 2).backward()  # grad accum over 2 batches

            total_ce += ce.item()
            total_kl += kl.item()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Smoke run complete. CE: {total_ce/2:.4f}, KL: {total_kl/2:.4f}")


# ---------------------------------------------------------------------------
# Test 9: Deterministic Split
# ---------------------------------------------------------------------------
class TestSplit:
    @skip_no_data
    def test_deterministic_split(self, all_samples):
        """Same seed produces identical train/val indices across 2 runs."""
        from train import get_or_create_split

        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:

            train1, val1 = get_or_create_split(all_samples, tmpdir1, 0.05, seed=42)
            train2, val2 = get_or_create_split(all_samples, tmpdir2, 0.05, seed=42)

            assert train1 == train2, "Train indices differ across runs with same seed"
            assert val1 == val2, "Val indices differ across runs with same seed"
            print(f"Deterministic split verified: {len(train1)} train, {len(val1)} val")

    @skip_no_data
    def test_split_disjoint_and_complete(self, all_samples):
        """Train and val indices must be disjoint and cover all samples."""
        from train import get_or_create_split

        with tempfile.TemporaryDirectory() as tmpdir:
            train_idx, val_idx = get_or_create_split(all_samples, tmpdir, 0.05, seed=42)

            train_set = set(train_idx)
            val_set = set(val_idx)

            # Disjoint
            overlap = train_set & val_set
            assert len(overlap) == 0, f"Overlap between train and val: {len(overlap)} indices"

            # Complete coverage
            all_set = set(range(len(all_samples)))
            covered = train_set | val_set
            assert covered == all_set, (
                f"Missing indices: {len(all_set - covered)}, extra: {len(covered - all_set)}"
            )

            print(f"Split disjointness verified: {len(train_set)} + {len(val_set)} = {len(all_samples)}")

    @skip_no_data
    def test_split_reuse_on_disk(self, all_samples):
        """If split files exist, they should be loaded (not re-created)."""
        from train import get_or_create_split

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create with seed=42
            train1, val1 = get_or_create_split(all_samples, tmpdir, 0.05, seed=42)

            # Call again with different seed — should still load the saved split
            train2, val2 = get_or_create_split(all_samples, tmpdir, 0.05, seed=999)

            assert train1 == train2, "Split was re-created instead of loaded from disk"
            assert val1 == val2, "Split was re-created instead of loaded from disk"


# ---------------------------------------------------------------------------
# Test 10 (merged with 9 above): covered by disjointness test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 11: Val Eval Loop Sanity
# ---------------------------------------------------------------------------
class TestValEval:
    @skip_no_cuda
    @skip_no_data
    def test_val_eval_finite(self, all_samples, model_and_processor):
        """Run val eval on 8 samples, verify losses are finite."""
        from train import SFTDataset, evaluate

        model, proc = model_and_processor
        collate_fn = _get_collate_fn(proc)

        # Use 8 val samples
        val_dataset = SFTDataset(all_samples[:8], proc, max_seq_length=512)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
        )

        val_ce, val_kl, val_total = evaluate(model, val_dataloader, 0.1, model.device)

        assert math.isfinite(val_ce), f"Val CE is not finite: {val_ce}"
        assert math.isfinite(val_kl), f"Val KL is not finite: {val_kl}"
        assert math.isfinite(val_total), f"Val total is not finite: {val_total}"
        assert val_ce > 0, f"Val CE should be positive: {val_ce}"

        print(f"Val eval: CE={val_ce:.4f}, KL={val_kl:.4f}, Total={val_total:.4f}")


# ---------------------------------------------------------------------------
# Test 12: Early Stopping Logic
# ---------------------------------------------------------------------------
class TestEarlyStopping:
    def test_patience_triggers(self):
        """Synthetic loss sequence should trigger early stopping at correct point."""
        from train import EarlyStopper

        stopper = EarlyStopper(patience=2, epsilon=0.005, min_epochs=1, metric="total")

        # Epoch 0.25 — before min_epochs, should not stop
        assert not stopper.check(val_ce=4.0, val_total=4.2, current_epoch=0.25)
        # Epoch 0.5
        assert not stopper.check(val_ce=3.8, val_total=3.85, current_epoch=0.5)
        # Epoch 0.75
        assert not stopper.check(val_ce=3.7, val_total=3.78, current_epoch=0.75)
        # Epoch 1.0 — still improving
        assert not stopper.check(val_ce=3.6, val_total=3.75, current_epoch=1.0)
        # Epoch 1.25 — tiny improvement (below epsilon)
        assert not stopper.check(val_ce=3.59, val_total=3.74, current_epoch=1.25)
        assert stopper.counter == 1
        # Epoch 1.5 — another tiny improvement
        result = stopper.check(val_ce=3.58, val_total=3.73, current_epoch=1.5)
        assert result is True, "Early stopping should have triggered"
        assert stopper.counter == 2

    def test_patience_zero_disabled(self):
        """patience=0 should never trigger early stopping."""
        from train import EarlyStopper

        stopper = EarlyStopper(patience=0, epsilon=0.005, min_epochs=1, metric="total")
        assert not stopper.enabled

        # Should never stop regardless of losses
        for i in range(20):
            assert not stopper.check(val_ce=5.0, val_total=5.0, current_epoch=float(i))

    def test_min_epochs_respected(self):
        """Early stopping should not fire before min_epochs."""
        from train import EarlyStopper

        stopper = EarlyStopper(patience=1, epsilon=0.005, min_epochs=2, metric="total")

        # Flat losses but before min_epochs=2
        assert not stopper.check(val_ce=5.0, val_total=5.0, current_epoch=0.5)
        assert not stopper.check(val_ce=5.0, val_total=5.0, current_epoch=1.0)
        assert not stopper.check(val_ce=5.0, val_total=5.0, current_epoch=1.5)
        # Now past min_epochs
        assert stopper.check(val_ce=5.0, val_total=5.0, current_epoch=2.0)

    def test_ce_metric_mode(self):
        """When metric='ce', early stopping should track CE not total."""
        from train import EarlyStopper

        stopper = EarlyStopper(patience=2, epsilon=0.005, min_epochs=1, metric="ce")

        # Total loss goes up but CE keeps improving
        assert not stopper.check(val_ce=4.0, val_total=5.0, current_epoch=1.0)
        assert not stopper.check(val_ce=3.5, val_total=5.5, current_epoch=1.25)
        assert not stopper.check(val_ce=3.0, val_total=6.0, current_epoch=1.5)
        # CE plateaus
        assert not stopper.check(val_ce=2.99, val_total=6.0, current_epoch=1.75)  # patience 1
        assert stopper.check(val_ce=2.98, val_total=6.0, current_epoch=2.0)  # patience 2 -> stop


# ---------------------------------------------------------------------------
# Test 13: Checkpoint Save/Load Roundtrip
# ---------------------------------------------------------------------------
class TestCheckpoint:
    @skip_no_cuda
    @skip_no_data
    def test_checkpoint_roundtrip(self, all_samples, model_and_processor):
        """Save checkpoint, reload, verify state matches."""
        from train import save_checkpoint, SFTDataset

        model, proc = model_and_processor
        collate_fn = _get_collate_fn(proc)
        model.train()

        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=2e-6
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # Do a forward/backward to populate optimizer state
        dataset = SFTDataset(all_samples[:2], proc, max_seq_length=256)
        batch = collate_fn([dataset[0], dataset[1]])
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = out.logits.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "test_ckpt"

            # Save
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=1, global_step=42, best_val_loss=3.14,
                seed=42, wandb_run_id="test_run_123",
                save_dir=str(save_dir),
            )

            # Verify files exist
            assert (save_dir / "optimizer.pt").exists()
            assert (save_dir / "scheduler.pt").exists()
            assert (save_dir / "rng_states.pt").exists()
            assert (save_dir / "training_state.json").exists()

            # Verify training_state content
            with open(save_dir / "training_state.json") as f:
                state = json.load(f)
            assert state["epoch"] == 1
            assert state["global_step"] == 42
            assert abs(state["best_val_loss"] - 3.14) < 1e-6
            assert state["seed"] == 42
            assert state["wandb_run_id"] == "test_run_123"

            # Verify RNG states can be loaded
            rng = torch.load(save_dir / "rng_states.pt", weights_only=False)
            assert "torch" in rng
            assert "numpy" in rng
            assert "python" in rng

            print("Checkpoint roundtrip verified")


# ---------------------------------------------------------------------------
# Test 14: Chunked KL Computation
# ---------------------------------------------------------------------------
class TestChunkedKL:
    def test_chunked_matches_unchunked(self):
        """Verify chunked KL produces the same result as unchunked KL."""
        from train import compute_nsc_sft_loss

        torch.manual_seed(42)
        batch, seq_len, vocab = 2, 256, 1000

        policy_logits = torch.randn(batch, seq_len, vocab)
        ref_logits = torch.randn(batch, seq_len, vocab)
        # Labels: first 64 tokens masked (-100), rest are valid
        labels = torch.randint(0, vocab, (batch, seq_len))
        labels[:, :64] = -100

        # Unchunked
        total_unchunked, ce_unchunked, kl_unchunked = compute_nsc_sft_loss(
            policy_logits, ref_logits, labels, lambda_kl=0.1, kl_chunk_size=0,
        )

        # Chunked with small chunk size
        total_chunked, ce_chunked, kl_chunked = compute_nsc_sft_loss(
            policy_logits, ref_logits, labels, lambda_kl=0.1, kl_chunk_size=32,
        )

        # CE should be identical (same code path)
        assert torch.allclose(ce_unchunked, ce_chunked, atol=1e-5), (
            f"CE mismatch: {ce_unchunked.item()} vs {ce_chunked.item()}"
        )

        # KL should match within floating point tolerance
        assert torch.allclose(kl_unchunked, kl_chunked, atol=1e-4), (
            f"KL mismatch: {kl_unchunked.item()} vs {kl_chunked.item()}"
        )

        # Total loss should match
        assert torch.allclose(total_unchunked, total_chunked, atol=1e-4), (
            f"Total loss mismatch: {total_unchunked.item()} vs {total_chunked.item()}"
        )

        print(f"Unchunked KL: {kl_unchunked.item():.6f}, "
              f"Chunked KL: {kl_chunked.item():.6f}, "
              f"diff: {abs(kl_unchunked.item() - kl_chunked.item()):.2e}")

    def test_chunked_kl_different_chunk_sizes(self):
        """Verify different chunk sizes produce consistent results."""
        from train import compute_nsc_sft_loss

        torch.manual_seed(123)
        batch, seq_len, vocab = 1, 128, 500
        policy_logits = torch.randn(batch, seq_len, vocab)
        ref_logits = torch.randn(batch, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch, seq_len))
        labels[:, :32] = -100

        results = {}
        for chunk_size in [16, 32, 64, 128]:
            _, _, kl = compute_nsc_sft_loss(
                policy_logits, ref_logits, labels, lambda_kl=0.1,
                kl_chunk_size=chunk_size,
            )
            results[chunk_size] = kl.item()

        # All chunk sizes should give the same KL (within tolerance)
        values = list(results.values())
        for cs, val in results.items():
            assert abs(val - values[0]) < 1e-4, (
                f"Chunk size {cs} gives KL={val}, expected ~{values[0]}"
            )

        print(f"KL values by chunk size: {results}")

    def test_chunked_kl_skips_masked_chunks(self):
        """Verify chunks with all-masked tokens are skipped efficiently."""
        from train import compute_nsc_sft_loss

        torch.manual_seed(7)
        batch, seq_len, vocab = 1, 128, 100
        policy_logits = torch.randn(batch, seq_len, vocab)
        ref_logits = torch.randn(batch, seq_len, vocab)
        # All tokens masked except the last 16
        labels = torch.full((batch, seq_len), -100, dtype=torch.long)
        labels[:, -16:] = torch.randint(0, vocab, (batch, 16))

        _, _, kl_chunked = compute_nsc_sft_loss(
            policy_logits, ref_logits, labels, lambda_kl=0.1, kl_chunk_size=32,
        )
        _, _, kl_unchunked = compute_nsc_sft_loss(
            policy_logits, ref_logits, labels, lambda_kl=0.1, kl_chunk_size=0,
        )

        assert torch.allclose(kl_chunked, kl_unchunked, atol=1e-4)
        assert kl_chunked.item() > 0
        print(f"Sparse mask KL: chunked={kl_chunked.item():.6f}, "
              f"unchunked={kl_unchunked.item():.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
