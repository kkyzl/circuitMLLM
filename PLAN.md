# AnalogSeeker NSC-SFT Training Pipeline — PLAN

---

## (0) Paper Reference & Ideas Being Implemented

**Paper:** "AnalogSeeker: An Open-source Foundation Language Model for Analog Circuit Design"
**Authors:** Zihao Chen, Ji Zhuang, et al. | **ArXiv:** `2508.10409`

**Ideas we are implementing:**

| Concept | Paper Section | Our Adaptation |
|---------|--------------|----------------|
| **NSC-SFT objective** | Eq. 7-8 | `L = CE + lambda * KL(p_theta \|\| p_theta0)` with lambda=0.1 |
| **SFT-centric recipe** (skip CPT) | Sec. 4.2 | Direct SFT on domain data, no continual pre-training |
| **Frozen reference model** | Sec. 4.1 | `theta0` = init checkpoint, frozen, forward with `no_grad` |
| **Flash attention disabled** | Sec. 4.3 | Disabled for KL numerical stability |
| **KL on assistant tokens only** | Eq. 8 | Mask KL + CE to assistant-token positions |

**Key adaptation:** The paper uses Qwen2.5-32B-Instruct (text-only LLM). We adapt this to **Qwen3-VL** (vision-language model) with a **frozen vision encoder**, training text-only.

---

## (1) Dataset Report

**File:** `./analog_data/sft.jsonl` — 107.6 MB, **15,307 samples**, JSONL format

| Property | Value |
|----------|-------|
| Format | JSONL, one JSON object per line |
| Fields | `prompt` (str), `response` (str) |
| Has `<think>` + `</think>` tags | 14,870 / 15,307 (97.1%) |
| Has `<answer>` tag in response | 15,216 / 15,307 (99.4%) |
| `<answer>` format requested in prompt | 15,307 / 15,307 (100%) |

**Length statistics (chars):**

| Metric | Prompt | Response | Combined |
|--------|--------|----------|----------|
| Average | 765 | 5,997 | 6,762 |
| P50 | 724 | 4,790 | 5,514 |
| P95 | 1,157 | 14,570 | 15,727 |
| Max | 8,861 | 36,287 | 45,148 |

**Token estimates** (paper reports 112.65M tokens for this dataset):

| Metric | Est. Tokens |
|--------|-------------|
| Average per sample | ~7,357 |
| P95 per sample | ~15,000-20,000 |
| Total dataset | ~112.65M |

> **Disclaimer:** Token count and time/cost estimates are approximate; final throughput
> will be measured in a short pilot run (`--pilot_steps`) and extrapolated.

**Chat message mapping:**
```
system: "You are an expert in analog circuit design."
user:   {prompt}
assistant: {response}
```

**Reasoning tags:** The `<think>...</think>` and `<answer>...</answer>` tags are part of the response content. They will be preserved as-is (the model learns to produce chain-of-thought reasoning). We use the **Instruct** variant as base (matching the paper), and the model learns the `<think>/<answer>` structure from the SFT data.

---

## (2) Fine-Tuning Qwen3-VL for Text-Only SFT

**HuggingFace classes:**

| Component | Class |
|-----------|-------|
| Model | `Qwen3VLForConditionalGeneration` (required; `AutoModelForCausalLM` won't load VL architecture) |
| Processor | `AutoProcessor.from_pretrained(...)` (wraps tokenizer + image processor) |
| Tokenizer | `Qwen2Tokenizer` (accessed via `processor.tokenizer`) |

**Text-only works natively:** When no `pixel_values` are passed, the vision encoder is never invoked. The model processes text embeddings directly through `model.language_model`.

**Chat template:** ChatML format with `<|im_start|>` / `<|im_end|>` delimiters.

**Label masking strategy:**
- Tokenize the full conversation: `system + user + assistant`
- Set labels to `-100` for all tokens in `system` and `user` turns (including `<|im_start|>`, role tokens, and `<|im_end|>`)
- Keep labels = token_ids only for `assistant` turn content (after `<|im_start|>assistant\n`)
- Include the final `<|im_end|>` token in labels so the model learns to stop
- Both CE and KL losses are computed only where `labels != -100`

**Max sequence length:** We fix max sequence length to 8,192 tokens to align with AnalogSeeker's experimental setting (the paper trains with 8,192). This is an experimental control choice, not a model limitation — Qwen3-VL supports up to 262,144 tokens (`text_config.max_position_embeddings = 262144`). Samples exceeding 8,192 are truncated from the right.

---

## (3) Model Freezing & Trainable Policy

**Target model:** `Qwen/Qwen3-VL-8B-Instruct` (~9B total params)

### Modules to freeze (vision encoder):
```
model.visual.patch_embed.*          # 3D patch embedding
model.visual.blocks.*              # 27 ViT layers
model.visual.merger.*              # Patch merger MLP
model.visual.deepstack_merger_list.*  # DeepStack mergers
```
All parameters under `model.visual` will have `requires_grad = False`.

### Recommended approach: A) LoRA on LLM

**Justification:**
- Memory efficient: keeps two copies of base weights (policy + reference) feasible
- The paper uses full fine-tune on 8x H200s — our constraint is more modest hardware
- LoRA achieves comparable quality for domain adaptation SFT tasks
- Allows reference model to share base weights (no separate copy needed for non-LoRA params)
- Chosen as a standard high-capacity LoRA setting; we do not claim exact equivalence to full FT.

**LoRA configuration:**

| Parameter | Value |
|-----------|-------|
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Rank | 64 |
| Alpha | 128 (alpha/rank = 2) |
| Dropout | 0.05 |
| Bias | none |

**Trainable parameter estimate** for Qwen3-VL-8B:
- LLM has ~36 layers, hidden_dim=4096, intermediate=11,008
- Attention: 4 modules * 2 * r * d = 4 * 2 * 64 * 4096 = 2.1M per layer
- MLP: 3 modules * 2 * r * d_ff ~ 3 * 2 * 64 * 11,008 = 4.2M per layer
- Total per layer: ~6.3M
- Total LoRA: 36 * 6.3M ~ **~227M trainable params** (~2.5% of LLM)
- LoRA weight storage: 227M * 2 bytes = **~454 MB** (BF16)
- Optimizer states (AdamW, FP32 m+v): 227M * 8 bytes = **~1.8 GB**
- Gradients: **~454 MB**

### Option B) Full fine-tune (for reference only):
- All LLM params trainable: ~8.5B params
- Optimizer: 8.5B * 8 = 68 GB (FP32 m+v)
- Needs ZeRO-3 across 4+ GPUs minimum
- Not recommended unless you have 4+ A100-80GB

---

## (4) NSC-SFT Objective — Implementation Details

### Loss formula:
```
L = CE(logits, labels) + lambda_kl * KL(p_theta || p_theta0)
```
where `lambda_kl = 0.1` (from paper).

### KL computation (numerically stable):
```python
# On assistant-token positions only (mask where labels != -100)
log_p = F.log_softmax(policy_logits, dim=-1)      # current model
log_q = F.log_softmax(ref_logits, dim=-1)          # frozen reference
# KL(p || q) = sum_v p(v) * (log p(v) - log q(v))
p = log_p.exp()
kl_per_token = (p * (log_p - log_q)).sum(dim=-1)   # [batch, seq_len]
kl_loss = (kl_per_token * mask).sum() / mask.sum()  # mean over valid positions
```

### Masking:
- `mask = (labels != -100).float()` — assistant tokens only
- Both CE and KL use this mask
- CE: standard `F.cross_entropy(logits, labels, ignore_index=-100, reduction='mean')`
- KL: manually masked and averaged

### Reference model strategy (LoRA adapter toggling):

The reference model IS the base model. Since LoRA only adds adapter weights, the reference forward pass = forward without LoRA. We must also switch to `eval()` mode to disable dropout — the reference distribution must be deterministic (frozen theta_0) for correct KL computation:

```python
# Reference forward: eval() disables dropout so KL target is deterministic
model.eval()
with model.disable_adapter():
    with torch.no_grad():
        ref_logits = model(input_ids, attention_mask=attn_mask).logits

# Policy forward: restore train() for dropout + gradient flow
model.train()
policy_logits = model(input_ids, attention_mask=attn_mask).logits
```

**This eliminates the need for a separate reference model copy**, saving ~18 GB VRAM.

### Efficiency tactics:
- Reference forward with `eval()` + `torch.no_grad()` — deterministic output, no gradient graph
- LoRA adapter toggling — single model copy serves both roles
- Gradient checkpointing on policy forward to reduce activation memory
- BF16 mixed precision throughout
- Flash attention **disabled** (per paper, for KL numerical stability): `attn_implementation="eager"` or `"sdpa"`

---

## (5) GPU / Memory / Cost Estimation

### Base memory components (Qwen3-VL-8B, BF16):

| Component | VRAM |
|-----------|------|
| Model weights (BF16) | 18 GB |
| LoRA weights (BF16) | 0.45 GB |
| Optimizer states (FP32) | 1.8 GB |
| Gradients (BF16) | 0.45 GB |
| Activations (batch=1, seq=4096, grad ckpt) | 2-4 GB |
| Activations (batch=1, seq=8192, grad ckpt) | 4-8 GB |
| Ref forward activations (no_grad, transient) | 1-2 GB |
| CUDA kernels + overhead | 1-2 GB |

**Key insight:** With LoRA adapter toggling, we need only ONE copy of the base model.

> **Disclaimer:** All throughput and time/cost numbers below are estimates. Use
> `--pilot_steps N` to run a short calibration and get actual measurements for your hardware.

### Config 1: Single GPU (A100-80GB / H100-80GB) — RECOMMENDED

| Setting | Value |
|---------|-------|
| GPU | 1x A100-80GB or H100-80GB |
| Model loading | BF16, single GPU |
| Reference model | LoRA adapter toggling (same model) |
| Batch size per GPU | 2 |
| Gradient accumulation | 16 (effective batch = 32) |
| Max seq length | 8,192 |
| Gradient checkpointing | Yes |
| Peak VRAM estimate | **~35-45 GB** |
| Throughput estimate | ~2,000-3,000 tokens/sec |
| Time per epoch (112.65M tokens) | ~10-16 hours |
| Time for 3 epochs | ~30-48 hours |

### Config 2: Multi-GPU (2-4x A100-80GB, DDP)

| Setting | Value |
|---------|-------|
| GPUs | 4x A100-80GB |
| Strategy | DDP (each GPU has full model; no need for sharding at 8B) |
| Batch size per GPU | 2 |
| Gradient accumulation | 4 (effective batch = 32) |
| Max seq length | 8,192 |
| Peak VRAM per GPU | **~35-45 GB** |
| Throughput estimate | ~8,000-12,000 tokens/sec |
| Time per epoch | ~2.5-4 hours |
| Time for 3 epochs | ~7.5-12 hours |

### Config 3: Budget Setup (Single GPU, smaller VRAM)

**Option 3a: L40S 48GB**

| Setting | Value |
|---------|-------|
| GPU | 1x L40S 48GB |
| Batch size | 1 |
| Gradient accumulation | 32 (effective batch = 32) |
| Max seq length | 4,096 (truncated from 8,192) |
| Gradient checkpointing | Yes |
| Peak VRAM | **~30-38 GB** |
| Throughput | ~1,000-1,500 tokens/sec |
| Time per epoch | ~21-31 hours |
| Time for 3 epochs | ~63-93 hours |

**Option 3b: RTX 4090 24GB (tightest fit, QLoRA)**

| Setting | Value |
|---------|-------|
| GPU | 1x RTX 4090 24GB |
| Model loading | BF16 + 4-bit quantization (QLoRA) |
| Quantized model weight | ~5 GB |
| Batch size | 1 |
| Max seq length | 2,048 |
| Gradient checkpointing | Yes |
| Peak VRAM | **~18-22 GB** |
| Throughput | ~500-800 tokens/sec |
| Time per epoch | ~39-63 hours |
| Time for 3 epochs | ~117-189 hours |

### Summary comparison:

| Config | GPU(s) | Peak VRAM | Time/epoch (est.) |
|--------|--------|-----------|-------------------|
| 1 (recommended) | 1x A100-80GB | ~40 GB | 10-16 hr |
| 2 (fast) | 4x A100-80GB | ~40 GB/GPU | 2.5-4 hr |
| 3a (budget) | 1x L40S 48GB | ~34 GB | 21-31 hr |
| 3b (cheapest) | 1x RTX 4090 | ~20 GB | 39-63 hr |

**Cost formula:** `Cost = (Time/epoch) × (number of epochs) × (your $/hr)`.
Use `--pilot_steps` to measure actual throughput on your hardware, then multiply
by your provider's hourly rate. Pricing varies too widely across providers to
give meaningful fixed estimates.

**Recommended epochs:** Default is 1 epoch (staged execution). Manually continue to epoch 2 or 3 via `--resume_from_checkpoint` after inspecting results (see Section 13). Early stopping is available but optional.

---

## (6) Containerization / Singularity Plan

### Reference .def analysis:

The provided `def_reference.md` is a SLURM job script that runs a pre-built SIF with
`singularity run --nv --no-home -B .:/data $SIF`. The `test_reference.md` shows a full
.def structure with `Bootstrap: docker`, `%environment`, `%post` sections and the
local build -> sftp upload workflow.

### Proposed `analogseeker_train.def`:

```def
BootStrap: docker
From: nvidia/cuda:12.4.1-devel-ubuntu22.04

%labels
    Author AnalogSeeker-Training
    Version 1.0

%environment
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/extras/CUPTI/lib64
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/opt/venv/bin:$PATH
    export HF_HOME=/hf_cache
    export TRANSFORMERS_CACHE=/hf_cache
    export TORCH_HOME=/hf_cache/torch
    export PYTHONUNBUFFERED=1
    export LC_ALL=C

%post
    export DEBIAN_FRONTEND=noninteractive
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/extras/CUPTI/lib64

    apt-get -y update
    apt-get install -y --no-install-recommends build-essential
    apt-get install -y --no-install-recommends git
    apt-get install -y --no-install-recommends vim
    apt-get install -y --no-install-recommends wget
    apt-get install -y --no-install-recommends curl
    apt-get install -y --no-install-recommends ca-certificates
    apt-get install -y --no-install-recommends python3.11
    apt-get install -y --no-install-recommends python3.11-venv
    apt-get install -y --no-install-recommends python3.11-dev
    apt-get install -y --no-install-recommends python3-pip
    apt-get install -y --no-install-recommends ninja-build
    rm -rf /var/lib/apt/lists/*

    python3.11 -m venv /opt/venv
    . /opt/venv/bin/activate

    pip install --upgrade pip setuptools wheel

    # PyTorch (CUDA 12.4)
    pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124

    # ML stack (pin versions for reproducibility)
    # Qwen3-VL requires transformers >= 4.57.0.
    # If model loading fails with 4.57.0 (e.g., due to dev-version config mismatch),
    # rebuild with a newer pinned stable release (4.57.1 / 4.58.x).
    # Do NOT use unbounded installs.
    pip install transformers==4.57.0
    pip install accelerate==1.6.0
    pip install peft==0.15.2
    pip install datasets==3.6.0
    pip install bitsandbytes==0.45.5
    pip install deepspeed==0.16.7
    pip install safetensors sentencepiece protobuf
    pip install qwen-vl-utils
    pip install wandb
    pip install pytest

%runscript
    exec python3 "$@"
```

> **Note:** Qwen3-VL requires `transformers >= 4.57.0`. The model config may reference
> `transformers_version: "4.57.0.dev0"`. If stable 4.57.0 fails to load the model,
> bump the pin to the next stable release (e.g., 4.57.1 or 4.58.0) and rebuild.
> Do NOT use unbounded installs.

### What goes WHERE:

| Item | Location | Mount strategy |
|------|----------|----------------|
| Training code (`train.py`, etc.) | Bind-mounted | `-B /path/to/code:/code` |
| Dataset (`analog_data/`) | Bind-mounted | `-B /path/to/data:/data` |
| Model cache (HF downloads) | Bind-mounted | `-B /path/to/hf_cache:/hf_cache` |
| Output (checkpoints, logs) | Bind-mounted | `-B /path/to/output:/output` |
| Python packages | Inside .sif | Baked into container |
| CUDA/cuDNN | Inside .sif | From base image |

### Build & upload workflow (following test_reference.md pattern):

```bash
# 1. Build the .sif on your local VM
[user@localhost ]$ singularity build --fakeroot analogseeker_train.sif analogseeker_train.def

# 2. Run local tests inside the container (see Section 8 below)
[user@localhost ]$ singularity exec --nv analogseeker_train.sif python3 /code/tests/test_pipeline.py

# 3. Upload to the HPC server
[user@localhost ]$ sftp user@server_ip
sftp> put analogseeker_train.sif
sftp> exit
```

### Run commands on the server:

**Single GPU training:**
```bash
apptainer exec --nv --no-home \
    -B ./:/code \
    -B ./analog_data:/data \
    -B /scratch/hf_cache:/hf_cache \
    -B ./output:/output \
    analogseeker_train.sif \
    python3 /code/train.py \
        --model_name Qwen/Qwen3-VL-8B-Instruct \
        --data_path /data/sft.jsonl \
        --output_dir /output \
        --max_seq_length 8192 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --num_train_epochs 1 \
        --learning_rate 2e-6 \
        --lambda_kl 0.1 \
        --lora_rank 64 \
        --bf16 \
        --report_to wandb \
        --wandb_project analogseeker_sft
```

**Multi-GPU with torchrun:**
```bash
apptainer exec --nv --no-home \
    -B ./:/code -B ./analog_data:/data \
    -B /scratch/hf_cache:/hf_cache -B ./output:/output \
    analogseeker_train.sif \
    torchrun --nproc_per_node=4 /code/train.py [args...]
```

**SLURM job script (`run_train.sh`):**
```bash
#!/bin/bash
#SBATCH --job-name=analogseeker_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --account=PROJECT_ID
#SBATCH --partition=normal
#SBATCH -o %j_train.out
#SBATCH -e %j_train.err

module purge
ml singularity

SIF=/path/to/analogseeker_train.sif

apptainer exec --nv --no-home \
    -B ./:/code -B ./analog_data:/data \
    -B /scratch/hf_cache:/hf_cache -B ./output:/output \
    $SIF \
    torchrun --nproc_per_node=4 /code/train.py \
        --model_name Qwen/Qwen3-VL-8B-Instruct \
        --data_path /data/sft.jsonl \
        --output_dir /output \
        --num_train_epochs 1 \
        --bf16 \
        --report_to wandb \
        --wandb_project analogseeker_sft
```

### Reproducibility:
- **Seed:** Configurable via `--seed` arg, default 42
- **Config saving:** All args saved as `training_config.json` in output dir
- **Git hash:** Saved if repo available
- **WandB:** Enabled by default via `--report_to wandb` (required for all training runs)

### WandB Logging (Required):

WandB is used to record all training runs. The following metrics are logged every step:

| Metric | WandB Key | Description |
|--------|-----------|-------------|
| Total loss | `train/loss` | CE + lambda_kl * KL |
| CE loss | `train/ce_loss` | Cross-entropy on assistant tokens |
| KL loss | `train/kl_loss` | KL divergence to frozen reference |
| Learning rate | `train/learning_rate` | Current LR from scheduler |
| Epoch | `train/epoch` | Fractional epoch progress |
| Grad norm | `train/grad_norm` | Gradient norm (for stability monitoring) |
| Tokens/sec | `train/tokens_per_sec` | Training throughput |

**Setup:** `WANDB_API_KEY` must be set as an environment variable (or logged in via `wandb login`). Pass it into the container:
```bash
apptainer exec --nv --no-home \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    -B ./:/code -B ./analog_data:/data \
    -B /scratch/hf_cache:/hf_cache -B ./output:/output \
    analogseeker_train.sif \
    python3 /code/train.py \
        --wandb_project analogseeker_sft \
        --wandb_run_name "qwen3vl-8b-lora-r64-nsc-sft" \
        [other args...]
```

**Offline mode** (for air-gapped HPC nodes):
```bash
--env WANDB_MODE=offline
# Sync later: wandb sync /output/wandb/offline-run-*
```

---

## (7) Architecture Diagram

```
                 +--------------------------+
                 |    Qwen3-VL-8B-Instruct  |
                 |                          |
    FROZEN ----> |  model.visual (ViT)      |  <-- requires_grad=False
                 |    patch_embed           |
                 |    blocks[0..26]         |
                 |    merger                |
                 |    deepstack_merger      |
                 |                          |
   TRAINABLE --> |  model.language_model    |  <-- LoRA adapters (r=64)
    (LoRA)       |    embed_tokens          |
                 |    layers[0..35]         |
                 |    norm                  |
                 |    lm_head              |
                 +------------+-------------+
                              |
          +-------------------+-------------------+
          |                                       |
    LoRA enabled                            LoRA disabled
    (policy model)                         (reference model)
          |                                       |
          v                                       v
    policy_logits                           ref_logits
          |                                       |
          +-------------------+-------------------+
                              |
                    +---------v----------+
                    |     NSC-SFT        |  <-- mask: labels != -100
                    |                    |
                    | CE(policy, y)      |
                    | + 0.1 * KL(p||ref) |
                    +--------------------+
```

---

## (8) Local Testing / Unit Tests (Pre-Upload Verification)

Before uploading the `.sif` to the HPC server, run these tests locally inside the
container to catch issues early. This follows the build-test-upload workflow from
`test_reference.md`.

### Test file: `tests/test_pipeline.py`

Uses `pytest` and is designed to run inside the Singularity container **with GPU**.

#### Test 1: Environment & imports
- Verify `torch`, `transformers`, `peft`, `accelerate`, `datasets` are importable
- Print `transformers.__version__` prominently
- Verify `torch.cuda.is_available() == True`
- Verify `transformers` version >= 4.57.0
- Attempt `from transformers import Qwen3VLForConditionalGeneration`; if import fails, print: `"ERROR: transformers {version} does not support Qwen3VLForConditionalGeneration. Rebuild container with transformers >= 4.57.0."`

#### Test 2: Dataset loading & preprocessing
- Load `sft.jsonl`, parse all 15,307 entries
- Verify each entry has `prompt` and `response` keys
- Apply chat template to 10 samples, verify tokenized output shape
- Verify label masking: system/user tokens are -100, assistant tokens are not
- Print 1 rendered example for visual inspection

#### Test 3: Model loading & freezing
- Load `Qwen3-VL-8B-Instruct` (or 2B for faster local test)
- Verify `model.visual` parameters all have `requires_grad == False`
- Apply LoRA, verify only LoRA params have `requires_grad == True`
- Print trainable vs total parameter counts

#### Test 4: Forward pass (text-only)
- Run a forward pass on 2 tokenized samples (no images)
- Verify output logits shape: `[batch, seq_len, vocab_size]`
- Verify loss is finite

#### Test 5: NSC-SFT loss computation
- Run policy forward (LoRA enabled) -> get policy_logits
- Run reference forward (LoRA disabled, no_grad) -> get ref_logits
- Compute CE loss and KL loss separately
- Verify both are finite, positive, and reasonable magnitude
- Verify total loss = CE + 0.1 * KL

#### Test 6: Backward pass & gradient checks
- Run backward on the NSC-SFT loss
- Assert vision encoder gradients are `None` for ALL `model.visual` params
- Assert at least some LoRA params have non-zero gradients
- Store vision encoder weight checksums before/after step
- Run one optimizer step
- Verify vision encoder weights are UNCHANGED (checksums match)

#### Test 7: LoRA adapter toggle
- Compute logits with LoRA enabled
- Compute logits with LoRA disabled
- After at least 1 training step, verify the two give DIFFERENT logits
- (Before any training step they would be identical)

#### Test 8: End-to-end smoke run (8 samples, 1 step)
- Load 8 samples, tokenize, create a batch
- Run full training step (ref forward + policy forward + loss + backward + optimizer.step)
- Verify training completes without error
- Log CE loss and KL loss values

### How to run locally:

```bash
# Build container
[user@localhost ]$ singularity build --fakeroot analogseeker_train.sif analogseeker_train.def

# Run all unit tests
[user@localhost ]$ singularity exec --nv \
    -B ./:/code \
    -B ./analog_data:/data \
    -B /path/to/hf_cache:/hf_cache \
    analogseeker_train.sif \
    python3 -m pytest /code/tests/test_pipeline.py -v --tb=short

# Run quick subset (env + dataset only, no GPU model loading)
[user@localhost ]$ singularity exec --nv \
    -B ./:/code -B ./analog_data:/data \
    analogseeker_train.sif \
    python3 -m pytest /code/tests/test_pipeline.py -v -k "env or dataset" --tb=short
```

### Test with smaller model for fast local iteration:

For local testing without 18+ GB of VRAM, use `Qwen3-VL-2B-Instruct` (~5 GB) by
passing `--model_name Qwen/Qwen3-VL-2B-Instruct` to the tests. The architecture
is identical so all checks remain valid.

```bash
# Fast local test with 2B model
[user@localhost ]$ singularity exec --nv \
    -B ./:/code -B ./analog_data:/data -B /path/to/hf_cache:/hf_cache \
    analogseeker_train.sif \
    python3 -m pytest /code/tests/test_pipeline.py -v \
        --model_name Qwen/Qwen3-VL-2B-Instruct
```

---

## (9) Deliverables — Files to Create

| File | Description |
|------|-------------|
| `train.py` | Main training script: data loading, model setup, NSC-SFT training loop |
| `tests/test_pipeline.py` | Unit tests for local container verification (8 tests) |
| `analogseeker_train.def` | Singularity container definition file |
| `run_train.sh` | SLURM submission script with multi-GPU support |
| `ds_zero2_config.json` | DeepSpeed ZeRO-2 config (for multi-GPU, optional) |

### No files modified:
- `analog_data/sft.jsonl` — untouched
- `dataset.py` — untouched
- `test_reference.md` — untouched
- `def_reference.md` — untouched

---

## (10) Data Split & Evaluation Structure

### 10.1 Train / Val Split

**Ratio:** 95% train / 5% val (14,541 train, 766 val)

**Justification:** We use 95/5 to maximize training data while keeping enough samples for stable loss monitoring.

**Deterministic splitting:**
```python
import numpy as np
rng = np.random.RandomState(seed)           # default seed=42
indices = rng.permutation(len(dataset))
split_point = int(len(dataset) * 0.95)
train_indices = sorted(indices[:split_point].tolist())
val_indices = sorted(indices[split_point:].tolist())
```

**Saved artifacts** (written once at start of training):
```
output_dir/
  splits/
    train_indices.json    # list of int indices into sft.jsonl
    val_indices.json      # list of int indices into sft.jsonl
    split_config.json     # {"seed": 42, "train_ratio": 0.95, "n_train": 14541, "n_val": 766}
```

If `output_dir/splits/train_indices.json` already exists, it is loaded directly
(no re-shuffle), guaranteeing identical splits across resumed runs.

**CLI args:**
```
--val_split_ratio 0.05      # default 0.05 (5% val)
--seed 42                   # controls split + training RNG
```

### 10.2 What Is Val vs. What Is External Eval

| Category | Source | Used for | When run |
|----------|--------|----------|----------|
| **Val (dev)** | 5% held-out from `sft.jsonl` | Loss monitoring, early stopping, format check | Every `eval_interval` during training |
| **External eval (test)** | AMSBench / CircuitSense / other benchmarks | Final model assessment, paper-reportable numbers | Only after training, via `--run_external_eval` |

**Separation guarantee:**
- Val metrics (val/ce_loss, val/kl_loss, val/total_loss) are used for checkpoint
  selection and early stopping.
- External eval is NEVER used for checkpoint selection or early stopping.
- External eval lives in a separate function (`run_external_eval()`) that is only
  called after training completes (or explicitly at checkpoint boundaries via
  `--eval_at_checkpoints`), never automatically during training.

### 10.3 External Eval Integration

**CLI flag:** `--run_external_eval` (default: off)

When enabled:
- After training finishes, load best checkpoint and run external eval.
- Results logged to WandB under `eval/external/*` and saved to
  `output_dir/external_eval_results.json`.
- If no external eval scripts exist in the repo, this is a no-op with a warning.

Optional: `--eval_at_checkpoints` — run external eval at each checkpoint boundary
(expensive; off by default).

### 10.4 Validation Metrics Computed

At each eval point, on the full val split (766 samples):

| Metric | WandB Key | Description |
|--------|-----------|-------------|
| Val CE loss | `val/ce_loss` | Cross-entropy on assistant tokens (val split) |
| Val KL loss | `val/kl_loss` | KL divergence to reference (val split) |
| Val total loss | `val/total_loss` | CE + 0.1 * KL (val split) |

At each eval point, on a fixed subset of 64 val samples (format compliance):

| Metric | WandB Key | Description |
|--------|-----------|-------------|
| Answer format rate | `val/answer_format_rate` | Fraction with `<answer>...</answer>` block in generated output |

The format compliance check runs greedy generation (max 2048 new tokens) on the
64-sample fixed subset. This is ~2-5 min per eval on A100.

---

## (11) Epoch Decision & Early Stopping Policy

### 11.1 Core Design — Two Modes

Training supports two modes of operation:

**Mode A — Staged execution (default):**
- `--num_train_epochs 1 --early_stop_patience 0`
- Train exactly 1 epoch, save checkpoint, stop.
- User inspects val metrics and WandB logs, then decides whether to continue.
- Resume via `--resume_from_checkpoint` (see Section 13).

**Mode B — Unattended multi-epoch (opt-in):**
- `--num_train_epochs 3 --early_stop_patience 2`
- Train up to 3 epochs with automatic early stopping.
- Use when you want hands-off training after the pilot run confirms throughput.

**Eval schedule (both modes):**
- Evaluate every **0.25 epoch** (4 times per epoch).
- With batch=32 and 14,541 training samples: **eval every ~113 steps**.
- Configurable via `--eval_steps` (explicit step count) or `--evals_per_epoch 4`.

### 11.2 Early Stopping Rule

**Tracked metric:** `val/total_loss` (CE + lambda*KL on val split).

**Justification for val/total_loss over val/ce_loss:** The total loss is what we
actually optimize. Using val/ce_loss alone would ignore the KL regularization
signal, which could mask cases where the model drifts from the reference distribution
even while CE stays low. Tracking the combined loss ensures early stopping reflects
the full NSC-SFT objective.

**Fallback note:** We log `val/ce_loss` and `val/kl_loss` alongside `val/total_loss`
at every eval point. If KL divergence dominates or becomes noisy (e.g., oscillating
while CE steadily improves), the early-stopping criterion can be switched to
`val/ce_loss` via `--early_stop_metric ce`. This is a documented manual fallback,
not an automatic switch. Default remains `total`.

**Stopping criterion:**
```
if best_val_loss has not improved by at least epsilon (relative)
   for patience consecutive eval points:
       stop training
```

**Default hyperparameters:**

| Parameter | CLI flag | Default | Meaning |
|-----------|----------|---------|---------|
| Max epochs | `--num_train_epochs` | 1 | Epochs to train in this run (default: 1 for staged execution) |
| Min epochs | `--min_epochs` | 1 | Must finish at least 1 epoch before early stop can fire |
| Epsilon | `--early_stop_epsilon` | 0.005 | Minimum relative improvement (0.5%) |
| Patience | `--early_stop_patience` | 0 | 0 = disabled (default); set to 2+ for unattended mode |
| Metric | `--early_stop_metric` | total | Which val metric to track: `total` (CE+KL) or `ce` (CE only) |

**Example walkthrough:**
```
Eval 1 (step 113, epoch 0.25): val_loss=4.20 -> best=4.20
Eval 2 (step 226, epoch 0.50): val_loss=3.85 -> best=3.85 (improved 8.3%)
Eval 3 (step 339, epoch 0.75): val_loss=3.78 -> best=3.78 (improved 1.8%)
Eval 4 (step 454, epoch 1.00): val_loss=3.75 -> best=3.75 (improved 0.8%)
  [epoch 1 complete, early stopping now eligible]
Eval 5 (step 567, epoch 1.25): val_loss=3.74 -> best=3.74 (improved 0.3% < 0.5%) patience=1/2
Eval 6 (step 680, epoch 1.50): val_loss=3.73 -> best=3.73 (improved 0.3% < 0.5%) patience=2/2
  -> EARLY STOP: no meaningful improvement for 2 consecutive evals
  -> Load best checkpoint (step 454, val_loss=3.75 or whichever was actual best)
```

**Disable early stopping:** `--early_stop_patience 0` (trains for all `num_train_epochs`).

### 11.3 Epoch Decision Logic (Human-Readable)

This is the automatic consequence of the early stopping logic above:

1. **Always finish epoch 1.** Early stopping cannot fire during epoch 1 (`min_epochs=1`).
2. **Continue to epoch 2** only if val_loss improved meaningfully during epoch 1
   (i.e., patience hasn't exhausted by the end of epoch 1).
3. **Continue to epoch 3** only if val_loss continues improving during epoch 2.
4. If the model plateaus at any point after epoch 1, training stops.

### 11.4 Checkpointing Strategy

**Save points:** At each eval boundary (every 0.25 epoch = every ~113 steps).

**What is kept on disk:**

| Checkpoint | Path | When kept |
|------------|------|-----------|
| Best (by val/total_loss) | `output_dir/checkpoints/best/` | Always kept, overwritten when new best found |
| Last | `output_dir/checkpoints/last/` | Always kept, overwritten each eval |
| Epoch boundary | `output_dir/checkpoints/epoch-{N}/` | Kept at epoch 1, 2, 3 boundaries |

**What is saved per checkpoint** (all needed for clean resume):
- `adapter_model.safetensors` — LoRA adapter weights
- `optimizer.pt` — AdamW optimizer state dict
- `scheduler.pt` — LR scheduler state dict
- `rng_states.pt` — torch, CUDA, numpy, python random states
- `training_state.json` — epoch, global_step, best_val_loss, seed, wandb_run_id

**Disk usage estimate:** ~600 MB per checkpoint (LoRA ~454 MB + optimizer ~100 MB + scheduler/RNG ~1 MB).
With best + last + up to 3 epoch checkpoints = ~3 GB max.

### 11.5 Eval Schedule Summary

For 3 epochs with 95/5 split and batch=32:

| Eval # | Step | Epoch | Action |
|--------|------|-------|--------|
| 1 | 113 | 0.25 | Val loss; log to WandB |
| 2 | 226 | 0.50 | Val loss; log |
| 3 | 339 | 0.75 | Val loss; log |
| 4 | 454 | 1.00 | Val loss + format check; save epoch-1 ckpt; early stop eligible |
| 5 | 567 | 1.25 | Val loss; early stop check |
| 6 | 680 | 1.50 | Val loss; early stop check |
| 7 | 793 | 1.75 | Val loss; early stop check |
| 8 | 908 | 2.00 | Val loss + format check; save epoch-2 ckpt; early stop check |
| 9 | 1021 | 2.25 | Val loss; early stop check |
| 10 | 1134 | 2.50 | Val loss; early stop check |
| 11 | 1247 | 2.75 | Val loss; early stop check |
| 12 | 1362 | 3.00 | Val loss + format check; save epoch-3 ckpt; training ends |

Format compliance check runs at epoch boundaries (evals 4, 8, 12) to limit cost.

---

## (13) Staged Execution & Resume

### 13.1 Default Behavior

**Default: train for exactly 1 epoch, then stop cleanly.**

The user inspects val metrics and WandB logs after epoch 1, then manually decides
whether to continue. This replaces the previous "run 3 epochs with automatic early
stopping" default. Early stopping remains available for unattended runs
(`--early_stop_patience 2 --num_train_epochs 3`), but is not the default.

### 13.2 Resume via `--resume_from_checkpoint`

**CLI flag:** `--resume_from_checkpoint <path>` (default: None)

When specified, the training script:
1. Loads LoRA adapter weights from `<path>/adapter_model.safetensors`
2. Loads optimizer state from `<path>/optimizer.pt`
3. Loads LR scheduler state from `<path>/scheduler.pt`
4. Loads RNG states from `<path>/rng_states.pt` (torch, numpy, python random, CUDA)
5. Reads `<path>/training_state.json` for: `epoch`, `global_step`, `best_val_loss`
6. Loads split indices from `output_dir/splits/` (NOT re-shuffled)
7. Continues training from `epoch` up to `--num_train_epochs`

**Note:** Resume is designed for epoch-boundary checkpoints. No mid-epoch
fast-forwarding is attempted; the dataloader restarts from the beginning of
the next epoch with restored RNG state.

**Epoch counting on resume:**
- `training_state.json` records the completed epoch (e.g., `"epoch": 1`)
- On resume with `--num_train_epochs 3`, training continues for epochs 2 and 3
- The `--num_train_epochs` flag means "total target epochs", not "additional epochs"
- If `num_train_epochs <= completed_epoch`, training exits immediately with a message

**LR scheduler on resume:**
- Scheduler state is saved/loaded
- If `num_train_epochs` changes on resume (e.g., 1→3), the cosine schedule is
  re-initialized for the new total steps, starting from the restored LR value
- This means the cosine shape may differ from a single uninterrupted run

**WandB resume:**
- If WandB is enabled, pass `resume="must"` with the same `wandb_run_id` (saved in
  `training_state.json`) to continue logging to the same run
- Epoch numbers in logs remain consistent (epoch 2 logs as epoch 2, not epoch 1)

### 13.3 Example Usage

```bash
# Run 1: Train epoch 1
python train.py \
    --num_train_epochs 1 \
    --output_dir /output

# Inspect results: check WandB, val losses, etc.
# Decision: continue to epoch 3.

# Run 2: Resume and train epochs 2-3
python train.py \
    --num_train_epochs 3 \
    --resume_from_checkpoint /output/checkpoints/epoch-1 \
    --output_dir /output
```

### 13.4 Updated Checkpoint Contents

Each epoch-boundary checkpoint (`output_dir/checkpoints/epoch-{N}/`) saves:

| File | Contents |
|------|----------|
| `adapter_model.safetensors` | LoRA adapter weights |
| `optimizer.pt` | AdamW optimizer state dict |
| `scheduler.pt` | LR scheduler state dict |
| `rng_states.pt` | `{"torch": ..., "cuda": ..., "numpy": ..., "python": ...}` |
| `training_state.json` | `{"epoch": N, "global_step": ..., "best_val_loss": ..., "seed": ..., "wandb_run_id": ...}` |

The `best/` and `last/` checkpoints also save the same set of files.

---

## (14) Updated Deliverables — Files to Create/Modify

| File | Description | Change type |
|------|-------------|-------------|
| `train.py` | Main training script: data loading, splitting, model setup, NSC-SFT loop, val eval, early stopping, checkpointing | **Create** |
| `tests/test_pipeline.py` | Unit tests for container verification (existing 8 tests + new split/eval tests) | **Create** |
| `analogseeker_train.def` | Singularity container definition file | **Create** |
| `run_train.sh` | SLURM submission script | **Create** |
| `ds_zero2_config.json` | DeepSpeed ZeRO-2 config (optional) | **Create** |

### New CLI args added to `train.py`:

```
# Split / Eval
--val_split_ratio 0.05
--evals_per_epoch 4          # or --eval_steps N
--format_check_samples 64

# Training schedule
--num_train_epochs 1         # default 1 (staged execution); set 3 for full run
--resume_from_checkpoint     # path to checkpoint dir (e.g., output/checkpoints/epoch-1)

# Early stopping (optional, not default)
--early_stop_epsilon 0.005
--early_stop_patience 0      # 0 = disabled (default); set to 2+ for unattended mode
--early_stop_metric total    # {total, ce} — which val metric to track
--min_epochs 1

# Pilot run
--pilot_steps 0              # if >0, run N steps then exit with timing report

# External eval
--run_external_eval           # flag, off by default
--eval_at_checkpoints         # flag, off by default
```

### New unit tests (added to `tests/test_pipeline.py`):

| Test | Description |
|------|-------------|
| Test 9: Deterministic split | Verify same seed produces identical train/val indices across 2 runs |
| Test 10: Split disjointness | Verify train and val index sets are disjoint and cover all samples |
| Test 11: Val eval loop | Run val eval on 8 val samples, verify val/ce_loss and val/kl_loss are finite |
| Test 12: Early stopping logic | Unit test the patience/epsilon decision with synthetic loss sequences |
| Test 13: Checkpoint save/load | Save checkpoint, reload, verify optimizer/scheduler/RNG states match |

### Output directory structure:

```
output_dir/
  splits/
    train_indices.json
    val_indices.json
    split_config.json
  checkpoints/
    best/                    # best by val/total_loss
      adapter_model.safetensors
      optimizer.pt
      scheduler.pt
      rng_states.pt
      training_state.json
    last/
      adapter_model.safetensors
      optimizer.pt
      scheduler.pt
      rng_states.pt
      training_state.json
    epoch-1/                 # always saved (resumable)
      adapter_model.safetensors
      optimizer.pt
      scheduler.pt
      rng_states.pt
      training_state.json
    epoch-2/                 # only if training reaches epoch 2
    epoch-3/                 # only if training reaches epoch 3
  training_config.json       # all CLI args
  pilot_metrics.json         # only if --pilot_steps > 0
  wandb/                     # WandB logs (if offline mode)
  external_eval_results.json # only if --run_external_eval
```
