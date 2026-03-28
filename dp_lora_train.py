#!/usr/bin/env python3
"""DP-LoRA fine-tuning of Apertus 8B on clinical text.

Implements Phases 1–2b of the DP-LoRA plan:
  --mode baseline   Non-DP LoRA causal LM training (Phase 1)
  --mode validate   Opacus model compatibility check (Phase 1.5)
  --mode dp         DP-SGD LoRA training with Opacus (Phase 2a/2b)

Usage:
    # Phase 1: Non-DP baseline (uses same data as DP-SGD Megatron pipeline)
    python dp_lora_train.py --config configs/dp_lora.yaml --mode baseline \
        --data-path ../synthetic-clinical-data/data/5000_patients/clinical_episodes.jsonl

    # Phase 1.5: Opacus validation (no training, just checks)
    python dp_lora_train.py --config configs/dp_lora.yaml --mode validate

    # Phase 2a: DP training (example-level, synthetic data)
    python dp_lora_train.py --config configs/dp_lora.yaml --mode dp \
        --data-path ../synthetic-clinical-data/data/5000_patients/clinical_episodes.jsonl

    # Phase 2b: DP training (patient-level, same data — already patient-grouped)
    python dp_lora_train.py --config configs/dp_lora.yaml --mode dp \
        --data-path data/patient_dataset

    # Override any config from CLI
    python dp_lora_train.py --config configs/dp_lora.yaml --mode dp \
        --dp-epsilon 1.0 --learning-rate 5e-4
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Fix for PyTorch < 2.4: nn.RMSNorm doesn't exist, but newer opacus references it.
# This stub is never used in practice (Llama uses LlamaRMSNorm, not nn.RMSNorm).
if not hasattr(nn, "RMSNorm"):
    nn.RMSNorm = nn.Identity

import yaml
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All configuration for DP-LoRA training."""

    # Mode
    mode: str = "baseline"  # baseline | validate | dp

    # Model
    model_name_or_path: str = "alehc/swissai-apertus-8b"
    dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0  # MUST be 0 under DP
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Data
    data_path: str = ""  # JSONL file or HF Dataset directory (from prepare_patient_data.py)
    dataset_name: Optional[str] = None  # HF Hub dataset (alternative to data_path)
    dataset_text_field: str = "text"
    max_length: int = 4096

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    per_device_batch_size: int = 4  # Physical batch (fits in GPU memory)
    gradient_accumulation_steps: int = 1  # Baseline only — set to dp_target_batch_size/per_device_batch_size for apples-to-apples
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_steps: int = -1  # -1 = use num_epochs
    seed: int = 42

    # DP parameters
    dp_epsilon: float = 3.0
    dp_delta: float = 1e-5  # Should be < 1/N
    dp_max_grad_norm: float = 1.0
    dp_target_batch_size: int = 512  # Logical batch for DP accounting
    dp_noise_multiplier: Optional[float] = None  # Override sigma directly (bypasses epsilon→sigma)
    dp_loss_aggregation: str = "sum"  # sum or mean — must match DP-SGD (sum = per-example gradients at natural scale)

    # Clean experiment: precomputed batch indices for identical data ordering
    batch_indices_file: Optional[str] = None  # JSON file with precomputed Poisson batch indices

    # Output
    output_dir: str = "output/dp_lora"
    logging_steps: int = 1
    save_steps: int = 0  # 0 = save only at end
    resume_from: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        known = set(cls.__dataclass_fields__.keys())
        unknown = set(data.keys()) - known
        if unknown:
            logger.warning(
                f"Unknown keys in {path} (ignored, possible typo): {sorted(unknown)}"
            )
        return cls(**{k: v for k, v in data.items() if k in known})

    def update_from_args(self, args: argparse.Namespace):
        """Override config values with CLI arguments (non-None only)."""
        for key, value in vars(args).items():
            if value is not None and key != "config" and hasattr(self, key):
                setattr(self, key, value)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClinicalTextDataset(Dataset):
    """Tokenized clinical text dataset for causal LM.

    Each item is one privacy unit (patient or episode), tokenized and padded
    to max_length. No packing — each sequence is independent.
    """

    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class PrecomputedBatchSampler:
    """Deterministic batch sampler from precomputed Poisson indices.

    Ensures identical data ordering across non_dp and dp runs for
    clean experiment comparison (like DP-SGD Phase 0).
    """

    def __init__(self, batches_file: str):
        with open(batches_file) as f:
            self.batches = json.load(f)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def generate_poisson_batches(
    N: int, sample_rate: float, num_steps: int, seed: int, output_file: str,
):
    """Generate Poisson batch indices and save to JSON.

    Each step samples each of the N examples independently with probability
    sample_rate, producing variable-size batches (expected size = N * sample_rate).
    """
    gen = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_steps):
        mask = torch.rand(N, generator=gen) < sample_rate
        indices = mask.nonzero(as_tuple=False).squeeze(1).tolist()
        batches.append(indices)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(batches, f)
    sizes = [len(b) for b in batches]
    logger.info(
        f"Generated {num_steps} Poisson batches (q={sample_rate:.4f}, "
        f"sizes: {min(sizes)}-{max(sizes)}, mean={sum(sizes)/len(sizes):.0f}) → {output_file}"
    )
    return batches


def load_data(config: Config, tokenizer) -> tuple[ClinicalTextDataset, int]:
    """Load clinical text data from JSONL, HF Dataset, or HF Hub.

    Supports the same JSONL format used by the DP-SGD Megatron pipeline
    (synthetic-clinical-data/data/5000_patients/clinical_episodes.jsonl).
    """
    texts = []

    if config.data_path:
        path = Path(config.data_path)

        if path.is_file() and path.suffix == ".jsonl":
            # Raw JSONL file (same format as DP-SGD pipeline)
            logger.info(f"Loading JSONL from {path}")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        texts.append(record[config.dataset_text_field])

        elif path.is_dir():
            # HF Dataset saved with save_to_disk()
            logger.info(f"Loading HF Dataset from {path}")
            from datasets import load_from_disk
            ds = load_from_disk(str(path))
            texts = ds[config.dataset_text_field]

        else:
            raise ValueError(f"data_path must be a .jsonl file or directory: {path}")

    elif config.dataset_name:
        logger.info(f"Loading HF Hub dataset: {config.dataset_name}")
        from datasets import load_dataset
        ds = load_dataset(config.dataset_name, split="train")
        texts = ds[config.dataset_text_field]

    else:
        raise ValueError("Must specify data_path or dataset_name")

    N = len(texts)
    logger.info(f"Loaded {N} privacy units")

    dataset = ClinicalTextDataset(texts, tokenizer, config.max_length)
    return dataset, N


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    model, batch: dict, vocab_size: int, aggregation: str = "sum",
) -> tuple[torch.Tensor, float]:
    """Compute causal LM loss with configurable batch aggregation.

    Per-example loss is always mean-over-tokens (length-invariant per example).
    Batch aggregation controls how per-example losses combine for backward():
      - "sum": loss = Σ_i L_i — per-sample gradients at natural scale, C directly
               comparable to DP-SGD Megatron. Opacus divides by B after clipping+noise.
      - "mean": loss = (1/B) Σ_i L_i — per-sample gradients scaled by 1/B.

    Returns (loss_for_backward, mean_loss_for_logging).
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].float()

    # Per-token cross-entropy (no reduction)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    ).view(input_ids.size(0), -1)

    # Per-example loss: mean over tokens (length-invariant)
    per_example_loss = (per_token_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

    # Always compute mean for logging (comparable across batch sizes)
    mean_loss = per_example_loss.mean()

    # Batch aggregation for backward
    if aggregation == "sum":
        return per_example_loss.sum(), mean_loss.item()
    else:
        return mean_loss, mean_loss.item()


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def _create_random_llama(hidden_size=256, num_layers=2, num_heads=4):
    """Create a random-init small Llama model for sanity testing.

    Default: 2L/256H/4A (matches DP-SGD Phase 0 sanity config).
    Same architecture as Apertus 8B (LlamaForCausalLM) so LoRA target modules
    (q_proj, k_proj, etc.) are identical.

    Vocab size is set to match whichever tokenizer is available (GPT-2 = 50257,
    Llama = 32000+). For sanity testing the exact vocab doesn't matter — we
    just need the DP machinery to work on the correct architecture.

    For the full 125M config (matches DP-SGD run_125m_dp.sh), use:
        _create_random_llama(hidden_size=768, num_layers=12, num_heads=12)
    """
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

    # Get a tokenizer first — determines vocab_size
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    llama_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        vocab_size=vocab_size,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(llama_config)
    logger.info(f"Created random Llama: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params "
                f"({num_layers}L, {hidden_size}H, {num_heads}A, vocab={vocab_size})")
    return model, tokenizer


def load_model_and_tokenizer(config: Config):
    """Load pretrained model, apply LoRA, return (model, tokenizer, vocab_size)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map.get(config.dtype, torch.bfloat16)

    if config.model_name_or_path == "random_llama_125m":
        model, tokenizer = _create_random_llama()
        model = model.to(dtype=model_dtype)
        # Save for reproducibility / cross-pipeline comparison with DP-SGD
        init_dir = Path(config.output_dir) / "random_llama_125m_init"
        if not init_dir.exists():
            init_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(init_dir))
            tokenizer.save_pretrained(str(init_dir))
            logger.info(f"Saved random init to {init_dir} (reuse with --model-name-or-path {init_dir})")
    else:
        model_kwargs = {
            "torch_dtype": model_dtype,
            "use_cache": not config.gradient_checkpointing,
        }
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation

        logger.info(f"Loading model: {config.model_name_or_path} ({config.dtype})")
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = model.config.vocab_size

    # Apply LoRA
    assert config.lora_dropout == 0.0, (
        f"lora_dropout must be 0 for DP training (got {config.lora_dropout}). "
        "Dropout adds variance to per-sample gradients."
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    return model, tokenizer, vocab_size


# ---------------------------------------------------------------------------
# Opacus validation (Phase 1.5)
# ---------------------------------------------------------------------------

def validate_opacus_compatibility(model, config: Config, tokenizer, vocab_size: int) -> bool:
    """Phase 1.5: Check that the model works with Opacus GradSampleModule.

    Tests:
    1. ModuleValidator.validate() — checks all modules are compatible
    2. Per-sample gradient computation with the real compute_loss function
    3. Memory overhead measurement
    """
    from opacus.validators import ModuleValidator
    from opacus import PrivacyEngine

    logger.info("=" * 60)
    logger.info("Phase 1.5: Opacus Model Validation")
    logger.info("=" * 60)

    # Step 1: Module validation
    logger.info("\n[1/3] Running ModuleValidator.validate()...")
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"Found {len(errors)} incompatible modules:")
        for err in errors:
            logger.warning(f"  - {err}")
        logger.info("Attempting ModuleValidator.fix()...")
        model = ModuleValidator.fix(model)
        errors_after = ModuleValidator.validate(model, strict=False)
        if errors_after:
            logger.error(f"Still {len(errors_after)} errors after fix:")
            for err in errors_after:
                logger.error(f"  - {err}")
            return False
        logger.info("All modules fixed successfully.")
    else:
        logger.info("All modules passed validation.")

    # Step 2: Test per-sample gradient computation with real loss function
    logger.info("\n[2/3] Testing per-sample gradient computation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    test_texts = [
        "Patient admitted with chest pain. Vitals stable. Assessment and Plan: "
        "Continue monitoring. Order CBC, BMP. Consult cardiology.",
        "Follow-up visit for diabetes management. HbA1c improved to 7.2%.",
        "Emergency admission for acute kidney injury. Creatinine elevated to 3.4.",
    ]
    test_batch_size = len(test_texts)
    test_encodings = tokenizer(
        test_texts,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    test_dataset = torch.utils.data.TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
    )
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    # Match training path: only trainable params in optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )

    privacy_engine = PrivacyEngine()
    try:
        model_wrapped, optimizer_wrapped, test_loader_wrapped = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=test_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
    except Exception as e:
        logger.error(f"PrivacyEngine.make_private() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run forward-backward with the real loss function
    try:
        for batch_tuple in test_loader_wrapped:
            input_ids, attention_mask = batch_tuple
            batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            loss, _ = compute_loss(model_wrapped, batch, vocab_size, config.dp_loss_aggregation)
            loss.backward()
            logger.info(f"  Loss on test batch: {loss.item():.4f}")
            break

        # Check that per-sample gradients exist
        grad_sample_count = 0
        for name, param in model_wrapped.named_parameters():
            if param.requires_grad and hasattr(param, "grad_sample"):
                if param.grad_sample is not None:
                    grad_sample_count += 1
                    if grad_sample_count <= 3:
                        logger.info(
                            f"  {name}: grad_sample shape = {list(param.grad_sample.shape)}"
                        )

        if grad_sample_count > 0:
            logger.info(f"Per-sample gradients found on {grad_sample_count} parameters.")
        else:
            logger.warning("No grad_sample found on any parameter — check Opacus wrapping.")

    except Exception as e:
        logger.error(f"Forward-backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Memory overhead
    logger.info("\n[3/3] Memory overhead...")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable params: {trainable_params:,}")
    logger.info(
        f"  Estimated per-sample grad memory at batch_size={config.per_device_batch_size}: "
        f"{config.per_device_batch_size * trainable_params * 4 / 1e9:.2f} GB"
    )
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        logger.info(f"  Peak GPU memory: {mem_allocated:.2f} GB "
                     f"(batch_size={test_batch_size}, seq_len=128)")

    logger.info("\n" + "=" * 60)
    logger.info("Validation PASSED")
    logger.info("=" * 60)
    return True


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model, optimizer, lr_scheduler, config: Config, global_step: int,
    epoch: int, privacy_engine=None, final: bool = False,
):
    """Save model, optimizer, scheduler, and DP accountant state."""
    tag = "final" if final else f"step_{global_step}"
    ckpt_dir = Path(config.output_dir) / f"checkpoint-{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter weights
    model_to_save = model
    # Unwrap GradSampleModule if present (Opacus wrapping)
    if hasattr(model, "_module"):
        model_to_save = model._module
    model_to_save.save_pretrained(str(ckpt_dir / "adapter"))

    # Save optimizer + scheduler
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
            "global_step": global_step,
            "epoch": epoch,
        },
        ckpt_dir / "training_state.pt",
    )

    # Save DP accountant state
    if privacy_engine is not None:
        try:
            current_eps = privacy_engine.get_epsilon(config.dp_delta)
        except Exception:
            current_eps = float("inf")  # sigma=0 → infinite epsilon
        dp_state = {
            "accountant_history": privacy_engine.accountant.history,
            "noise_multiplier": optimizer.noise_multiplier,
            "current_epsilon": current_eps,
            "steps_completed": global_step,
        }
        with open(ckpt_dir / "dp_state.json", "w") as f:
            json.dump(dp_state, f, indent=2, default=str)

    # Save config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info(f"Checkpoint saved: {ckpt_dir}")
    if privacy_engine:
        logger.info(f"  epsilon = {dp_state['current_epsilon']:.4f}")


def load_checkpoint(model, optimizer, lr_scheduler, config: Config, privacy_engine=None):
    """Load checkpoint and restore training state. Returns (global_step, epoch)."""
    ckpt_dir = Path(config.resume_from)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    # Load LoRA weights
    adapter_dir = ckpt_dir / "adapter"
    if adapter_dir.exists():
        logger.info(f"Loading adapter from {adapter_dir}")
        model_to_load = model._module if hasattr(model, "_module") else model
        model_to_load.load_adapter(str(adapter_dir), adapter_name="default")

    # Load training state
    state_path = ckpt_dir / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if lr_scheduler and state.get("scheduler_state_dict"):
            lr_scheduler.load_state_dict(state["scheduler_state_dict"])
        global_step = state["global_step"]
        epoch = state["epoch"]
    else:
        global_step, epoch = 0, 0

    # Restore DP accountant
    if privacy_engine is not None:
        dp_state_path = ckpt_dir / "dp_state.json"
        if dp_state_path.exists():
            with open(dp_state_path) as f:
                dp_state = json.load(f)
            privacy_engine.accountant.history = [
                tuple(h) for h in dp_state["accountant_history"]
            ]
            logger.info(
                f"Restored DP accountant: "
                f"epsilon={dp_state['current_epsilon']:.4f}, "
                f"steps={dp_state['steps_completed']}"
            )

    logger.info(f"Resumed from {ckpt_dir} at step {global_step}, epoch {epoch}")
    return global_step, epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_grad_norm(model) -> float:
    """Compute the total gradient norm across all trainable parameters.

    This is the definitive metric for matching non_dp vs dp_s0_cinf:
    if grad_norms match, the optimizer sees the same update regardless
    of BF16 loss drift through different autograd graphs.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm_sq += p.grad.float().norm().item() ** 2
    return total_norm_sq ** 0.5


def _did_optimizer_step(optimizer) -> bool:
    """Check if the last optimizer.step() was a real step.

    With Opacus BatchMemoryManager, optimizer.step() is a no-op for
    sub-batch accumulation steps. Only the final sub-batch triggers
    a real parameter update. We detect this via the Opacus-internal
    _is_last_step_skipped flag.
    """
    if hasattr(optimizer, "_is_last_step_skipped"):
        return not optimizer._is_last_step_skipped
    return True  # Non-DP optimizer always steps


class _nullcontext:
    """Minimal context manager for non-DP dataloader iteration."""
    def __init__(self, enter_result):
        self.enter_result = enter_result
    def __enter__(self):
        return self.enter_result
    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: Config):
    """Main training function supporting baseline and DP modes."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_dp = config.mode == "dp"

    # Load model + tokenizer + LoRA
    model, tokenizer, vocab_size = load_model_and_tokenizer(config)

    # Validate-only mode (Phase 1.5)
    if config.mode == "validate":
        success = validate_opacus_compatibility(model, config, tokenizer, vocab_size)
        sys.exit(0 if success else 1)

    # Load data
    dataset, N = load_data(config, tokenizer)

    if is_dp:
        if config.dp_delta >= 1.0 / N:
            logger.warning(
                f"dp_delta={config.dp_delta} >= 1/N={1.0/N:.2e}. "
                f"Should be < 1/N for meaningful privacy guarantees."
            )

    # DataLoader
    # For clean experiment: precomputed Poisson batches give identical data
    # ordering across non_dp and dp runs (like DP-SGD Phase 0).
    if config.batch_indices_file:
        logger.info(f"Using precomputed batch indices from {config.batch_indices_file}")
        batch_sampler = PrecomputedBatchSampler(config.batch_indices_file)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
    elif is_dp:
        # Opacus replaces sampler with UniformWithReplacementSampler
        dataloader = DataLoader(
            dataset,
            batch_size=config.dp_target_batch_size,
            shuffle=True,
            num_workers=0,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.per_device_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    # DP wrapping (Phase 2a/2b)
    # IMPORTANT: Validate/fix model BEFORE creating optimizer, because fix()
    # may replace modules and create new parameters.
    privacy_engine = None
    if is_dp:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator

        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.info(f"Fixing {len(errors)} Opacus-incompatible modules...")
            model = ModuleValidator.fix(model)

    # Optimizer — created after fix() so parameter references are valid
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if is_dp:
        privacy_engine = PrivacyEngine()

        # When using precomputed batch indices, tell Opacus NOT to replace the
        # sampler. This ensures non_dp and dp see identical data (clean experiment).
        use_poisson = config.batch_indices_file is None

        if config.dp_noise_multiplier is not None:
            # Direct sigma mode — for clean experiments (sigma=0 → no noise)
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=config.dp_noise_multiplier,
                max_grad_norm=config.dp_max_grad_norm,
                poisson_sampling=use_poisson,
            )
        else:
            # Target epsilon mode — computes sigma automatically
            model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                epochs=config.num_epochs,
                target_epsilon=config.dp_epsilon,
                target_delta=config.dp_delta,
                max_grad_norm=config.dp_max_grad_norm,
                poisson_sampling=use_poisson,
            )

        sigma = optimizer.noise_multiplier
        logger.info("DP-SGD enabled:")
        if config.dp_noise_multiplier is not None:
            logger.info(f"  sigma (noise)  = {sigma:.4f} (set directly)")
        else:
            logger.info(f"  target epsilon = {config.dp_epsilon}")
            logger.info(f"  delta          = {config.dp_delta}")
            logger.info(f"  sigma (noise)  = {sigma:.4f} (computed from epsilon)")
        logger.info(f"  C (clip norm)  = {config.dp_max_grad_norm}")
        logger.info(f"  logical batch  = {config.dp_target_batch_size}")
        logger.info(f"  physical batch = {config.per_device_batch_size}")
        logger.info(f"  N (dataset)    = {N}")
        logger.info(f"  sample rate q  = {config.dp_target_batch_size / N:.4f}")

    model = model.to(device)
    model.train()

    # LR scheduler — total_steps counts OPTIMIZER steps (logical batches),
    # not physical micro-batches.
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * config.num_epochs
    if config.max_steps > 0:
        total_steps = config.max_steps

    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    # Resume from checkpoint
    global_step = 0
    start_epoch = 0
    if config.resume_from:
        global_step, start_epoch = load_checkpoint(
            model, optimizer, lr_scheduler, config, privacy_engine
        )

    # Output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(config.output_dir) / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Gradient accumulation (baseline only — DP uses BatchMemoryManager instead)
    # With precomputed batches: no accumulation, each Poisson batch = one step.
    # The loss is divided by expected_batch_size to match Opacus's /B normalization.
    using_precomputed = config.batch_indices_file is not None
    grad_accum = 1 if using_precomputed else (config.gradient_accumulation_steps if not is_dp else 1)
    expected_batch_size = config.dp_target_batch_size  # For loss normalization

    if not is_dp and grad_accum > 1:
        logger.info(f"  Gradient accumulation: {grad_accum} steps "
                     f"(effective batch = {config.per_device_batch_size * grad_accum})")
    if using_precomputed:
        logger.info(f"  Precomputed batches: each batch = one step, "
                     f"loss /= {expected_batch_size} (matching Opacus /B)")

    # For baseline steps_per_epoch, account for gradient accumulation
    if not is_dp and grad_accum > 1:
        steps_per_epoch = max(1, steps_per_epoch // grad_accum)
        total_steps = steps_per_epoch * config.num_epochs
        if config.max_steps > 0:
            total_steps = config.max_steps

    # Re-create scheduler with correct total_steps if it changed
    if not is_dp and grad_accum > 1:
        lr_scheduler = get_scheduler(
            config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * config.warmup_ratio),
            num_training_steps=total_steps,
        )

    logger.info(f"\nStarting training: {total_steps} optimizer steps, {config.num_epochs} epochs")
    logger.info(f"  Device: {device}")
    logger.info(f"  Mode: {config.mode}")
    logger.info(f"  Loss aggregation: {config.dp_loss_aggregation}")

    train_start = time.time()
    step_loss_accum = 0.0  # Mean-loss accumulator for logging (weighted by sample count)
    step_samples = 0
    step_tokens = 0
    step_start_time = time.time()
    accum_counter = 0  # Tracks gradient accumulation steps (baseline only)

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()

        if is_dp:
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            data_iter = BatchMemoryManager(
                data_loader=dataloader,
                max_physical_batch_size=config.per_device_batch_size,
                optimizer=optimizer,
            )
        else:
            data_iter = dataloader

        with data_iter if is_dp else _nullcontext(data_iter) as loader:
            for batch in loader:
                # For precomputed baseline, keep batch on CPU — micro-batch loop
                # moves slices to GPU. For all other modes, move full batch to GPU.
                if not (using_precomputed and not is_dp):
                    batch = {k: v.to(device) for k, v in batch.items()}

                loss, mean_loss_val = compute_loss(
                    model, batch, vocab_size, config.dp_loss_aggregation,
                ) if not (using_precomputed and not is_dp) else (None, None)

                if is_dp:
                    # Opacus: zero_grad + step every micro-batch (internal accumulation)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    is_boundary = _did_optimizer_step(optimizer)
                elif using_precomputed:
                    # Clean experiment baseline: full Poisson batch from DataLoader.
                    # Split into micro-batches (like BatchMemoryManager), accumulate
                    # gradients, step once. Avoids OOM on large Poisson batches.
                    optimizer.zero_grad()
                    full_ids = batch["input_ids"]
                    full_mask = batch["attention_mask"]
                    B_actual = full_ids.size(0)
                    phys = config.per_device_batch_size
                    _mb_loss_sum = 0.0
                    _mb_samples = 0
                    for mb_start in range(0, B_actual, phys):
                        mb_end = min(mb_start + phys, B_actual)
                        mb = {
                            "input_ids": full_ids[mb_start:mb_end].to(device),
                            "attention_mask": full_mask[mb_start:mb_end].to(device),
                        }
                        mb_loss, mb_mean = compute_loss(
                            model, mb, vocab_size, config.dp_loss_aggregation,
                        )
                        (mb_loss / expected_batch_size).backward()
                        _mb_loss_sum += mb_mean * (mb_end - mb_start)
                        _mb_samples += mb_end - mb_start
                    optimizer.step()
                    is_boundary = True
                    mean_loss_val = _mb_loss_sum / max(_mb_samples, 1)
                else:
                    # Standard baseline: gradient accumulation with fixed micro-batches
                    if accum_counter == 0:
                        optimizer.zero_grad()
                    (loss / grad_accum).backward()
                    accum_counter += 1
                    is_boundary = (accum_counter >= grad_accum)
                    if is_boundary:
                        optimizer.step()
                        accum_counter = 0

                # Accumulate stats (always use mean_loss_val for comparable logging)
                bs = batch["input_ids"].size(0)
                tokens_in_batch = batch["attention_mask"].sum().item()
                step_loss_accum += mean_loss_val * bs
                step_samples += bs
                step_tokens += tokens_in_batch

                if not is_boundary:
                    continue

                # --- Real optimizer step boundary ---
                # Compute grad_norm BEFORE lr_scheduler.step() (after optimizer.step())
                # This is the definitive metric for non_dp vs dp matching —
                # grad_norms should be identical even if loss drifts from BF16 rounding.
                grad_norm = _compute_grad_norm(model)

                lr_scheduler.step()
                global_step += 1

                step_time = time.time() - step_start_time
                avg_loss = step_loss_accum / max(step_samples, 1)

                # Logging
                if global_step % config.logging_steps == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    samples_per_sec = step_samples / max(step_time, 1e-6)
                    tokens_per_sec = step_tokens / max(step_time, 1e-6)
                    msg = (
                        f"Epoch {epoch+1}/{config.num_epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"grad_norm: {grad_norm:.3f} | "
                        f"LR: {lr:.2e} | "
                        f"{samples_per_sec:.0f} samples/s | "
                        f"{tokens_per_sec:.0f} tok/s"
                    )
                    if is_dp and privacy_engine:
                        if config.dp_noise_multiplier is not None and config.dp_noise_multiplier == 0:
                            msg += " | eps: inf (sigma=0)"
                        else:
                            eps = privacy_engine.get_epsilon(config.dp_delta)
                            msg += f" | eps: {eps:.2f}"
                    if torch.cuda.is_available():
                        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                        msg += f" | mem: {mem_gb:.1f}GB"
                    logger.info(msg)

                # Reset accumulators for next optimizer step
                step_loss_accum = 0.0
                step_samples = 0
                step_tokens = 0
                step_start_time = time.time()

                # Checkpointing
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, lr_scheduler, config,
                        global_step, epoch, privacy_engine,
                    )

                if config.max_steps > 0 and global_step >= config.max_steps:
                    break

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")

        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    # Final save
    total_time = time.time() - train_start
    logger.info(f"\nTraining completed in {total_time:.1f}s ({global_step} optimizer steps)")

    save_checkpoint(
        model, optimizer, lr_scheduler, config,
        global_step, config.num_epochs, privacy_engine, final=True,
    )

    if is_dp and privacy_engine:
        if config.dp_noise_multiplier is not None and config.dp_noise_multiplier == 0:
            logger.info("Final: sigma=0 (clean experiment, no privacy cost)")
        else:
            final_eps = privacy_engine.get_epsilon(config.dp_delta)
            logger.info(f"Final (epsilon, delta) = ({final_eps:.4f}, {config.dp_delta})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="DP-LoRA fine-tuning")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--mode", type=str, choices=["baseline", "validate", "dp"])

    # Model overrides
    parser.add_argument("--model-name-or-path", type=str, dest="model_name_or_path")
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--attn-implementation", type=str, dest="attn_implementation")

    # LoRA overrides
    parser.add_argument("--lora-r", type=int, dest="lora_r")
    parser.add_argument("--lora-alpha", type=int, dest="lora_alpha")

    # Data overrides
    parser.add_argument("--data-path", type=str, dest="data_path")
    parser.add_argument("--dataset-name", type=str, dest="dataset_name")
    parser.add_argument("--max-length", type=int, dest="max_length")

    # Training overrides
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--num-epochs", type=int, dest="num_epochs")
    parser.add_argument("--per-device-batch-size", type=int, dest="per_device_batch_size")
    parser.add_argument("--gradient-accumulation-steps", type=int, dest="gradient_accumulation_steps")
    parser.add_argument("--max-steps", type=int, dest="max_steps")
    parser.add_argument("--seed", type=int)

    # DP overrides
    parser.add_argument("--dp-epsilon", type=float, dest="dp_epsilon")
    parser.add_argument("--dp-delta", type=float, dest="dp_delta")
    parser.add_argument("--dp-max-grad-norm", type=float, dest="dp_max_grad_norm")
    parser.add_argument("--dp-target-batch-size", type=int, dest="dp_target_batch_size")
    parser.add_argument("--dp-noise-multiplier", type=float, dest="dp_noise_multiplier",
                        help="Set sigma directly (overrides --dp-epsilon). Use 0 for clean experiment.")
    parser.add_argument("--dp-loss-aggregation", type=str, dest="dp_loss_aggregation",
                        choices=["sum", "mean"],
                        help="Batch loss aggregation: sum (matches DP-SGD) or mean.")

    # Output overrides
    parser.add_argument("--output-dir", type=str, dest="output_dir")
    parser.add_argument("--logging-steps", type=int, dest="logging_steps")
    parser.add_argument("--save-steps", type=int, dest="save_steps")
    parser.add_argument("--resume-from", type=str, dest="resume_from")
    parser.add_argument("--batch-indices-file", type=str, dest="batch_indices_file",
                        help="Precomputed Poisson batch indices (JSON) for clean experiment.")

    args = parser.parse_args()

    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    config.update_from_args(args)

    return config, args


def main():
    config, args = parse_args()

    logger.info(f"Mode: {config.mode}")
    logger.info(f"Model: {config.model_name_or_path}")
    if config.mode == "dp":
        logger.info(f"DP: epsilon={config.dp_epsilon}, delta={config.dp_delta}, "
                     f"C={config.dp_max_grad_norm}, target_batch={config.dp_target_batch_size}")

    train(config)


if __name__ == "__main__":
    main()
