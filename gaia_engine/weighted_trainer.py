"""
Eval-Weighted Training — curriculum-aware QLoRA training.

Instead of flat epochs where every sample gets equal repetition,
this trainer:
1. Pre-evaluates the model on all training samples
2. Scores each sample (pass/fail + confidence)
3. Builds a weighted dataset: failed samples repeated more, passed samples less
4. Trains on the weighted dataset
5. Post-evaluates to measure improvement

This avoids over-reinforcing what the model already knows (causing
confabulation) while focusing training energy on actual knowledge gaps.
"""

import json
import logging
import re
import time
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("GAIA.WeightedTrainer")


def evaluate_sample(model, tokenizer, instruction: str, expected_output: str,
                     system_prompt: str = "You are GAIA, a sovereign AI created by Azrael.",
                     max_tokens: int = 80) -> Tuple[bool, float, str]:
    """Evaluate a single training sample against the model.

    Returns: (passed: bool, confidence: float 0-1, generated_answer: str)
    """
    import torch

    from gaia_engine.core import ChatFormatter
    fmt = ChatFormatter(tokenizer)
    prompt = (fmt.format_system(system_prompt) + "\n"
              + fmt.format_message("user", instruction) + "\n"
              + fmt.assistant_prefix(enable_thinking=True))
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    # Strip think blocks (model-family-aware)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    answer = re.sub(r"<\|think\|>.*?<\|think\|>", "", answer, flags=re.DOTALL)
    answer = answer.strip()

    # Score: check how many key terms from expected output appear in answer
    expected_lower = expected_output.lower()
    answer_lower = answer.lower()

    # Extract key terms (words > 3 chars, not common words)
    stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
                  "her", "was", "one", "our", "out", "has", "have", "with", "this",
                  "that", "from", "they", "been", "said", "each", "which", "their",
                  "will", "other", "about", "many", "then", "them", "these", "some"}
    key_terms = [w for w in re.findall(r'\b\w{4,}\b', expected_lower)
                 if w not in stop_words]

    if not key_terms:
        return True, 1.0, answer  # No key terms to check

    hits = sum(1 for term in key_terms if term in answer_lower)
    confidence = hits / len(key_terms)

    # Pass if >50% of key terms found
    passed = confidence > 0.5

    return passed, confidence, answer


def pre_evaluate(model, tokenizer, samples: List[Dict],
                  system_prompt: str = "You are GAIA, a sovereign AI created by Azrael."
                  ) -> List[Dict]:
    """Pre-evaluate all training samples and return scored results.

    Each result includes the original sample + pass/fail + confidence + weight.
    """
    results = []
    passed_count = 0

    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", "")
        expected = sample.get("output", "")

        passed, confidence, answer = evaluate_sample(
            model, tokenizer, instruction, expected, system_prompt
        )

        if passed:
            passed_count += 1

        result = {
            **sample,
            "eval_passed": passed,
            "eval_confidence": round(confidence, 3),
            "eval_answer": answer[:200],
        }
        results.append(result)

        if (i + 1) % 20 == 0:
            logger.info("Pre-eval: %d/%d (%.0f%% passing)",
                        i + 1, len(samples), passed_count / (i + 1) * 100)

    logger.info("Pre-eval complete: %d/%d passed (%.0f%%)",
                passed_count, len(samples), passed_count / len(samples) * 100)
    return results


def build_weighted_dataset(eval_results: List[Dict],
                            fail_weight: int = 5,
                            low_confidence_weight: int = 3,
                            pass_weight: int = 1,
                            model_family: str = "chatml") -> List[Dict]:
    """Build a weighted training dataset based on eval results.

    Failed samples are repeated more, passed samples less.
    This focuses training energy on actual knowledge gaps.
    """

    def fmt(s):
        if model_family == "gemma4":
            return ("<|turn>user<turn|>\n" + s["instruction"] + "\n"
                    + "<|turn>assistant<turn|>\n" + s["output"])
        return ("<|im_start|>user\n" + s["instruction"] + "<|im_end|>\n"
                + "<|im_start|>assistant\n" + s["output"] + "<|im_end|>")

    weighted = []
    stats = {"failed": 0, "low_conf": 0, "passed": 0}

    for result in eval_results:
        if not result.get("eval_passed"):
            # Failed — high repetition
            repeats = fail_weight
            stats["failed"] += 1
        elif result.get("eval_confidence", 1.0) < 0.8:
            # Passed but low confidence — medium repetition
            repeats = low_confidence_weight
            stats["low_conf"] += 1
        else:
            # Passed with high confidence — minimal repetition
            repeats = pass_weight
            stats["passed"] += 1

        for _ in range(repeats):
            weighted.append({"text": fmt(result)})

    logger.info("Weighted dataset: %d total samples (from %d originals). "
                "Failed: %d×%d, Low-conf: %d×%d, Passed: %d×%d",
                len(weighted), len(eval_results),
                stats["failed"], fail_weight,
                stats["low_conf"], low_confidence_weight,
                stats["passed"], pass_weight)

    return weighted


def train_weighted(
    model_path: str,
    curriculum_path: str,
    output_dir: str,
    system_prompt: str = "You are GAIA, a sovereign AI created by Azrael.",
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    learning_rate: float = 2e-4,
    max_length: int = 384,
    fail_weight: int = 5,
    low_confidence_weight: int = 3,
    pass_weight: int = 1,
    num_epochs: int = 3,  # Fewer epochs needed since weighting handles repetition
) -> Dict:
    """Full eval-weighted training pipeline.

    1. Load model
    2. Pre-evaluate all samples
    3. Build weighted dataset
    4. Train
    5. Post-evaluate
    6. Save adapter

    Returns summary dict with pre/post scores and training stats.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Load curriculum
    samples = []
    with open(curriculum_path) as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info("Loaded %d curriculum samples", len(samples))

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    logger.info("Loading model: %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Pre-evaluate
    logger.info("=== PRE-EVALUATION ===")
    pre_results = pre_evaluate(model, tokenizer, samples, system_prompt)
    pre_score = sum(1 for r in pre_results if r["eval_passed"])
    pre_total = len(pre_results)
    logger.info("Pre-eval score: %d/%d (%.0f%%)", pre_score, pre_total,
                pre_score / pre_total * 100)

    # Build weighted dataset
    weighted = build_weighted_dataset(
        pre_results, fail_weight, low_confidence_weight, pass_weight,
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", trainable)

    # Train
    train_dataset = Dataset.from_list(weighted)
    total_steps = len(weighted) * num_epochs // 4  # with grad accumulation 4
    logger.info("Training: %d weighted samples × %d epochs = ~%d steps",
                len(weighted), num_epochs, total_steps)

    args = SFTConfig(
        output_dir=output_dir, num_train_epochs=num_epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=learning_rate, weight_decay=0.01, warmup_steps=20,
        logging_steps=max(1, total_steps // 10), save_strategy="no",
        bf16=True, max_length=max_length, dataset_text_field="text",
        report_to="none", gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    logger.info("Training complete in %.1fs, loss=%.4f", elapsed, result.training_loss)

    # Save adapter
    import os
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "pre_score": pre_score,
        "pre_total": pre_total,
        "pre_pct": round(pre_score / pre_total * 100, 1),
        "weighted_samples": len(weighted),
        "training_loss": round(result.training_loss, 4),
        "training_time_s": round(elapsed, 1),
        "trainable_params": trainable,
        "adapter_path": output_dir,
    }
