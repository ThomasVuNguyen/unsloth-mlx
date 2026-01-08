# Running the Minimal Fine-Tuning Sanity Check

This quick guide shows how to execute the bundled sanity-test script (`examples/10_minimal_local_finetune.py`) that fine-tunes the 135M SmolLM2 checkpoint on the toy `sample_train.jsonl` dataset. The steps assume you are in the project root (`unsloth-mlx/`).

## 1. Install Dependencies

Use the provided requirements inside a project-local virtual environment so dependencies stay isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

If Python warns about scripts not being on your `PATH`, add `/Library/Frameworks/Python.framework/Versions/3.11/bin` (or the equivalent for your installation) so tools such as `mlx_lm` are accessible. Remember to re-run `source .venv/bin/activate` in each new shell before training.

## 2. Run the Sample Training Script

```bash
python3 examples/10_minimal_local_finetune.py
```

What the script does:
- Loads `unsloth/SmolLM2-135M` (quantized 4-bit) with `FastLanguageModel`.
- Reads the bundled `sample_train.jsonl`, applies the auto-detected chat template, and prepares a tiny dataset of three sample interactions.
- Runs `SFTTrainer` for 100 steps (batch size 1, cosine schedule) on MLX native hardware acceleration.
- Saves LoRA adapters to `model/lora_adapters/` and the merged full model to `model/merged_model/`.

During execution you will see progress logs, validation losses, and periodic adapter checkpoints (every 25 steps) inside the `model/` directory.

## 3. Inspect Outputs (Optional)

- Inspect adapters: `ls model/lora_adapters`
- Inspect merged weights: `ls model/merged_model`
- Use `mlx_lm.generate --model model/merged_model` to manually verify generations.

That's it—after these steps you've verified the end-to-end fine-tuning pipeline on a 16 GB Apple Silicon machine.
