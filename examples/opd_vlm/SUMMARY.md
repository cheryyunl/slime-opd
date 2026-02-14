# OPD-VLM Implementation Summary

## What Was Created

I've set up **On-Policy Distillation for Vision-Language Models** in your forked repo at `/Users/yongyuanl/slime-opd/examples/opd_vlm/`

### Files Created:

```
examples/opd_vlm/
├── README.md                   # Detailed documentation
├── SUMMARY.md                  # This file
├── __init__.py                 # Python module init
├── opd_vlm_reward.py          # OPD reward function with VLM support
└── run_opd_vlm_geo3k.sh       # Training script (executable)
```

## What Changed from Original Code

### ✅ No Data Processing Changes
- Uses existing GEO3K VLM dataset handling
- Uses existing image encoding functions
- Uses existing Sample structure with multimodal_inputs

### ✅ Only OPD-Specific Additions

**1. opd_vlm_reward.py** (~100 lines)
- `reward_func`: Calls teacher model with images
- `post_process_rewards`: Extracts teacher log-probs

**Key modification** (line 28-34):
```python
# VLM-specific: Add images to payload if present
if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
    from slime.utils.processing_utils import encode_image_for_rollout_engine
    payload["image_data"] = [
        encode_image_for_rollout_engine(img)
        for img in sample.multimodal_inputs["images"]
    ]
```

**2. run_opd_vlm_geo3k.sh** (~300 lines)
- Based on `examples/geo3k_vlm/run_geo3k_vlm.sh`
- Added teacher server startup
- Added OPD arguments
- Added GPU allocation for teacher/student

## Answers to Your Questions

### ❌ Multi-turn Complexity
**You were right!** Multi-turn needs `env_geo3k.py` environment - too complex for starting.

**Solution**: Used single-turn GEO3K (`examples/geo3k_vlm`) instead.

### ❓ Does it need CoT (Chain-of-Thought)?

**Short answer**: Not required, but helpful.

**Without CoT** (current implementation):
- Answers: "Answer: \boxed{270}" (~5-10 tokens)
- ✅ Simple to start
- ⚠️ Less distillation signal

**With CoT** (optional enhancement):
- Answers: "Let me solve... Area = 0.5 × 15 × 36 = 270. Answer: \boxed{270}" (~30-50 tokens)
- ✅ More tokens to learn from
- ⚠️ Requires prompt modification

**Recommendation**: Start without CoT, add later if needed.

### ✅ What Needs Revision?

**Nothing in data processing!** Just:
1. Set environment variables (see Quick Start below)
2. Run the script
3. (Optional) Tune hyperparameters

## Quick Start

### 1. Basic Run (16 GPUs recommended)

```bash
cd /Users/yongyuanl/slime-opd

# Set wandb key (optional)
export WANDB_API_KEY=your_key

# Run with defaults:
# Student: Qwen3-VL-8B, Teacher: Qwen3-VL-32B
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

### 2. For 8 GPUs

```bash
SLIME_SCRIPT_NUM_GPUS=8 \
SLIME_SCRIPT_NUM_GPUS_TEACHER=4 \
SLIME_SCRIPT_NUM_GPUS_STUDENT=4 \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

### 3. Customize Models

```bash
# Smaller student
SLIME_SCRIPT_STUDENT_MODEL=Qwen3-VL-4B-Instruct \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# Different teacher
SLIME_SCRIPT_TEACHER_MODEL=Qwen3-VL-30B-A3B-Instruct \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

## What Happens When You Run It

1. **Downloads** (automatic):
   - Qwen3-VL-8B-Instruct (student)
   - Qwen3-VL-32B-Instruct (teacher)
   - GEO3K dataset

2. **Starts teacher server**:
   - Launches SGLang on 8 GPUs (or 4 if using 8 total)
   - Waits for server to be ready
   - Port: 13141

3. **Trains student with OPD**:
   - Student generates responses
   - Teacher evaluates with images
   - Student learns to match teacher's distribution

4. **Evaluates periodically**:
   - Every 20 iterations on test set
   - Logs to wandb/tensorboard

## Expected Results

Based on text OPD results (76% → 94%), expect:

```
Baseline (Qwen3VL-8B):           65-70%
After OPD (teacher=32B):         80-85%
Teacher (Qwen3VL-32B):           88-90%
```

## Configuration Options

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SLIME_SCRIPT_STUDENT_MODEL` | `Qwen3-VL-8B-Instruct` | Student model |
| `SLIME_SCRIPT_TEACHER_MODEL` | `Qwen3-VL-32B-Instruct` | Teacher model |
| `SLIME_SCRIPT_NUM_GPUS` | `16` | Total GPUs |
| `SLIME_SCRIPT_NUM_GPUS_TEACHER` | `8` | GPUs for teacher |
| `SLIME_SCRIPT_NUM_GPUS_STUDENT` | `8` | GPUs for student |
| `SLIME_SCRIPT_OPD_KL_COEF` | `1.0` | OPD KL coefficient |
| `SLIME_SCRIPT_TRAIN_BACKEND` | `megatron` | Backend (megatron/fsdp) |
| `SLIME_SCRIPT_DATASET_NAME` | `chenhegu/geo3k_imgurl` | Dataset |

## How to Add CoT Later

If you want longer responses for better distillation:

**Option 1**: Modify dataset preprocessing
```python
# Add reasoning to answers
answer = f"Let me solve step by step.\n{reasoning}\nAnswer: \\boxed{{{value}}}"
```

**Option 2**: Add to prompt (in run script)
```bash
ROLLOUT_ARGS+=(
   --system-prompt "Think step-by-step before answering."
)
```

## Monitoring Training

### Wandb Metrics
- `opd_kl_loss`: Should decrease (student matching teacher better)
- `eval/accuracy`: Should increase
- `reward`: Task performance (if using hybrid)

### Logs
- Teacher server: `/tmp/teacher_sglang_*.log`
- Training: Ray dashboard at http://localhost:8265

## Troubleshooting

### Teacher won't start
```bash
# Check log
tail -f /tmp/teacher_sglang_*.log

# Common fixes:
# 1. OOM: Reduce teacher GPUs or use smaller teacher
# 2. Port conflict: Change TEACHER_PORT in script
```

### Training fails
```bash
# 1. Verify teacher is running
curl http://localhost:13141/health_generate

# 2. Check Ray dashboard
open http://localhost:8265

# 3. Check GPU allocation
nvidia-smi
```

## Next Steps

1. **Run basic OPD** (start simple)
   ```bash
   bash examples/opd_vlm/run_opd_vlm_geo3k.sh
   ```

2. **Monitor results** (check wandb)
   - Is opd_kl_loss decreasing?
   - Is eval/accuracy increasing?

3. **Optimize** (if needed)
   - Adjust `OPD_KL_COEF`
   - Add CoT prompting
   - Try hybrid rewards

4. **Scale up** (if working well)
   - Try multi-turn OPD
   - Try different datasets
   - Experiment with model sizes

## Key Insight

**The beauty of this implementation**: You don't need to change ANY data processing code! The VLM infrastructure in slime already handles images. We just added:
1. Teacher server startup (~30 lines)
2. OPD reward function with image support (~15 lines modified)
3. Training script modifications (~50 lines added)

Everything else (dataset loading, image encoding, rollout) already works! 🎉
