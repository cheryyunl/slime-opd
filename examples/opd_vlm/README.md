# On-Policy Distillation for Vision-Language Models (OPD-VLM)

This example demonstrates **on-policy distillation (OPD)** for vision-language models using the GEO3K dataset.

A smaller student model (e.g., Qwen3-VL-8B) learns to imitate a larger teacher model (e.g., Qwen3-VL-32B) by:
1. Generating responses on its own (on-policy)
2. Getting token-level log-probabilities from the teacher
3. Minimizing KL divergence between student and teacher distributions

## Key Features

- ✅ **Single-turn VLM OPD**: Based on `examples/geo3k_vlm` (no complex environment needed)
- ✅ **Multimodal support**: Images automatically included in teacher evaluation
- ✅ **-Thinking models**: Use models with built-in Chain-of-Thought reasoning
- ✅ **Pure distillation or hybrid**: Can use OPD alone or combine with task rewards
- ✅ **Flexible GPU allocation**: Works with 8 or 16 GPUs

## About -Thinking Models

The default configuration uses **-Thinking** models which have several advantages for OPD:

### What are -Thinking Models?
- Models trained to output **step-by-step reasoning** before final answers
- **Built-in CoT** (Chain-of-Thought) - no special prompting needed
- Naturally produce longer, more detailed responses

### Example Output Comparison:

**-Instruct model** (short):
```
Answer: \boxed{270}
```

**-Thinking model** (detailed CoT):
```
Let me analyze this geometry problem step by step.

First, I need to identify the shape and relevant measurements:
- The figure shows a triangle
- Base = 15 units
- Height = 36 units

To find the area of a triangle, I'll use the formula:
Area = 1/2 × base × height

Calculating:
Area = 1/2 × 15 × 36
Area = 1/2 × 540
Area = 270

Therefore, the area is 270 square units.

Answer: \boxed{270}
```

### Why -Thinking Models Are Better for OPD:
1. **More tokens to distill** (~100+ vs ~10 tokens)
2. **Richer learning signal** - student learns reasoning process, not just answers
3. **Teacher's thought process** - student mimics how teacher thinks through problems
4. **Better generalization** - student learns problem-solving approach

### Teacher Model: Qwen3-VL-30B-A3B-Thinking
- **30B parameters** (MoE architecture: 128 experts, top-8 routing)
- **Built-in reasoning** - outputs detailed step-by-step solutions
- **Strong math performance** on GEO3K geometry problems

### Student Model: Qwen3-VL-4B-Thinking
- **4B parameters** (dense model)
- **Same reasoning style** as teacher (both -Thinking)
- **7.5x smaller** than teacher - significant compression!

## Quick Start

### Prerequisites

1. **Install slime** (see main README)
2. **Set up environment**:
   ```bash
   export WANDB_API_KEY=your_wandb_key  # Optional
   ```

### Basic Usage (16 GPUs Recommended)

```bash
cd /root/slime-opd

# Run with default settings:
# - Student: Qwen3-VL-4B-Thinking (naturally outputs CoT reasoning)
# - Teacher: Qwen3-VL-30B-A3B-Thinking (MoE model with built-in CoT)
# - 8 GPUs for teacher, 8 for student
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

### Custom Configuration

```bash
# Use different student model (e.g., dense 8B model)
SLIME_SCRIPT_STUDENT_MODEL=Qwen3-VL-8B-Instruct bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# Use FSDP backend instead of Megatron
SLIME_SCRIPT_TRAIN_BACKEND=fsdp bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# Adjust GPU allocation (for 8 GPUs total)
SLIME_SCRIPT_NUM_GPUS=8 \
SLIME_SCRIPT_NUM_GPUS_TEACHER=4 \
SLIME_SCRIPT_NUM_GPUS_STUDENT=4 \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# Adjust OPD KL coefficient
SLIME_SCRIPT_OPD_KL_COEF=0.5 bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

## What Gets Installed/Downloaded

The script automatically downloads:
1. **Student model**: e.g., `Qwen/Qwen3-VL-8B-Instruct` → `/root/models/`
2. **Teacher model**: e.g., `Qwen/Qwen3-VL-32B-Instruct` → `/root/models/`
3. **Dataset**: `chenhegu/geo3k_imgurl` → `/root/datasets/`

## How It Works

### 1. Teacher Server Startup

The script starts an SGLang server running the teacher model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
    --model-path Qwen3-VL-32B-Instruct \
    --tp 8 \
    --port 13141
```

### 2. Student Training with OPD

Training uses these key arguments:
```bash
--use-opd \
--opd-type sglang \
--opd-kl-coef 1.0 \
--rm-url http://localhost:13141/generate \
--custom-reward-function examples.opd_vlm.opd_vlm_reward.reward_func \
--custom-post-process-rewards examples.opd_vlm.opd_vlm_reward.post_process_rewards
```

### 3. OPD Workflow

```
┌─────────────────────────────────────────┐
│ 1. Student generates response          │
│    Input: [image] + "What is the area?"│
│    Output: "Answer: \boxed{270}"       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. Send to teacher for evaluation      │
│    Input: [image] + prompt + response   │
│    Output: token-level log-probs        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Extract response log-probs          │
│    Teacher log-probs → only response    │
│    Store in sample.teacher_log_probs    │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. Train with KL penalty               │
│    Loss = -advantages + λ*KL(S||T)      │
│    Student learns to match teacher      │
└─────────────────────────────────────────┘
```

## Files

- `opd_vlm_reward.py`: OPD reward function with VLM support (handles images)
- `run_opd_vlm_geo3k.sh`: Training script (teacher server + student training)
- `README.md`: This file

## Evaluation

### During Training

Metrics tracked in wandb/tensorboard:
- `opd_kl_loss`: KL divergence between student and teacher (should decrease)
- `reward`: Task performance if using hybrid OPD
- `eval/accuracy`: Accuracy on test set (evaluated every 20 iterations)

### After Training

Evaluate on held-out test set:
```bash
# The script automatically evaluates on test set during training
# Check wandb for eval/accuracy metrics
```

Expected improvement (based on text OPD results):
```
Baseline (Qwen3VL-8B SFT):        ~65%
+ OPD (teacher=32B):              ~80-85%
Teacher (Qwen3VL-32B):            ~88%
```

## Pure Distillation vs Hybrid

### Pure Distillation (Default)
Learning signal comes **only** from KL divergence with teacher:
```python
# In opd_vlm_reward.py
scalar_rewards = [0.0] * len(samples)  # No task reward
```

### Hybrid OPD + Task Rewards
Combine distillation with task performance:
```python
# In opd_vlm_reward.py (uncomment this section)
from slime.rollout.rm_hub.math_rm import compute_math_reward
task_rewards = [compute_math_reward(s) for s in samples]
scalar_rewards = task_rewards  # Or blend: 0.5*task + 0.5*kl
```

## Customization

### Use Different Dataset

Edit `DATASET_NAME` in the script:
```bash
DATASET_NAME="your-dataset-name"
# Dataset should have: images, problem (text), answer (label)
```

### Add Chain-of-Thought (CoT) Prompting

To encourage longer responses:

1. **Option A**: Modify dataset to include CoT in answers
```python
# Preprocess dataset
answer = "Let me solve step by step... Area = 0.5 × 15 × 36 = 270. Answer: \boxed{270}"
```

2. **Option B**: Add system prompt
```bash
# In ROLLOUT_ARGS, add:
--system-prompt "Think step-by-step before answering."
```

### Adjust OPD Strength

```bash
# Stronger distillation (student matches teacher more closely)
SLIME_SCRIPT_OPD_KL_COEF=2.0

# Weaker distillation (student has more freedom)
SLIME_SCRIPT_OPD_KL_COEF=0.5
```

## GPU Requirements

### Recommended: 16 GPUs
- **8 GPUs**: Teacher (Qwen3-VL-32B with TP=8)
- **8 GPUs**: Student training (Qwen3-VL-8B with TP=4)

### Minimum: 8 GPUs
- **4 GPUs**: Teacher (Qwen3-VL-32B with TP=4)
- **4 GPUs**: Student training (Qwen3-VL-8B with TP=2)

```bash
# For 8 GPUs
SLIME_SCRIPT_NUM_GPUS=8 \
SLIME_SCRIPT_NUM_GPUS_TEACHER=4 \
SLIME_SCRIPT_NUM_GPUS_STUDENT=4 \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

## Troubleshooting

### Teacher server fails to start
Check the log file:
```bash
tail -f /tmp/teacher_sglang_*.log
```

Common issues:
- Out of memory: Reduce teacher TP or use smaller teacher model
- Port conflict: Change `TEACHER_PORT` in script

### Student training fails
1. Check GPU visibility: Student uses GPUs 8-15 (or 4-7 for 8 GPU setup)
2. Check teacher server is running: `curl http://localhost:13141/health_generate`
3. Check logs in wandb or Ray dashboard (http://localhost:8265)

### Low distillation performance
1. Increase `OPD_KL_COEF` (default: 1.0)
2. Check teacher is actually being called (monitor teacher server logs)
3. Verify images are being sent correctly (check payload in logs)

## Next Steps

1. **Try multi-turn OPD**: Adapt `examples/geo3k_vlm_multi_turn` for OPD (longer sequences)
2. **Add CoT prompting**: Encourage longer, more detailed responses
3. **Experiment with hybrid rewards**: Balance task performance and distillation
4. **Try different datasets**: MathVista, ChartQA, etc.

## Differences from Text OPD

| Aspect | Text OPD | VLM OPD (This Example) |
|--------|----------|------------------------|
| Input | Text tokens only | Text tokens + images |
| Payload | `{"input_ids": [...]}` | `{"input_ids": [...], "image_data": [...]}` |
| Teacher model | Text LLM | Vision-Language Model |
| Dataset | Text problems | Multimodal (image + text) |
| Response length | Usually longer (with CoT) | Shorter (unless CoT added) |

The key modification is in `opd_vlm_reward.py:reward_func` which adds `image_data` to the payload when calling the teacher.

## References

- Original OPD paper: https://arxiv.org/abs/2306.13649
- slime OPD example: `examples/on_policy_distillation/`
- slime VLM example: `examples/geo3k_vlm/`
