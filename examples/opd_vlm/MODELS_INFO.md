# Model Configuration for OPD-VLM

## Default Models (Updated for -Thinking)

### Teacher: Qwen3-VL-30B-A3B-Thinking
- **Size**: 30B parameters (Mixture-of-Experts)
- **Architecture**:
  - 128 experts
  - Top-8 routing per token
  - 48 layers
  - 2048 hidden size
- **Special Features**:
  - Built-in Chain-of-Thought reasoning
  - Outputs detailed step-by-step solutions
  - No special prompting needed
- **Config File**: `scripts/models/qwen3-30B-A3B.sh`
- **Download**: `Qwen/Qwen3-VL-30B-A3B-Thinking`

### Student: Qwen3-VL-4B-Thinking
- **Size**: 4B parameters (Dense model)
- **Architecture**:
  - 36 layers
  - 2560 hidden size
  - 32 attention heads
- **Special Features**:
  - Same -Thinking behavior as teacher
  - Naturally outputs reasoning steps
  - 7.5x smaller than teacher
- **Config File**: `scripts/models/qwen3-4B.sh`
- **Download**: `Qwen/Qwen3-VL-4B-Thinking`

## -Thinking vs -Instruct Models

| Aspect | -Instruct | -Thinking |
|--------|-----------|-----------|
| **Response Style** | Direct answer | Step-by-step reasoning + answer |
| **Typical Length** | 5-20 tokens | 50-200+ tokens |
| **CoT Prompting** | Required for detailed responses | Built-in, automatic |
| **Best For** | Quick inference | Training/distillation |
| **OPD Benefit** | Less signal | More signal (more tokens) |

## Example Outputs on GEO3K

### Problem:
*[Image of triangle with base=15, height=36]*
"What is the area of this triangle?"

### -Instruct Model Response:
```
Answer: \boxed{270}
```
*~3 tokens*

### -Thinking Model Response:
```
Let me solve this step by step.

I can see a triangle in the image with:
- Base (b) = 15 units
- Height (h) = 36 units

The formula for the area of a triangle is:
A = (1/2) × base × height

Substituting the values:
A = (1/2) × 15 × 36
A = 7.5 × 36
A = 270

Therefore, the area of the triangle is 270 square units.

Answer: \boxed{270}
```
*~100 tokens*

## Why -Thinking Models Are Better for OPD

### 1. More Distillation Signal
- **100+ tokens** vs 3 tokens per sample
- Student learns from teacher's full reasoning process
- Better gradient estimates from more tokens

### 2. Reasoning Transfer
- Student learns **how** teacher solves problems, not just answers
- Transfers problem-solving approach
- Better generalization to new problems

### 3. Natural CoT
- No prompt engineering needed
- Consistent reasoning format
- Works out-of-the-box

### 4. Measured Benefits (Expected)
Based on text OPD results:
- **Without CoT**: 76% → 94% (+18%)
- **With -Thinking models**: Expect similar or better improvement
- More tokens should lead to stronger distillation

## Model Compatibility

### Supported -Thinking Models in slime:
- ✅ Qwen3-VL-2B-Thinking
- ✅ Qwen3-VL-4B-Thinking (default student)
- ✅ Qwen3-VL-8B-Thinking
- ✅ Qwen3-VL-30B-A3B-Thinking (default teacher)
- ✅ Qwen3-VL-235B-A22B-Thinking

All models use the same config as their -Instruct counterparts.
The `-Thinking` suffix is automatically stripped when loading configs.

## GPU Requirements

### Qwen3-VL-30B-A3B-Thinking (Teacher)
- **Recommended**: 8xH100 (80GB each) with TP=8
- **Minimum**: 4xH100 with TP=4
- **Memory**: ~40-50GB model weights + KV cache

### Qwen3-VL-4B-Thinking (Student Training)
- **Recommended**: 8xH100 for training
- **Minimum**: 4xH100
- **Memory**: Training needs 3-4x model size (gradients, optimizer states)

## Configuration in Scripts

Models are configured via environment variables:

```bash
# Set student model
export SLIME_SCRIPT_STUDENT_MODEL="Qwen3-VL-4B-Thinking"

# Set teacher model
export SLIME_SCRIPT_TEACHER_MODEL="Qwen3-VL-30B-A3B-Thinking"

# Run training
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

The script automatically:
1. Strips `-Thinking` suffix
2. Maps to correct config file (e.g., `qwen3-30B-A3B.sh`)
3. Sets VL-specific args (rotary-base=5000000)
4. Loads model from HuggingFace

## No Special Prompting Needed!

Unlike -Instruct models where you might add:
```python
system_prompt = "Think step-by-step before answering."
```

With -Thinking models:
```python
# No special prompt needed!
# The model naturally outputs CoT reasoning
```

This makes OPD setup simpler and more robust.
