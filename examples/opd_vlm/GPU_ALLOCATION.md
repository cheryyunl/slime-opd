# GPU Allocation Guide for OPD-VLM

## Quick Answer: 8x H100 80GB ✅ WORKS PERFECTLY!

Your 8x H100 80GB setup is **ideal** for OPD with:
- Teacher: Qwen3-VL-30B-A3B-Thinking
- Student: Qwen3-VL-4B-Thinking

## Recommended Configuration (8 GPUs)

### GPU Assignment:
```
GPUs 0-3: Teacher (Qwen3-VL-30B-A3B-Thinking, TP=4)
GPUs 4-7: Student (Qwen3-VL-4B-Thinking)
          ├─ GPUs 4-5: Training (TP=2)
          └─ GPUs 6-7: Rollout (2 engines)
```

### Memory Usage Breakdown:

**Teacher (30B-A3B MoE on 4x H100)**:
- Model weights: ~35-40GB (MoE with 128 experts)
- Per GPU (TP=4): ~10GB weights + ~15GB KV cache = ~25GB/GPU
- **Total per GPU: ~25-30GB** (well under 80GB!)
- ✅ Fits comfortably with room for batching

**Student Training (4B on 2x H100)**:
- Model weights: ~8GB
- Gradients: ~8GB
- Optimizer states: ~16GB
- Activations: ~10-20GB
- Per GPU (TP=2): ~20-25GB/GPU
- **Total per GPU: ~30-35GB** (comfortable!)

**Student Rollout (4B on 2x H100)**:
- Model weights: ~8GB
- KV cache: ~10-15GB per GPU
- **Total per GPU: ~20-25GB** (plenty of room!)

### Why This Works:

1. **30B-A3B is MoE**:
   - 128 experts, but only activates **8 experts per token**
   - Effective params per forward pass: ~3-4B
   - More memory efficient than dense 30B

2. **4B Student is Small**:
   - Much smaller than typical 8B models
   - Training overhead manageable on 2 GPUs
   - Rollout is very lightweight

3. **H100 80GB is Generous**:
   - Way more memory than needed
   - Can increase batch sizes for faster training
   - Can handle larger context lengths

## Comparison: Original Text OPD vs VLM OPD

### Original Text OPD (8 GPUs):
```
GPU 7:     Teacher (Qwen3-32B Dense, TP=1) ~60GB
GPUs 0-1:  Student Training (Qwen3-8B, TP=2)
GPUs 2-5:  Student Rollout (4 engines)
GPU 6:     Unused
```

### VLM OPD (8 GPUs) - Your Setup:
```
GPUs 0-3:  Teacher (Qwen3-VL-30B-A3B MoE, TP=4) ~25-30GB/GPU
GPUs 4-5:  Student Training (Qwen3-VL-4B, TP=2) ~30-35GB/GPU
GPUs 6-7:  Student Rollout (2 engines) ~20-25GB/GPU
```

**Key differences**:
- ✅ VLM uses all 8 GPUs efficiently
- ✅ More balanced distribution
- ✅ Better memory utilization
- ✅ MoE teacher is more efficient than dense

## Alternative Configurations

### Config 1: Maximum Teacher Performance (8 GPUs)
```bash
# Better teacher throughput, slightly slower student training
SLIME_SCRIPT_NUM_GPUS_TEACHER=6 \
SLIME_SCRIPT_NUM_GPUS_STUDENT=2 \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

**Use when**: Teacher is the bottleneck (slower generation)

### Config 2: Maximum Student Training (8 GPUs)
```bash
# Faster student training, might bottleneck on teacher
SLIME_SCRIPT_NUM_GPUS_TEACHER=3 \
SLIME_SCRIPT_NUM_GPUS_STUDENT=5 \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

**Use when**: Training is slow, teacher is fast enough

### Config 3: Balanced (Default - 8 GPUs)
```bash
# Best balance for most cases
SLIME_SCRIPT_NUM_GPUS_TEACHER=4 \
SLIME_SCRIPT_NUM_GPUS_STUDENT=4 \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

**Use when**: Starting out, not sure which is bottleneck

## If You Have 16 GPUs

### Optimal 16 GPU Configuration:
```
GPUs 0-7:   Teacher (Qwen3-VL-30B-A3B-Thinking, TP=8)
            - Better throughput
            - Faster rollout generation
            - Can serve larger batches

GPUs 8-15:  Student (Qwen3-VL-4B-Thinking)
            - More GPUs for training parallelism
            - Larger batch sizes
            - Faster convergence
```

```bash
# For 16 GPUs
SLIME_SCRIPT_NUM_GPUS=16 \
SLIME_SCRIPT_NUM_GPUS_TEACHER=8 \
SLIME_SCRIPT_NUM_GPUS_STUDENT=8 \
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

## Memory Estimates by Model Size

### Teacher Models (with TP):

| Model | Params | GPUs | TP | GB/GPU | Total GB |
|-------|--------|------|----|----|----------|
| Qwen3-VL-8B | 8B | 2 | 2 | 20-25 | 40-50 |
| Qwen3-VL-30B-A3B (MoE) | 30B | 4 | 4 | 25-30 | 100-120 |
| Qwen3-VL-30B-A3B (MoE) | 30B | 8 | 8 | 15-20 | 120-160 |
| Qwen3-VL-235B-A22B (MoE) | 235B | 8 | 8 | 40-50 | 320-400 |

### Student Models (Training):

| Model | Params | Training GPUs | TP | GB/GPU | Total GB |
|-------|--------|---------------|----|----|----------|
| Qwen3-VL-2B | 2B | 2 | 2 | 15-20 | 30-40 |
| Qwen3-VL-4B | 4B | 2 | 2 | 30-35 | 60-70 |
| Qwen3-VL-4B | 4B | 4 | 4 | 20-25 | 80-100 |
| Qwen3-VL-8B | 8B | 4 | 4 | 35-40 | 140-160 |

## Monitoring GPU Usage

### Check GPU allocation:
```bash
# Before starting
nvidia-smi

# During training - watch in real-time
watch -n 1 nvidia-smi

# Check specific GPU
nvidia-smi -i 0,1,2,3  # Teacher GPUs
nvidia-smi -i 4,5,6,7  # Student GPUs
```

### Expected utilization:
- **Teacher GPUs**: 40-60% utilization (inference workload)
- **Student Training GPUs**: 80-100% utilization (compute intensive)
- **Student Rollout GPUs**: 30-50% utilization (inference workload)

## Troubleshooting Memory Issues

### Teacher OOM (Out of Memory):
```bash
# Option 1: Reduce teacher TP (use fewer GPUs)
SLIME_SCRIPT_NUM_GPUS_TEACHER=6  # Instead of 4

# Option 2: Use smaller teacher
SLIME_SCRIPT_TEACHER_MODEL=Qwen3-VL-8B-Thinking

# Option 3: Reduce batch size (in teacher server)
# Edit run script: --chunked-prefill-size 2048  # Instead of 4096
```

### Student Training OOM:
```bash
# Option 1: Use more GPUs for training
# Edit run script: --actor-num-gpus-per-node 4  # Instead of 2

# Option 2: Reduce batch size
# Edit run script: --global-batch-size 256  # Instead of 512

# Option 3: Use smaller student
SLIME_SCRIPT_STUDENT_MODEL=Qwen3-VL-2B-Thinking
```

### Student Rollout OOM:
```bash
# Option 1: Reduce rollout batch size
# Edit run script: --rollout-batch-size 32  # Instead of 64

# Option 2: Reduce max response length
# Edit run script: --rollout-max-response-len 2048  # Instead of 4096
```

## Performance Optimization

### For 8x H100 80GB:

Since you have **plenty of memory**, you can:

1. **Increase batch sizes** for faster training:
   ```bash
   # In run script, change:
   --global-batch-size 1024      # Instead of 512
   --rollout-batch-size 128      # Instead of 64
   ```

2. **Increase context length** for longer responses:
   ```bash
   --rollout-max-response-len 8192  # Instead of 4096
   ```

3. **Increase teacher batch size**:
   ```bash
   # In teacher server startup:
   --chunked-prefill-size 8192  # Instead of 4096
   ```

## Summary: Your 8x H100 Setup

✅ **Perfect for**:
- Teacher: Qwen3-VL-30B-A3B-Thinking (4 GPUs)
- Student: Qwen3-VL-4B-Thinking (4 GPUs)

✅ **Expected usage**:
- ~25-35GB per GPU (well under 80GB!)
- ~60% of total capacity
- Plenty of headroom for optimization

✅ **Can handle**:
- Large batch sizes
- Long context lengths
- Multiple experiments in parallel

✅ **No issues with**:
- Memory constraints
- GPU allocation
- Model sizes

**You're all set to run OPD efficiently on 8x H100!** 🚀
