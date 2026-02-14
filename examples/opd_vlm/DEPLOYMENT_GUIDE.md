# Deployment Guide: Local → GitHub → Server with Wandb

This guide shows how to deploy your OPD-VLM code from your local Mac to a server with 8x H100 and track experiments with wandb.

## Step-by-Step Workflow

### Step 1: Prepare Code Locally (Already Done! ✅)

Your code is ready at: `/Users/yongyuanl/slime-opd/examples/opd_vlm/`

Files created:
- `opd_vlm_reward.py` - OPD reward function with VLM support
- `run_opd_vlm_geo3k.sh` - Training script
- `README.md`, `GPU_ALLOCATION.md`, etc. - Documentation

### Step 2: Push to GitHub

```bash
# On your Mac

cd /Users/yongyuanl/slime-opd

# Add all new files
git add examples/opd_vlm/

# Commit
git commit -m "Add OPD-VLM implementation with Qwen3-VL-Thinking models"

# Push to your fork
git push origin main
```

**Verify**: Check https://github.com/cheryyunl/slime-opd to see your files

### Step 3: Clone on Server

```bash
# SSH to your server with 8x H100
ssh your-server

# Clone your fork
cd /root  # or wherever you want
git clone https://github.com/cheryyunl/slime-opd.git
cd slime-opd
```

### Step 4: Set Up Environment on Server

```bash
# Install slime (if not already installed)
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Set wandb API key (IMPORTANT!)
export WANDB_API_KEY=your_wandb_api_key_here

# Or add to ~/.bashrc for persistence:
echo 'export WANDB_API_KEY=your_wandb_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

**Get your wandb API key**:
1. Go to https://wandb.ai/settings
2. Scroll to "API keys"
3. Copy your key

### Step 5: Verify GPU Setup on Server

```bash
# Check GPUs
nvidia-smi

# Should show 8x H100 80GB
# Output should look like:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xxx      Driver Version: 535.xxx      CUDA Version: 12.x   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA H100 80GB... Off  | 00000000:XX:00.0 Off |                    0 |
# ...
# |   7  NVIDIA H100 80GB... Off  | 00000000:XX:00.0 Off |                    0 |
# +-----------------------------------------------------------------------------+
```

### Step 6: Run Training with Wandb

```bash
cd /root/slime-opd

# Set wandb key (if not in ~/.bashrc)
export WANDB_API_KEY=your_wandb_api_key_here

# Run training (defaults already set for 8 GPUs!)
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

**What happens**:
1. Script downloads models (if not present)
2. Downloads GEO3K dataset
3. Starts teacher server (GPUs 0-3)
4. Starts student training (GPUs 4-7)
5. **Logs to wandb automatically!**

### Step 7: Monitor Training on Wandb

#### During Training:

1. **Open wandb dashboard**: https://wandb.ai/your-username/slime-opd-vlm

2. **You'll see**:
   - Run name (auto-generated or custom)
   - Real-time metrics updating
   - System metrics (GPU usage, etc.)

#### Key Metrics to Watch:

| Metric | What it means | Expected trend |
|--------|---------------|----------------|
| `opd_kl_loss` | KL divergence student↔teacher | ↓ Decreasing |
| `train/loss` | Training loss | ↓ Decreasing |
| `eval/accuracy` | Test set accuracy | ↑ Increasing |
| `reward` | Task rewards (if hybrid) | → Stable or ↑ |
| `system/gpu_X_memory_used` | GPU memory usage | → Stable |
| `system/gpu_X_utilization` | GPU compute usage | → High (70-100%) |

### Step 8: Customize Wandb Settings (Optional)

Edit the script to customize wandb:

```bash
# On server, edit the script
nano examples/opd_vlm/run_opd_vlm_geo3k.sh

# Find WANDB_ARGS section (around line 300):
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-opd-vlm           # Change project name
   --wandb-group ${STUDENT_MODEL_LOWER}-opd-${TRAIN_BACKEND}  # Group name
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix           # Remove to add random suffix
)

# Add custom run name:
WANDB_ARGS+=(
   --wandb-run-name "my-experiment-v1"    # Custom run name
)
```

### Step 9: View Results in Wandb

#### Real-time during training:
```
https://wandb.ai/your-username/slime-opd-vlm/runs/run-id
```

#### After training:
- **Metrics tab**: See all metrics over time
- **System tab**: GPU/CPU/memory usage
- **Logs tab**: Console output
- **Files tab**: Saved model checkpoints
- **Overview tab**: Summary statistics

#### Compare experiments:
1. Go to project page: `https://wandb.ai/your-username/slime-opd-vlm`
2. Select multiple runs
3. Click "Compare" or use the parallel coordinates plot

## Advanced Wandb Features

### 1. Add Custom Metrics

Edit `opd_vlm_reward.py` to log custom metrics:

```python
# After computing rewards
import wandb

wandb.log({
    "custom/teacher_confidence": teacher_confidence,
    "custom/student_entropy": student_entropy,
})
```

### 2. Log Model Checkpoints

Wandb automatically logs checkpoints if enabled in slime config.

### 3. Sweep for Hyperparameter Tuning

Create `sweep_config.yaml`:

```yaml
program: examples/opd_vlm/run_opd_vlm_geo3k.sh
method: grid
parameters:
  SLIME_SCRIPT_OPD_KL_COEF:
    values: [0.5, 1.0, 2.0]
```

Run sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent your-username/slime-opd-vlm/sweep-id
```

## Troubleshooting

### Issue: "wandb: ERROR api_key not configured"

**Solution**:
```bash
# Set the key
export WANDB_API_KEY=your_key_here

# Or login interactively
wandb login
```

### Issue: "wandb: Network error"

**Solution**:
```bash
# Check internet connection
ping wandb.ai

# If behind proxy, set:
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### Issue: No metrics showing in wandb

**Solution**:
```bash
# Check script has --use-wandb flag
grep "use-wandb" examples/opd_vlm/run_opd_vlm_geo3k.sh

# Check wandb key is set
echo $WANDB_API_KEY

# Check logs for wandb errors
tail -f /tmp/teacher_sglang_*.log
```

### Issue: Want to run offline (no internet)

**Solution**:
```bash
# Run in offline mode
export WANDB_MODE=offline

bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# Sync later when online
wandb sync wandb/run-xxx
```

## Complete Example Session

```bash
# 1. SSH to server
ssh your-h100-server

# 2. Clone repo (first time only)
cd /root
git clone https://github.com/cheryyunl/slime-opd.git
cd slime-opd

# 3. Set up environment (first time only)
pip install -e .
export WANDB_API_KEY=your_key_here

# 4. Run training
bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# 5. Monitor progress
# Open: https://wandb.ai/your-username/slime-opd-vlm
# Watch metrics update in real-time!

# 6. After training completes
# Check wandb for:
# - Final accuracy
# - Training curves
# - GPU utilization
# - Model checkpoints
```

## Wandb Dashboard Tour

### Main Views:

1. **Workspace** (Project view):
   - See all runs
   - Compare experiments
   - Filter by tags/groups

2. **Run Page**:
   - **Overview**: Summary stats, config
   - **Charts**: Custom plots
   - **System**: Hardware metrics
   - **Logs**: Console output
   - **Files**: Artifacts, checkpoints
   - **Notes**: Add experiment notes

3. **Reports**:
   - Create reports with visualizations
   - Share with team
   - Document findings

### Useful Charts to Create:

1. **OPD KL Loss over time**
   - X-axis: Step
   - Y-axis: opd_kl_loss
   - Shows distillation progress

2. **Accuracy improvement**
   - X-axis: Step
   - Y-axis: eval/accuracy
   - Shows performance gains

3. **GPU utilization**
   - X-axis: Time
   - Y-axis: system.gpu.*.utilization
   - Check hardware efficiency

4. **KL vs Accuracy correlation**
   - X-axis: opd_kl_loss
   - Y-axis: eval/accuracy
   - Shows if distillation helps performance

## Tips for Effective Experiment Tracking

1. **Use descriptive run names**:
   ```bash
   --wandb-run-name "opd-4b-student-kl1.0-v1"
   ```

2. **Add tags**:
   ```bash
   --wandb-tags "experiment,baseline,4b-student"
   ```

3. **Group related runs**:
   ```bash
   --wandb-group "kl-coefficient-sweep"
   ```

4. **Take notes**:
   - Add notes in wandb UI about what worked
   - Document hyperparameter choices
   - Record any issues encountered

5. **Compare systematically**:
   - Keep one variable changing at a time
   - Use same random seeds for fair comparison
   - Document all changes in run config

## Quick Reference

### Essential Commands:

```bash
# Set wandb key
export WANDB_API_KEY=xxx

# Run with default wandb settings
bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# Run without wandb
export WANDB_MODE=disabled
bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# Custom wandb project
export WANDB_PROJECT=my-custom-project
bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

### Wandb URLs:

- Your projects: `https://wandb.ai/your-username`
- This project: `https://wandb.ai/your-username/slime-opd-vlm`
- Specific run: `https://wandb.ai/your-username/slime-opd-vlm/runs/run-id`
- API keys: `https://wandb.ai/settings`

---

**You're all set to run experiments and track them with wandb!** 🚀
