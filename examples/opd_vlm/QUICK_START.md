# Quick Start Checklist

## 🎯 What You Have

✅ **Code ready** at: `/Users/yongyuanl/slime-opd/examples/opd_vlm/`
✅ **Models configured**: Qwen3-VL-30B-A3B-Thinking (teacher) + Qwen3-VL-4B-Thinking (student)
✅ **Built-in CoT**: -Thinking models automatically output step-by-step reasoning
✅ **GPU allocation optimized**: 4 teacher + 4 student = 8 GPUs total
✅ **Wandb integration**: Already configured in the script

## 📋 Deployment Checklist

### On Your Mac (Local)

- [ ] **Push to GitHub**
  ```bash
  cd /Users/yongyuanl/slime-opd
  git add examples/opd_vlm/
  git commit -m "Add OPD-VLM with Thinking models"
  git push origin main
  ```

- [ ] **Verify on GitHub**
  - Visit: https://github.com/cheryyunl/slime-opd
  - Check `examples/opd_vlm/` folder is there

### On Your Server (8x H100)

- [ ] **Clone repo**
  ```bash
  ssh your-server
  cd /root
  git clone https://github.com/cheryyunl/slime-opd.git
  cd slime-opd
  ```

- [ ] **Install dependencies**
  ```bash
  pip install -e .
  pip install -r requirements.txt
  ```

- [ ] **Get wandb API key**
  - Visit: https://wandb.ai/settings
  - Copy your API key

- [ ] **Set wandb key**
  ```bash
  export WANDB_API_KEY=your_key_here
  # Or add to ~/.bashrc for persistence
  ```

- [ ] **Verify GPUs**
  ```bash
  nvidia-smi  # Should show 8x H100 80GB
  ```

- [ ] **Run training**
  ```bash
  bash examples/opd_vlm/run_opd_vlm_geo3k.sh
  ```

- [ ] **Monitor on wandb**
  - Open: https://wandb.ai/your-username/slime-opd-vlm
  - Watch metrics update in real-time!

## 🔑 Key Points

### 1. CoT is Already Included! ✅
-Thinking models have **built-in Chain-of-Thought**:
- No special prompting needed
- Automatically outputs reasoning steps
- ~100+ tokens per response (vs ~10 for -Instruct)

**Example output**:
```
Let me solve this step by step.

The triangle has:
- Base = 15 units
- Height = 36 units

Using the formula: Area = 1/2 × base × height
Area = 1/2 × 15 × 36 = 270

Answer: \boxed{270}
```

### 2. GPU Allocation Explained

**Original Text OPD (8 GPUs)**:
- 1 GPU: Teacher (32B text, fits on 1 GPU)
- 6 GPUs: Student
- Total: 7 used, 1 unused

**Your VLM OPD (8 GPUs)**:
- 4 GPUs: Teacher (30B-A3B VLM, needs more GPUs for vision)
- 4 GPUs: Student (4B VLM)
- Total: 8 used efficiently

**Why different?**
- VLM = Language model + Vision encoder (larger!)
- Vision processing needs more memory
- Both use 8 GPUs total, just different split

### 3. Wandb Already Configured ✅

Script automatically:
- Creates project: `slime-opd-vlm`
- Logs metrics: `opd_kl_loss`, `eval/accuracy`, etc.
- Tracks GPU usage
- Saves checkpoints

Just need to set `WANDB_API_KEY`!

## 📊 Expected Results

### Startup (5-10 min):
- Downloads models (~50GB total)
- Downloads GEO3K dataset (~500MB)
- Starts teacher server
- Begins training

### Training Progress (watch in wandb):
- `opd_kl_loss`: Should decrease ↓
- `eval/accuracy`: Should increase ↑
- GPU memory: ~30-40GB/GPU (plenty of room!)

### Final Results (expected):
```
Qwen3VL-4B baseline (SFT):        ~65%
Qwen3VL-4B + OPD (teacher=30B):   ~80-85%  ← Your goal!
Qwen3VL-30B teacher:              ~88-90%
```

## 🚨 Common Issues

### "wandb: ERROR api_key not configured"
```bash
export WANDB_API_KEY=your_key_here
# Or: wandb login
```

### "CUDA out of memory"
```bash
# Unlikely with H100 80GB!
# But if it happens, check GPU_ALLOCATION.md
```

### Teacher server won't start
```bash
# Check log
tail -f /tmp/teacher_sglang_*.log
```

## 📖 Documentation Files

Created for you:

1. **DEPLOYMENT_GUIDE.md** ← Read this for full workflow
2. **GPU_ALLOCATION.md** ← Detailed GPU info
3. **MODELS_INFO.md** ← About -Thinking models
4. **README.md** ← Full documentation
5. **READY_TO_RUN.md** ← What was created
6. **QUICK_START.md** ← This file

## ⚡ One-Line Quick Start

Once on server:

```bash
export WANDB_API_KEY=xxx && bash examples/opd_vlm/run_opd_vlm_geo3k.sh
```

That's it! Everything else is automatic.

## 🎓 Next Steps After First Run

1. **Check wandb** - See if metrics look good
2. **Compare** - Run baseline (no OPD) to compare
3. **Tune** - Adjust `OPD_KL_COEF` if needed
4. **Scale** - Try larger student (8B) or teacher (235B)

## 💡 Pro Tips

1. **Use tmux/screen** for long training runs:
   ```bash
   tmux new -s opd
   bash examples/opd_vlm/run_opd_vlm_geo3k.sh
   # Ctrl+B, then D to detach
   # tmux attach -t opd to reattach
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check logs**:
   ```bash
   tail -f /tmp/teacher_sglang_*.log
   ```

4. **Multiple experiments**:
   - Use different wandb run names
   - Tag experiments for easy comparison
   - Document what you change

---

**Ready to go! Push to GitHub, clone on server, and run!** 🚀
