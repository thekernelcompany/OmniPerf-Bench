# Quick Start: TRAE Agent Integration

## One-Command Setup

```bash
# Clone and setup (replace with your actual repo URL)
git clone <your-omniperf-bench-repo>
cd OmniPerf-Bench
git checkout v1.0-trae-integration
./install_trae_integration.sh
```

## Set API Key

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Run TRAE Agent

```bash
# Activate environment
source bench-env/bin/activate
cd ISO-Bench
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run TRAE agent on chunked local attention task
python -m bench.cli prepare tasks/chunked_local_attn_optimization.yaml \
  --from-plan ./state/chunked_plan.json \
  --bench-cfg bench_test.yaml \
  --max-workers 1 \
  --resume
```

## Expected Results

✅ **Real-time logging** - See agent steps as they happen  
✅ **File changes detected** - "Files changed by agent: 2"  
✅ **Task completion** - "Task status determined as: success"  
✅ **Git commits** - Agent creates proper commits  
✅ **Patch generation** - Complete model_patch.diff files  

## What You'll See

```
2025-09-18 22:47:40 - bench.prepare - INFO - Starting task processing
2025-09-18 22:47:40 - bench.prepare - INFO - TRAE STDOUT: │ Status │ ✅ Step 1: Completed
...
2025-09-18 22:53:51 - bench.prepare - INFO - Files changed by agent: 2
2025-09-18 22:53:51 - bench.prepare - INFO -   Changed file: vllm/config.py
2025-09-18 22:53:51 - bench.prepare - INFO -   Changed file: vllm/envs.py
2025-09-18 22:53:51 - bench.prepare - INFO - Task status determined as: success
```

## Need Help?

- 📖 **Full Guide**: See `TRAE_AGENT_REPLICATION_GUIDE.md`
- 📋 **Summary**: See `TRAE_AGENT_INTEGRATION_SUMMARY.md`  
- 🐛 **Issues**: Check troubleshooting section in replication guide

---

**Ready to go!** 🎉 Your TRAE agent integration is now complete and reproducible.
