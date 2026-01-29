# Plotting offline executive benchmark figures

This folder contains scripts to generate paper-ready figures from the files under:

- `grasp-copilot/evaluation/eval_outputs/<run>/`

## Generate figures for a run

Install plotting deps (once):

```bash
python -m pip install -e "grasp-copilot[plot]"
```

Example (for `paper_benchmark_run001`):

```bash
conda activate llm
cd /home/ali/github/ali-rabiee/grasp-copilot
python -m evaluation.plots.make_offline_exec_figures \
  --run_dir /home/ali/github/ali-rabiee/grasp-copilot/evaluation/eval_outputs/paper_benchmark_run001 \
  --out_dir /home/ali/github/ali-rabiee/grasp-copilot/evaluation/plots \
  --tag paper_benchmark_run001
```

Outputs (PNG):
- `*_fig1_main_metrics.png`
- `*_fig2_reliability_invalid_rates.png`
- `*_fig3_tradeoff_accuracy_vs_speed.png`
- `*_fig4_context_heatmap.png`
- `*_fig5_confusion_<model>.png`


