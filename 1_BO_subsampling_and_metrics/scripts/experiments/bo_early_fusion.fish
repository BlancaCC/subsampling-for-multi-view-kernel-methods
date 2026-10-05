conda activate coco
python -m experiments.bo_early_fusion
python -m analysis_scripts.bo_accuracy_in_view --results-root results_early_fusion --analysis-dir analysis_early_fusion