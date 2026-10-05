conda activate coco
python -m experiments.bo_late_fusion
python -m analysis_scripts.bo_accuracy_in_view --results-root results_late_fusion --analysis-dir analysis_late_fusion