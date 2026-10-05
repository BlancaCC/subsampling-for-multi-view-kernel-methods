conda activate coco
python -m experiments.bo_accuracy_in_view --no-correction
python -m analysis_scripts.bo_accuracy_in_view --results-root results_mv_lssvm_no_correction --analysis-dir analysis_mv_lssvm_no_correction