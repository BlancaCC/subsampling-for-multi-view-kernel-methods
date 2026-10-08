conda activate coco
python -m analysis_scripts.subset_properties --results-root results_late_fusion --analysis-dir analysis_late_fusion
python -m analysis_scripts.subset_properties --results-root results_mv_lssvm_correction --analysis-dir analysis_mv_lssvm_correction
python -m analysis_scripts.subset_properties --results-root results_mv_lssvm_no_correction --analysis-dir analysis_mv_lssvm_no_correction
