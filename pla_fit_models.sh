# Rscript --vanilla pl_analysis/compute_effects.R --setting combined --steps 80000 > combined_new.log&
# Rscript --vanilla pl_analysis/compute_effects.R --setting traindist --steps 0 > traindist_new.log&
# Rscript --vanilla pl_analysis/compute_effects.R --setting uniform --steps 0 > uniform_new.log&
# Rscript --vanilla pl_analysis/compute_effects.R --setting local --steps 0 > eyettention_new.log&
Rscript --vanilla pl_analysis/compute_effects.R --setting ez-reader --steps 0 > ez-reader_new.log
Rscript --vanilla pl_analysis/compute_effects.R --setting swift --steps 0 > swift_new.log
