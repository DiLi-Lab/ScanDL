nohup sudo Rscript --vanilla pl_analysis/compute_effects.R --setting reader --steps 50000 >& reader.log
nohup sudo Rscript --vanilla pl_analysis/compute_effects.R --setting sentence --steps 50000 >& sentence.log
nohup sudo Rscript --vanilla pl_analysis/compute_effects.R --setting combined --steps 50000 >& combined.log
nohup sudo Rscript --vanilla pl_analysis/compute_effects.R --setting cross_dataset --steps 70000 >& cross_dataset.log