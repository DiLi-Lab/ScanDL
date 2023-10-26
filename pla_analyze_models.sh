nohup Rscript --vanilla pl_analysis/analyze_fit.R --setting reader --steps 50000 > log/p_reader.log&
nohup Rscript --vanilla pl_analysis/analyze_fit.R --setting sentence --steps 50000 > log/p_sentence.log&
nohup Rscript --vanilla pl_analysis/analyze_fit.R --setting combined --steps 50000 > p_combined.log&
nohup Rscript --vanilla pl_analysis/analyze_fit.R --setting cross_dataset --steps 70000 > p_cross_dataset.log&