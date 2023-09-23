#!/usr/bin/env Rscript
library(boot)
library(readr)
library(tidyr)
library(dplyr)
library(lme4)
library(MASS)
library(brms)
library(optparse)
library(lme4)
library(arm)

dir.create("pl_analysis/model_fits/", showWarnings = F)


option_list <- list(
  make_option(c("-m", "--setting"), default=NA, type='character',
              help="setting (has to match name of subdirectory",
              metavar="character"),
  make_option(c("-s", "--steps"), default=NA, type='character',
              help="setting (has to match name of subdirectory"),
  make_option(c("-o", "--original"), action="store_true", default=FALSE,
              help="process original data")
);

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

options(mc.cores = parallel::detectCores())
# options(mc.cores = 4)
ITERATIONS <- 4000

preprocess <- function(data_frame){
  data_frame$reader_id <- factor(data_frame$reader_id)
  data_frame$lexical_frequency <- as.vector(scale(data_frame$lexical_frequency))
  data_frame$surprisal <- as.vector(scale(data_frame$surprisal))
  data_frame$word_length <- as.vector(scale(data_frame$word_length))
  return(data_frame)
}

fit_bayesian_model_predicted_data <- function(data){
  m_reg <- fit_logreg_model(data, "regression", "predicted")
  m_skip <- fit_logreg_model(data, "skipped", "predicted")
  m_count_tot <- fit_count_model(data, "n_tot_count", "predicted")
  m_firstpass_count <- fit_count_model(data, "n_firstpass_count", "predicted")
  all_rm <- list(m_reg=m_reg, m_count_tot=m_count_tot, m_firstpass_count=m_firstpass_count, m_skip=m_skip)
  return(all_rm)
}

fit_bayesian_model_original_data <- function(data){
  m_reg <- fit_logreg_model(data, "regression", "original")
  m_skip <- fit_logreg_model(data, "skipped", "original")
  m_count_tot <- fit_count_model(data, "n_tot_count", "original")
  m_firstpass_count <- fit_count_model(data, "n_firstpass_count", "original")
  all_rm <- list(m_reg=m_reg, m_count_tot=m_count_tot, m_firstpass_count=m_firstpass_count, m_skip=m_skip)
  return(all_rm)
}

fit_logreg_model <- function(data, target, prediction){
  if (prediction == "original"){
    formula <- paste0(target, "~ (1|reader_id) + word_length + surprisal + lexical_frequency")
  } else {
    formula <- paste0(target, "~ word_length + surprisal + lexical_frequency")
  }

  model <- brm(formula = formula,
                   data=data,
                   family = bernoulli(link = "logit"),
                   warmup = ITERATIONS/4,
                   iter = ITERATIONS,
                   chains = 4,
                   cores = 4,
                   seed = 123)
  return(model)
}

fit_count_model <- function(data, target, prediction){
  if (prediction == "original"){
    formula <- paste0(target, "~ (1|reader_id) + word_length + surprisal + lexical_frequency")
  } else {
    formula <- paste0(target, "~ word_length + surprisal + lexical_frequency")
  }
  model <- brm(
    formula = formula,
    data=data,
    family = poisson,
    warmup = ITERATIONS/4,
    iter = ITERATIONS,
    chains = 4,
    cores = 4,
    seed = 123)
  return(model)
}

fit_original_data <- function(file_paths){
  prediction <- "original"
  for(i in 1:2){
    file <- file_paths[i]
    data <- read.csv(file)
    data <- preprocess(data)
    if (grepl("zuco", file)){
      outfile_name <- "zuco"}
    else if (grepl("celer", file)){
      outfile_name <- "celer"}
    else {print("invalid file input")}
    print(outfile_name)
    fit_original_data <- fit_bayesian_model_original_data(data)
    saveRDS(fit_original_data, paste0("pl_analysis/model_fits/fits_all_rm_", outfile_name, "_", prediction, ".rds"))
  }
  return(1)
}

fit_model_data <- function(file_paths){
  prediction <- "predicted"
  print(length(file_paths)) # should be 1
  file <- file_paths[1]
  data <- read.csv(file)
  data <- preprocess(data)
  fits_predicted_data <- fit_bayesian_model_predicted_data(data)
  saveRDS(fits_predicted_data, paste0("pl_analysis/model_fits/fits_all_rm_", opt$setting, "_", prediction, ".rds"))
  return(1)
}

if (opt$o == T){
  file_paths <- list.files(path = "pl_analysis/reading_measures",
                        pattern = paste0("reading_measures_(celer|zuco).csv"),
                        full.names = T)
  fit_original_data(file_paths)
} else {
  file_paths <- list.files(path = "pl_analysis/reading_measures",
                        pattern = paste0("reading_measures_", opt$setting, "_", opt$steps, "\\w+.csv"),
                        full.names = T)
  fit_model_data(file_paths)
}
