#!/usr/bin/env Rscript
library(boot)
library(readr)
library(tidyr)
library(dplyr)
library(gridExtra)
library(lme4)
library(MASS)
library(brms)
library(optparse)
library(lme4)
library(arm)
library(ggplot2)
library(lemon)
library(RColorBrewer)
library(colorspace)

dir.create("pl_analysis/figs/", showWarnings = F)
dir.create("pl_analysis/tabs/", showWarnings = F)

option_list <- list(
  make_option(c("-m", "--setting"), default=NA, type='character',
              help="setting (has to match name of subdirectory",
              metavar="character"),
  make_option(c("-s", "--steps"), default=NA, type='character',
              help="setting (has to match name of subdirectory")
);

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

# reading_measures <- c("first-pass regressions", "total fixation count", "first-pass fixation count", "skips")
# reading_measures <- c("regression", "skipped", "n_tot_count", "n_firstpass_count")
reading_measures <- c("m_reg", "m_count_tot", "m_firstpass_count", "m_skip")
effects <- c("word_length", "surprisal", "lexical_frequency")
b_estimates <- c("Estimate", "Q2.5", "Q97.5", "Est.Error")

extract_bayesian_results <- function(results_list, source){
    results_table <- data.frame(
      source = character(),
      reading_measure = character(),
      effect = character(),
      mean_effect_size = double(),
      ci_lower = double(),
      ci_upper = double(),
      est_error = double(),
      stringsAsFactors=TRUE
    )
  source <- source
    for (i in 1:4){
    reading_measure <- reading_measures[i]
    # for each reading measure
    model_fit <- results_list[[reading_measure]]
    results <- fixef(model_fit)
    for (j in 1:3){
      effect <- effects[j]
      mean_est <- results[effect, b_estimates[1]]
      ci_lower <- results[effect, b_estimates[2]]
      ci_upper <- results[effect, b_estimates[3]]
      est_error <- results[effect, b_estimates[4]]
      res_frame <- data.frame(
        source=source,
        reading_measure=reading_measure,
        effect=effect,
        mean_effect_size=mean_est,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        est_error=est_error
      )
      results_table <- rbind(results_table, res_frame)
    }
  }
  return(results_table)
}

get_all_results <- function(setting){
  if (setting == "cross_dataset"){
    fits_original_data <- readRDS(paste0("pl_analysis/model_fits/fits_all_rm_zuco_original.rds"))
  } else {
    fits_original_data <- readRDS(paste0("pl_analysis/model_fits/fits_all_rm_celer_original.rds"))
  }
  original_results <- extract_bayesian_results(fits_original_data, source="original")

  fits_scandl <- readRDS(paste0("pl_analysis/model_fits/fits_all_rm_", opt$setting, "_predicted.rds"))
  predicted_results_scandl <- extract_bayesian_results(fits_scandl, source="prediction")

  fits_ez <- readRDS("pl_analysis/model_fits/fits_all_rm_ez-reader_predicted.rds")
  predicted_results_ez <- extract_bayesian_results(fits_ez, source="ez-reader")

  fits_swift <- readRDS("pl_analysis/model_fits/fits_all_rm_swift_predicted.rds")
  predicted_results_swift <- extract_bayesian_results(fits_swift, source="swift")

  fits_unif <- readRDS("pl_analysis/model_fits/fits_all_rm_uniform_predicted.rds")
  predicted_results_uniform <- extract_bayesian_results(fits_unif, source="uniform")

  fits_td <- readRDS("pl_analysis/model_fits/fits_all_rm_traindist_predicted.rds")
  predicted_results_traindist <- extract_bayesian_results(fits_td, source="traindist")

  fits_et <- readRDS("pl_analysis/model_fits/fits_all_rm_local_predicted.rds")
  predicted_results_et <- extract_bayesian_results(fits_et, source="eyettention")

  all_results <- rbind(original_results, predicted_results_scandl, predicted_results_ez, predicted_results_swift,
                       predicted_results_et, predicted_results_uniform, predicted_results_traindist)
  return(all_results)
}

myjit <- ggproto("fixJitter", PositionDodge,
                 width = 0.3,
                 dodge.width = 0.25,
                 jit = NULL,
                 compute_panel =  function (self, data, params, scales)
                 {

                   #Generate Jitter if not yet
                   if(is.null(self$jit) ) {
                    self$jit <-jitter(rep(0, nrow(data)), amount=self$dodge.width)
                   }

                   data <- ggproto_parent(PositionDodge, self)$compute_panel(data, params, scales)

                   data$x <- data$x + self$jit
                   #For proper error extensions
                   if("xmin" %in% colnames(data)) data$xmin <- data$xmin + self$jit
                   if("xmax" %in% colnames(data)) data$xmax <- data$xmax + self$jit
                   data
                 }
)


make_plots_bayes <- function(all_results){
  colors <- c(
    sequential_hcl(7, palette = "BuGn")[3], #human
    sequential_hcl(7, palette = "PuRd")[3], # scandl
    sequential_hcl(7, palette = "YlOrBr")[3], # et
    sequential_hcl(7, palette = "YlGnBu")[c(2, 4)],
    sequential_hcl(7, palette = "Turku")[c(4, 6)] # unif and td
)
  p<-ggplot(all_results,
            aes(x=effect, y=mean_effect_size, colour=source, position = "dodge")) +
    scale_x_discrete(guide = guide_axis(angle = 20)) +
    scale_color_manual(values=colors) +
    geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper),
                       width=.25, position = myjit) +
    geom_point(position = myjit, size = 0.2) +
    # facet_wrap(reading_measure ~ .) +
    facet_rep_wrap(reading_measure ~ .,  scales = "free_y")  +
    geom_hline(yintercept=0, linetype="dashed", color = "grey51") +
    xlab("Psycholinguistic phenomenon") + ylab("Posterior effect estimate") +
    theme_light() +
    theme(legend.position="top", legend.box = "horizontal") +
    guides(colour = guide_legend(label.position = "bottom")) +
    guides(colour=guide_legend(title="Model"))
  ggsave(paste0("pl_analysis/figs/pla_", opt$setting, "_", opt$steps, ".jpg"), width = 5, height = 6, dpi=500)
  return(1)
}

create_summary_tables <- function(df){
  outfile <- paste0("pl_analysis/tabs/overleaf_table_", opt$setting, ".txt")
  df <- df[order(df$effect, df$reading_measure, df$source), ]
  df[c("mean_effect_size", "est_error", "ci_lower", "ci_upper")] <- lapply(
    df[c("mean_effect_size", "est_error", "ci_lower", "ci_upper")], function(x) round(x, digits=2))
  sink(outfile)
  for (i in 1:nrow(df)) {
    formatted_row <- c(
      paste0(df[i, "effect"]),
      paste0(df[i, "reading_measure"]),
      paste0(df[i, "source"]),
      paste0("$", format(df[i, "mean_effect_size"], nsmall = 2), "\\pm{", format(df[i, "est_error"], nsmall = 2), "}$"),
      paste0("$[", format(df[i, "ci_lower"], nsmall = 2), ", ", format(df[i, "ci_upper"], nsmall = 2), "]$")
    )
    cat(paste(formatted_row, collapse = " & "))
    cat(" \\\\", "\n")
    }
    sink()
  return(1)
}

main <- function(){
  all_results <- get_all_results(setting=opt$setting)
  all_results$source <- factor(all_results$source)
  levels(all_results$source) <- c('Eyettention', 'EZ-reader', 'Human', 'ScanDL', 'SWIFT', 'Train-label-dist', 'Uniform')
  all_results$source <- factor(
    all_results$source, levels =  c('Human', 'ScanDL', 'Eyettention', 'EZ-reader', 'SWIFT', 'Train-label-dist', 'Uniform')
  )
  all_results$effect <- factor(all_results$effect)
  levels(all_results$effect) <- c('lexical frequency', 'surprisal', 'word length')
  all_results$reading_measure <- factor(all_results$reading_measure)
  levels(all_results$reading_measure) <- c(
    'total fixation count', 'first-pass fixation count', 'first-pass regression rate', 'skipping rate')
  all_results$reading_measure <- factor(
    all_results$reading_measure, levels = c('skipping rate', 'first-pass regression rate',
                                            'total fixation count', 'first-pass fixation count'))

  make_plots_bayes(all_results)
  create_summary_tables(all_results)
}

main()