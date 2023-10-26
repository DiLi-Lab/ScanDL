#!/sr/bin/env Rscript
library(boot)
library(readr)
library(tidyr)
library(dplyr)
library(gridExtra)
library(lme4)
library(MASS)
library(testit)
library(stringr)
library(useful)
library(ggplot2)
library(tibble)
library(RColorBrewer)
library(colorspace)

rm(list = ls())

dirinput <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dirinput)

# In-depth comparison scandl human data

celer_nld_rm_0 <- read.csv("rm_model_inspection/scandl_only/reading_measures_scandl_only_celer_0.csv", header=T)
celer_nld_rm_1 <- read.csv("rm_model_inspection/scandl_only/reading_measures_scandl_only_celer_1.csv", header=T)
celer_nld_rm_2 <- read.csv("rm_model_inspection/scandl_only/reading_measures_scandl_only_celer_2.csv", header=T)
celer_nld_rm_3 <- read.csv("rm_model_inspection/scandl_only/reading_measures_scandl_only_celer_3.csv", header=T)
celer_nld_rm_4 <- read.csv("rm_model_inspection/scandl_only/reading_measures_scandl_only_celer_4.csv", header=T)
zuco_nld_rm <- read.csv("rm_model_inspection/scandl_only/reading_measures_scandl_only_zuco_0.csv", header=T)
celer_nld_rm <- rbind(celer_nld_rm_0, celer_nld_rm_1, celer_nld_rm_2, celer_nld_rm_3 ,celer_nld_rm_4)

preprocess <- function(nld_rm){
  nld_rm$reader_id <- as.factor(nld_rm$reader_id)
  nld_rm$avg_rsacc_original <- abs(nld_rm$avg_rsacc_original)
  nld_rm$avg_rsacc_predicted <- abs(nld_rm$avg_rsacc_predicted)
  return(nld_rm)
}

celer_nld_rm <- preprocess(celer_nld_rm)
zuco_nld_rm <- preprocess(zuco_nld_rm)

compare_reading_measures <- function(nld_rm){
  avg_rm_original <- nld_rm %>%
    group_by() %>%
    summarise_at(c("avg_firstpass_original",  "avg_tft_original", "avg_skips_original", "avg_regs_original",
                   "avg_psacc_original", "avg_rsacc_original"),
                 list(mean = mean, sd = sd), na.rm = TRUE)

  colnames(avg_rm_original) <-  c(
    "avg_firstpass_mean", "avg_tft_mean", "avg_skips_mean", "avg_regs_mean", "avg_psacc_mean", "avg_rsacc_mean",
    "avg_firstpass_sd", "avg_tft_sd", "avg_skips_sd", "avg_regs_sd", "avg_psacc_sd", "avg_rsacc_sd"
  )
  avg_rm_original$data <- "original"

  avg_rm_predicted <- nld_rm %>%
    group_by() %>%
    summarise_at(c("avg_firstpass_predicted",  "avg_tft_predicted", "avg_skips_predicted", "avg_regs_predicted",
                   "avg_psacc_predicted", "avg_rsacc_predicted"),
                 list(mean = mean, sd = sd), na.rm = TRUE)

  colnames(avg_rm_predicted) <-  c(
    "avg_firstpass_mean", "avg_tft_mean", "avg_skips_mean", "avg_regs_mean", "avg_psacc_mean", "avg_rsacc_mean",
    "avg_firstpass_sd", "avg_tft_sd", "avg_skips_sd", "avg_regs_sd", "avg_psacc_sd", "avg_rsacc_sd"
  )
  avg_rm_predicted$data <- "predicted"

  avg_rm <- rbind(avg_rm_original, avg_rm_predicted)

  avg_rm_long_mean <-  gather(avg_rm, "reading_measure", "mean_measure", 1:6)[,7:9]
  avg_rm_long_mean$reading_measure <- factor(avg_rm_long_mean$reading_measure)
  levels(avg_rm_long_mean$reading_measure) <- c(
    "avg_firstpass", "avg_psacc", "avg_reg", "avg_rsacc", "avg_skips", "avg_tft"
  )

  avg_rm_long_sd <-  gather(avg_rm, "reading_measure", "sd_measure", 7:12)[,7:9]
  avg_rm_long_sd$reading_measure <- factor(avg_rm_long_sd$reading_measure)
  levels(avg_rm_long_sd$reading_measure) <- c(
    "avg_firstpass", "avg_psacc", "avg_reg", "avg_rsacc", "avg_skips", "avg_tft"
  )

  avg_rm_long_mean$sd_measure <- avg_rm_long_sd$sd_measure
  return(avg_rm_long_mean)
}

rm_zuco <- compare_reading_measures(zuco_nld_rm)
rm_celer <- compare_reading_measures(celer_nld_rm)

rm_zuco$dataset <- "Zuco"
rm_celer$dataset <- "Celer"
rm_all <- rbind(rm_zuco, rm_celer)
rm_all$dataset <- as.factor(rm_all$dataset)
rm_all$reading_measure <- factor(rm_all$reading_measure, levels=c("avg_firstpass",  "avg_tft", "avg_skips","avg_reg", "avg_psacc",  "avg_rsacc"))

levels(rm_all$reading_measure) <- c("Firstpass fix. count", "Total fix. count", "Skipping rate", "Regression rate", "Progressive sacc. length",
                                    "Regressive sacc. length")

rm_all <- mutate(rm_all,
                       rm_type = case_when(
                         (reading_measure=="Firstpass fix. count"|reading_measure=="Total fix. count")  ~ "counts",
                         (reading_measure=="Skipping rate"|reading_measure=="Regression rate") ~ "rates",
                         (reading_measure=="Progressive sacc. length"|reading_measure=="Regressive sacc. length") ~ "lengths"))

  colors <- diverging_hcl(7, palette = "Blue-Red")[c(2, 6)]
rm_plot_all <- ggplot(rm_all, aes(x=reading_measure, y=mean_measure, group=dataset, color=dataset, shape=data)) +
  geom_point(position=position_dodge2(0.45), size=2) +
  geom_errorbar(aes(ymin=mean_measure-sd_measure, ymax=mean_measure+sd_measure), width=.45,
                position=position_dodge2(0.45)) +
  scale_color_manual(values=colors) +
  # scale_x_discrete(guide = guide_axis(angle = 20, hjust=5)) +
  xlab("Measures") + ylab("") +
  guides(colour = guide_legend(label.position = "bottom")) +
  facet_wrap(.~rm_type, scales="free", ncol=3) +
  theme_light() +
  theme(strip.background = element_blank(),
        strip.text.x = element_blank()) +
  theme(axis.text.x = element_text(angle = 20, hjust=1)) +
  guides(colour = guide_legend(nrow = 1), group = guide_legend(nrow = 1), shape = guide_legend(nrow = 1)) +
  guides(color=guide_legend(title='Dataset'), shape=guide_legend(title='')) +
  theme(legend.position = "top")

ggsave("figs/rm_scandl_original_predicted.png", width = 8, height = 4, dpi=300)

### correlation between nld and average rm

corrFunc <- function(var1, var2, data) {
  x <- as.numeric(unlist(data[,var1]))
  y <- as.numeric(unlist(data[,var2]))
  result = cor.test(x, y)
  data.frame(var1, var2, result[c("estimate","p.value")],
             stringsAsFactors=FALSE)
}

is_sig <- function(value){
  if(value<=0.001){
    return("***")
  } else if (value<=0.01){
    return("**")
  } else if (value <0.05){
    return("*")
  }
  else {return("not sig")}
}

summarize_rm_per_reader <- function(nld_rm){
  avg_nld_rm <- nld_rm %>%
    group_by(reader_id) %>%
    summarise_at(6:(ncol(nld_rm)-1), list(mean = mean, sd = sd, min= min, max = max), na.rm = TRUE)
  return(avg_nld_rm)
}

correlate_nld_with_rm <- function(df){
  train = df[, c("nld_mean", "avg_firstpass_original_mean", "avg_tft_original_mean", "avg_regs_original_mean",
                 "avg_psacc_original_mean", "avg_rsacc_original_mean", "avg_skips_original_mean")]
  vars = data.frame(v1="nld_mean",
                  v2=c("avg_firstpass_original_mean", "avg_tft_original_mean", "avg_regs_original_mean",
                 "avg_psacc_original_mean", "avg_rsacc_original_mean", "avg_skips_original_mean"))
  corrs = do.call(rbind, mapply(corrFunc, vars[,1], vars[,2], MoreArgs=list(data=train),
                              SIMPLIFY=FALSE))

  corrs <- corrs %>%
    rowwise() %>%
    mutate(sig = is_sig(p.value))
  attach(corrs)
  corrs_ordered <- corrs[order(estimate, decreasing = T),]
  return(corrs_ordered)
}

avg_nld_rm_celer <- summarize_rm_per_reader(celer_nld_rm)
celer_correlations <- correlate_nld_with_rm(avg_nld_rm_celer)
avg_nld_rm_zuco <- summarize_rm_per_reader(zuco_nld_rm)
zuco_correlations <- correlate_nld_with_rm(avg_nld_rm_zuco)


### Cross-model comparison

cross_model <- read.csv("rm_model_inspection/cross_model/reading_measures_crossmodel_combined.csv", header=T)
cross_model$sentence <- factor(cross_model$sentence)
cross_model$model <- factor(cross_model$model)

cross_model %>% count(model)

cross_model$avg_rsacc_original <- abs(cross_model$avg_rsacc_original)
cross_model$avg_rsacc_predicted <- abs(cross_model$avg_rsacc_predicted)


correlate_against_rm <- function(df){
    train = df[, c("nld", "avg_firstpass_original", "avg_tft_original", "avg_regs_original",
                 "avg_psacc_original", "avg_rsacc_original", "avg_skips_original")]

  vars = data.frame(v1="nld",
                    v2=c("avg_firstpass_original", "avg_tft_original", "avg_regs_original",
                 "avg_psacc_original", "avg_rsacc_original", "avg_skips_original"))
  corrs = do.call(rbind, mapply(corrFunc, vars[,1], vars[,2], MoreArgs=list(data=train),
                              SIMPLIFY=FALSE))

  corrs <- corrs %>%
    rowwise() %>%
    mutate(sig = is_sig(p.value))
  attach(corrs)
  corrs_ordered <- corrs[order(corrs$estimate, decreasing = T),]
  return(corrs_ordered)
}


scandl <- cross_model[cross_model$model=="scandl",]
corrs_scandl <- correlate_against_rm(scandl)
corrs_scandl$model <- "scandl"

eyettention <- cross_model[cross_model$model=="eyettention",]
corrs_eyettention <- correlate_against_rm(eyettention)
corrs_eyettention$model <- "eyettention"

ez <- cross_model[cross_model$model=="ez_reader",]
corrs_ez <- correlate_against_rm(ez)
corrs_ez$model <- "ez-reader"

swift <- cross_model[cross_model$model=="swift",]
corrs_swift <- correlate_against_rm(swift)
corrs_swift$model <- "swift"

traindist <- cross_model[cross_model$model=="traindist",]
corrs_traindist <- correlate_against_rm(traindist)
corrs_traindist$model <- "traindist"

uniform <- cross_model[cross_model$model=="uniform",]
corrs_uniform <- correlate_against_rm(uniform)
corrs_uniform$model <- "uniform"

all_cors <- rbind(
  corrs_scandl, corrs_eyettention, corrs_ez, corrs_swift, corrs_traindist, corrs_uniform
)

all_cors$estimate <- round(all_cors$estimate, 2)

write.csv(all_cors, "correlations_nld_vs_rm.csv", row.names = F)
