# ==============================================================================
# Script: 02_lme_models_complete.R
# Purpose: Comprehensive linear mixed-effects models for all analyses:
#          - Subjective ratings ~ semantic segmentation
#          - ERPs ~ semantic segmentation
#          - Subjective ratings ~ LLVF features
#          - ERPs ~ LLVF features
# Note: First prepare the data
# ==============================================================================

# Load required packages -------------------------------------------------------
library(dplyr)
library(readr)
library(lme4)
library(lmerTest)
library(effectsize)
library(broom.mixed)
library(MuMIn)
library(janitor)
# library(sjPlot)

# Configuration ----------------------------------------------------------------
DATA_FILE <- "/path/to/combined_data_full.csv"
P3_LPP_FILE <- "/path/to/AverageP3_LPP.csv"
LLVF_FILE <- "/path/to/llvf_file.csv"
DEPTH_FILE <- "/path/to/depth_file.csv"
GREEN_MASK_FILE <- "/path/to/green_mask_file.csv"
OUTPUT_DIR <- "/path/to/output"

# Analysis control (set to FALSE to skip sections) ----------------------------
RUN_SEMSEG_SUBJECTIVE <- TRUE
RUN_SEMSEG_ERP <- TRUE
RUN_LLVF_SUBJECTIVE <- TRUE
RUN_LLVF_ERP <- TRUE

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

clean_str <- function(x) trimws(tolower(as.character(x)))

#' Prepare aggregated data for subjective rating models
prepare_question_data <- function(data, question_label, invert_scale = FALSE) {
  agg_data <- data %>%
    filter(Question == question_label) %>%
    group_by(Subject, Condition, Question) %>%
    summarise(mean_rating = mean(Rating, na.rm = TRUE), .groups = "drop")
  
  if (invert_scale) agg_data <- agg_data %>% mutate(mean_rating = 10 - mean_rating)
  
  # Add image features
  agg_data <- agg_data %>%
    left_join(
      data %>% select(Subject, Condition, sky, car, person, green_imperv) %>% distinct(),
      by = c("Subject", "Condition")
    ) %>%
    distinct()
  
  # Add personal predictors
  agg_data <- agg_data %>%
    left_join(
      data %>% select(Subject, Gender_binary, City_15, extrov, emostab) %>% distinct(),
      by = "Subject"
    )
  
  # Scale predictors
  agg_data <- agg_data %>%
    mutate(
      z_green_imperv = as.numeric(scale(green_imperv)),
      z_sky = as.numeric(scale(sky)),
      z_car = as.numeric(scale(car)),
      z_person = as.numeric(scale(person)),
      Gender_binary = relevel(as.factor(Gender_binary), ref = "Male")
    )
  
  return(agg_data)
}

#' Prepare aggregated data for LLVF models
prepare_llvf_data <- function(data, question_label, invert_scale = FALSE) {
  agg_data <- data %>%
    filter(Question == question_label) %>%
    group_by(Subject, Condition, Question) %>%
    summarise(mean_rating = mean(Rating, na.rm = TRUE), .groups = "drop")
  
  if (invert_scale) agg_data <- agg_data %>% mutate(mean_rating = 10 - mean_rating)
  
  # Add green_imperv if needed
  if (!"green_imperv" %in% names(agg_data)) {
    agg_data <- agg_data %>%
      left_join(
        data %>% select(Subject, Condition, green_imperv) %>% distinct(),
        by = c("Subject", "Condition")
      )
  }
  
  # Add LLVF features
  agg_data <- agg_data %>%
    left_join(
      data %>% select(
        Subject, Condition, gvHue, rcHue, Brightness, sdBrightness,
        Saturation, sdSaturation, StraightEdgeDensity, NonStraightEdgeDensity,
        NonStraightEdgeDensityWithGreen, Entropy, FractalD,
        depth_metric_mean, depth_metric_sd, GreenMaskWThresholdedDensity
      ) %>% distinct(),
      by = c("Subject", "Condition")
    ) %>%
    distinct()
  
  # Add personal predictors
  agg_data <- agg_data %>%
    left_join(
      data %>% select(Subject, Gender_binary, City_15, extrov, emostab) %>% distinct(),
      by = "Subject"
    ) %>%
    distinct()
  
  # Scale all predictors
  agg_data <- agg_data %>%
    mutate(
      z_green_imperv = as.numeric(scale(green_imperv)),
      z_gvHue = as.numeric(scale(gvHue)),
      z_rcHue = as.numeric(scale(rcHue)),
      z_Brightness = as.numeric(scale(Brightness)),
      z_sdBrightness = as.numeric(scale(sdBrightness)),
      z_Saturation = as.numeric(scale(Saturation)),
      z_sdSaturation = as.numeric(scale(sdSaturation)),
      z_StraightEdgeD = as.numeric(scale(StraightEdgeDensity)),
      z_NonStraightED = as.numeric(scale(NonStraightEdgeDensity)),
      z_NonStraightEDG = as.numeric(scale(NonStraightEdgeDensityWithGreen)),
      z_Entropy = as.numeric(scale(Entropy)),
      z_FractalD = as.numeric(scale(FractalD)),
      z_depth_metric_mean = as.numeric(scale(depth_metric_mean)),
      z_depth_metric_sd = as.numeric(scale(depth_metric_sd)),
      z_green_mask_density = as.numeric(scale(GreenMaskWThresholdedDensity)),
      Gender_binary = relevel(as.factor(Gender_binary), ref = "Male")
    )
  
  return(agg_data)
}

#' Fit semantic segmentation model sequence
fit_semseg_models <- function(data, dv_name = "mean_rating") {
  formula_base <- paste0(dv_name, " ~ z_green_imperv + z_sky + z_car + z_person")
  
  models <- list(
    lmer(as.formula(paste(formula_base, "+ (1 | Condition) + (1 | Subject)")),
         data = data, REML = TRUE),
    lmer(as.formula(paste(formula_base, "+ Gender_binary + City_15 + extrov + emostab + (1 | Condition) + (1 | Subject)")),
         data = data, REML = TRUE),
    lmer(as.formula(paste(formula_base, "+ Gender_binary + City_15 + extrov + emostab + (1 | Condition) + (1 + z_green_imperv | Subject)")),
         data = data, REML = TRUE),
    lmer(as.formula(paste(formula_base, "+ Gender_binary + City_15 + extrov + emostab + (1 | Condition) + (1 + z_green_imperv + z_sky | Subject)")),
         data = data, REML = TRUE),
    lmer(as.formula(paste(formula_base, "+ Gender_binary + City_15 + extrov + emostab + (1 | Condition) + (1 + z_green_imperv + z_sky + z_car + z_person | Subject)")),
         data = data, REML = TRUE)
  )
  return(models)
}

#' Fit LLVF model sequence
fit_llvf_models <- function(data, dv_name = "mean_rating") {
  formula_base <- paste0(dv_name, " ~ z_green_mask_density + z_Saturation + z_sdSaturation + z_StraightEdgeD + z_NonStraightED + z_Entropy + z_FractalD + z_depth_metric_mean")
  
  models <- list(
    lmer(as.formula(paste(formula_base, "+ (1 | Condition) + (1 | Subject)")),
         data = data, REML = FALSE),
    lmer(as.formula(paste(formula_base, "+ Gender_binary + City_15 + extrov + emostab + (1 | Condition) + (1 | Subject)")),
         data = data, REML = FALSE),
    lmer(as.formula(paste(formula_base, "+ Gender_binary + City_15 + extrov + emostab + (1 | Condition) + (1 + z_gvHue | Subject)")),
         data = data, REML = TRUE),
    lmer(as.formula(paste(formula_base, "+ Gender_binary + City_15 + extrov + emostab + (1 | Condition) + (1 + z_depth_metric_mean | Subject)")),
         data = data, REML = TRUE),
    lmer(as.formula(paste(formula_base, "+ Gender_binary + City_15 + extrov + emostab + (1 | Condition) + (1 + z_gvHue + z_depth_metric_mean | Subject)")),
         data = data, REML = TRUE)
  )
  return(models)
}

#' Print comprehensive model summary
print_model_summary <- function(model, model_name) {
  
  cat("Cohen's f²:\n")
  print(cohens_f_squared(model))
  
  cat("\nFixed effects (FDR-corrected):\n")
  fixed_fx <- tidy(model, effects = "fixed", conf.int = TRUE)
  if (!"p.value" %in% colnames(fixed_fx)) {
    fixed_fx <- fixed_fx %>% mutate(p.value = 2 * (1 - pnorm(abs(statistic))))
  }
  fixed_fx <- fixed_fx %>% mutate(p.adj = p.adjust(p.value, method = "fdr"))
  print(fixed_fx, n = Inf)
  
  cat("\nRandom effects:\n")
  print(VarCorr(model), comp = c("Variance", "Std.Dev."))
  
  cat("\nR²:\n")
  print(r.squaredGLMM(model))
  cat("\n")
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

# Load base data
combined_data <- read_csv(DATA_FILE, show_col_types = FALSE) %>%
  mutate(Subject = as.numeric(Subject), Condition = clean_str(Condition)) %>%
  filter(Subject != 80)

# Calculate greenery:impervious ratio
combined_data <- combined_data %>%
  mutate(
    greenery = plant + tree + grass,
    impervious = building + road + sidewalk,
    green_imperv = ifelse(impervious == 0, NA, greenery / impervious)
  )

# Load P3/LPP data if needed for ERP analyses
if (RUN_SEMSEG_ERP || RUN_LLVF_ERP) {
  p3lpp_data <- read_csv(P3_LPP_FILE, show_col_types = FALSE) %>%
    rename(Subject = subject, Condition = cond) %>%
    mutate(
      Subject = as.numeric(Subject),
      Condition = clean_str(Condition),
      ParietalP3 = as.numeric(ParietalP3),
      ParietalLPP = as.numeric(ParietalLPP)
    ) %>%
    select(Subject, Condition, ParietalP3, ParietalLPP) %>%
    distinct()
  
  dup_keys <- p3lpp_data %>% count(Subject, Condition) %>% filter(n > 1)
  if (nrow(dup_keys) > 0) stop("Duplicate keys in P3/LPP data")
  
  combined_data <- combined_data %>%
    left_join(p3lpp_data, by = c("Subject", "Condition"))
  }

# Load LLVF features if needed
if (RUN_LLVF_SUBJECTIVE || RUN_LLVF_ERP) {
  llvf_vars <- c("gvHue", "Saturation", "StraightEdgeDensity", "Entropy", 
                 "FractalD", "depth_metric_mean", "GreenMaskWThresholdedDensity")
  
  if (!all(llvf_vars %in% names(combined_data))) {
    cat("\nLoading LLVF features...\n")
    
    # LLVF data
    llvf_data <- read.csv(LLVF_FILE, stringsAsFactors = FALSE) %>%
      mutate(ImageName = tolower(trimws(ImageName))) %>%
      filter(ImageName %in% unique(combined_data$Condition))
    
    combined_data <- combined_data %>%
      left_join(
        llvf_data %>% select(ImageName, gvHue, rcHue, HSVHue, Brightness, sdBrightness,
                             Saturation, sdSaturation, StraightEdgeDensity,
                             NonStraightEdgeDensity, NonStraightEdgeDensityWithGreen,
                             Entropy, FractalD),
        by = c("Condition" = "ImageName")
      )
    
    # Depth data
    depth_data <- read.csv(DEPTH_FILE, stringsAsFactors = FALSE) %>%
      clean_names() %>%
      mutate(file_name = tolower(trimws(file_name)))
    
    combined_data <- combined_data %>%
      mutate(Condition = tolower(trimws(Condition))) %>%
      left_join(
        depth_data %>% select(file_name, depth_metric_mean, depth_metric_sd),
        by = c("Condition" = "file_name")
      )
    
    # Green mask data
    green_mask_data <- read.csv(GREEN_MASK_FILE, stringsAsFactors = FALSE) %>%
      mutate(ImageName = tolower(trimws(ImageName))) %>%
      filter(ImageName %in% unique(combined_data$Condition))
    
    combined_data <- combined_data %>%
      left_join(
        green_mask_data %>% select(ImageName, GreenMaskWThresholdedDensity),
        by = c("Condition" = "ImageName")
      )
      }
}

# ==============================================================================
# SECTION 1: SUBJECTIVE RATINGS ~ SEMANTIC SEGMENTATION
# ==============================================================================

if (RUN_SEMSEG_SUBJECTIVE) {
  
  # SAM Scales -----------------------------------------------------------------

  # Arousal
  agg_arousal <- prepare_question_data(combined_data, "sam-arousal", invert_scale = TRUE)
  arousal_models <- fit_semseg_models(agg_arousal)
  print(anova(arousal_models[[1]], arousal_models[[2]], arousal_models[[3]], 
              arousal_models[[4]], arousal_models[[5]]))
  arousal_semseg_final <- arousal_models[[3]]  # Adjust based on anova
  print_model_summary(arousal_semseg_final, "Arousal ~ SemSeg")
  
  # Valence
  agg_valence <- prepare_question_data(combined_data, "sam-valence", invert_scale = TRUE)
  valence_models <- fit_semseg_models(agg_valence)
  print(anova(valence_models[[1]], valence_models[[2]], valence_models[[3]], 
              valence_models[[4]], valence_models[[5]]))
  valence_semseg_final <- valence_models[[4]]
  print_model_summary(valence_semseg_final, "Valence ~ SemSeg")
  
  # Dominance
  agg_dominance <- prepare_question_data(combined_data, "sam-dominance", invert_scale = TRUE)
  dominance_models <- fit_semseg_models(agg_dominance)
  print(anova(dominance_models[[1]], dominance_models[[2]], dominance_models[[3]], 
              dominance_models[[4]], dominance_models[[5]]))
  dominance_semseg_final <- dominance_models[[4]]
  print_model_summary(dominance_semseg_final, "Dominance ~ SemSeg")
  
  # Likert Scales --------------------------------------------------------------

  # Safety
  agg_safety <- prepare_question_data(combined_data, "likert-sicherheit")
  safety_models <- fit_semseg_models(agg_safety)
  print(anova(safety_models[[1]], safety_models[[2]], safety_models[[3]], 
              safety_models[[4]], safety_models[[5]]))
  safety_semseg_final <- safety_models[[4]]
  print_model_summary(safety_semseg_final, "Safety ~ SemSeg")
  
  # Beauty
  agg_beauty <- prepare_question_data(combined_data, "likert-schonheit")
  beauty_models <- fit_semseg_models(agg_beauty)
  print(anova(beauty_models[[1]], beauty_models[[2]], beauty_models[[3]], 
              beauty_models[[4]], beauty_models[[5]]))
  beauty_semseg_final <- beauty_models[[4]]
  print_model_summary(beauty_semseg_final, "Beauty ~ SemSeg")
  
  # Hominess
  agg_hominess <- prepare_question_data(combined_data, "likert-heimeligkeit")
  hominess_models <- fit_semseg_models(agg_hominess)
  print(anova(hominess_models[[1]], hominess_models[[2]], hominess_models[[3]], 
              hominess_models[[4]], hominess_models[[5]]))
  hominess_semseg_final <- hominess_models[[4]]
  print_model_summary(hominess_semseg_final, "Hominess ~ SemSeg")
  
  # Openness
  agg_openness <- prepare_question_data(combined_data, "likert-offenheit")
  openness_models <- fit_semseg_models(agg_openness)
  print(anova(openness_models[[1]], openness_models[[2]], openness_models[[3]], 
              openness_models[[4]], openness_models[[5]]))
  openness_semseg_final <- openness_models[[4]]
  print_model_summary(openness_semseg_final, "Openness ~ SemSeg")
  
  # Fascination
  agg_fascination <- prepare_question_data(combined_data, "likert-faszination")
  fascination_models <- fit_semseg_models(agg_fascination)
  print(anova(fascination_models[[1]], fascination_models[[2]], fascination_models[[3]], 
              fascination_models[[4]], fascination_models[[5]]))
  fascination_semseg_final <- fascination_models[[3]]
  print_model_summary(fascination_semseg_final, "Fascination ~ SemSeg")
  
  # Stress
  agg_stress <- prepare_question_data(combined_data, "likert-stress")
  stress_models <- fit_semseg_models(agg_stress)
  print(anova(stress_models[[1]], stress_models[[2]], stress_models[[3]], 
              stress_models[[4]], stress_models[[5]]))
  stress_semseg_final <- stress_models[[3]]
  print_model_summary(stress_semseg_final, "Stress ~ SemSeg")
}

# ==============================================================================
# SECTION 2: ERP COMPONENTS ~ SEMANTIC SEGMENTATION
# ==============================================================================

if (RUN_SEMSEG_ERP) {
  
  # Prepare ERP dataset
  erp_per_sc <- combined_data %>%
    mutate(
      P1_cluster = rowMeans(across(c(
        amplitude_O1_145_185, amplitude_O2_145_185,
        amplitude_Oz_145_185, amplitude_POz_145_185
      )), na.rm = TRUE),
      N1_cluster = rowMeans(across(c(
        amplitude_O1_192_232, amplitude_O2_192_232,
        amplitude_Oz_192_232, amplitude_POz_192_232
      )), na.rm = TRUE)
    ) %>%
    group_by(Subject, Condition, Stimulus) %>%
    summarise(
      mean_P3 = if (all(is.na(ParietalP3))) NA_real_ else mean(ParietalP3, na.rm = TRUE),
      mean_LPP = if (all(is.na(ParietalLPP))) NA_real_ else mean(ParietalLPP, na.rm = TRUE),
      mean_P1 = if (all(is.na(P1_cluster))) NA_real_ else mean(P1_cluster, na.rm = TRUE),
      mean_N1 = if (all(is.na(N1_cluster))) NA_real_ else mean(N1_cluster, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Add features
  erp_per_sc <- erp_per_sc %>%
    left_join(
      combined_data %>%
        select(Subject, Condition, Stimulus, sky, car, person, green_imperv) %>%
        distinct(),
      by = c("Subject", "Condition", "Stimulus")
    ) %>%
    left_join(
      combined_data %>%
        select(Subject, Gender_binary, City_15, extrov, emostab) %>%
        distinct(),
      by = "Subject"
    ) %>%
    distinct()
  
  # Scale predictors
  erp_per_sc <- erp_per_sc %>%
    mutate(
      z_green_imperv = as.numeric(scale(green_imperv)),
      z_sky = as.numeric(scale(sky)),
      z_car = as.numeric(scale(car)),
      z_person = as.numeric(scale(person)),
      Gender_binary = relevel(as.factor(Gender_binary), ref = "Male")
    )
  
  # P1 Component ---------------------------------------------------------------
  P1_semseg_models <- fit_semseg_models(erp_per_sc, "mean_P1")
  print(anova(P1_semseg_models[[1]], P1_semseg_models[[2]], P1_semseg_models[[3]], 
              P1_semseg_models[[4]], P1_semseg_models[[5]]))
  P1_semseg_final <- P1_semseg_models[[3]]
  print_model_summary(P1_semseg_final, "P1 ~ SemSeg")
  
  # N1 Component ---------------------------------------------------------------
  N1_semseg_models <- fit_semseg_models(erp_per_sc, "mean_N1")
  print(anova(N1_semseg_models[[1]], N1_semseg_models[[2]], N1_semseg_models[[3]], 
              N1_semseg_models[[4]], N1_semseg_models[[5]]))
  N1_semseg_final <- N1_semseg_models[[3]]
  print_model_summary(N1_semseg_final, "N1 ~ SemSeg")
  
  # P3 Component ---------------------------------------------------------------
  P3_semseg_models <- fit_semseg_models(erp_per_sc, "mean_P3")
  print(anova(P3_semseg_models[[1]], P3_semseg_models[[2]], P3_semseg_models[[3]], 
              P3_semseg_models[[4]], P3_semseg_models[[5]]))
  P3_semseg_final <- P3_semseg_models[[2]]
  print_model_summary(P3_semseg_final, "P3 ~ SemSeg")
  
  # LPP Component --------------------------------------------------------------
  LPP_semseg_models <- fit_semseg_models(erp_per_sc, "mean_LPP")
  print(anova(LPP_semseg_models[[1]], LPP_semseg_models[[2]], LPP_semseg_models[[3]], 
              LPP_semseg_models[[4]], LPP_semseg_models[[5]]))
  LPP_semseg_final <- LPP_semseg_models[[2]]
  print_model_summary(LPP_semseg_final, "LPP ~ SemSeg")
  
}

# ==============================================================================
# SECTION 3: SUBJECTIVE RATINGS ~ LLVF FEATURES
# ==============================================================================

if (RUN_LLVF_SUBJECTIVE) {

  # SAM Scales -----------------------------------------------------------------

  # Arousal
  agg_arousal_llvf <- prepare_llvf_data(combined_data, "sam-arousal", invert_scale = TRUE)
  arousal_llvf_models <- fit_llvf_models(agg_arousal_llvf)
  print(anova(arousal_llvf_models[[1]], arousal_llvf_models[[2]], arousal_llvf_models[[3]], 
              arousal_llvf_models[[4]], arousal_llvf_models[[5]]))
  arousal_llvf_final <- arousal_llvf_models[[5]]
  print_model_summary(arousal_llvf_final, "Arousal ~ LLVF")
  
  # Valence
  agg_valence_llvf <- prepare_llvf_data(combined_data, "sam-valence", invert_scale = TRUE)
  valence_llvf_models <- fit_llvf_models(agg_valence_llvf)
  print(anova(valence_llvf_models[[1]], valence_llvf_models[[2]], valence_llvf_models[[3]], 
              valence_llvf_models[[4]], valence_llvf_models[[5]]))
  valence_llvf_final <- valence_llvf_models[[5]]
  print_model_summary(valence_llvf_final, "Valence ~ LLVF")
  
  # Dominance
  agg_dominance_llvf <- prepare_llvf_data(combined_data, "sam-dominance", invert_scale = TRUE)
  dominance_llvf_models <- fit_llvf_models(agg_dominance_llvf)
  print(anova(dominance_llvf_models[[1]], dominance_llvf_models[[2]], dominance_llvf_models[[3]], 
              dominance_llvf_models[[4]], dominance_llvf_models[[5]]))
  dominance_llvf_final <- dominance_llvf_models[[5]]
  print_model_summary(dominance_llvf_final, "Dominance ~ LLVF")
  
  # Likert Scales --------------------------------------------------------------

  # Safety
  agg_safety_llvf <- prepare_llvf_data(combined_data, "likert-sicherheit")
  safety_llvf_models <- fit_llvf_models(agg_safety_llvf)
  print(anova(safety_llvf_models[[1]], safety_llvf_models[[2]], safety_llvf_models[[3]], 
              safety_llvf_models[[4]], safety_llvf_models[[5]]))
  safety_llvf_final <- safety_llvf_models[[5]]
  print_model_summary(safety_llvf_final, "Safety ~ LLVF")
  
  # Beauty
  agg_beauty_llvf <- prepare_llvf_data(combined_data, "likert-schonheit")
  beauty_llvf_models <- fit_llvf_models(agg_beauty_llvf)
  print(anova(beauty_llvf_models[[1]], beauty_llvf_models[[2]], beauty_llvf_models[[3]], 
              beauty_llvf_models[[4]], beauty_llvf_models[[5]]))
  beauty_llvf_final <- beauty_llvf_models[[5]]
  print_model_summary(beauty_llvf_final, "Beauty ~ LLVF")
  
  # Hominess
  agg_hominess_llvf <- prepare_llvf_data(combined_data, "likert-heimeligkeit")
  hominess_llvf_models <- fit_llvf_models(agg_hominess_llvf)
  print(anova(hominess_llvf_models[[1]], hominess_llvf_models[[2]], hominess_llvf_models[[3]], 
              hominess_llvf_models[[4]], hominess_llvf_models[[5]]))
  hominess_llvf_final <- hominess_llvf_models[[5]]
  print_model_summary(hominess_llvf_final, "Hominess ~ LLVF")
  
  # Openness
  agg_openness_llvf <- prepare_llvf_data(combined_data, "likert-offenheit")
  openness_llvf_models <- fit_llvf_models(agg_openness_llvf)
  print(anova(openness_llvf_models[[1]], openness_llvf_models[[2]], openness_llvf_models[[3]], 
              openness_llvf_models[[4]], openness_llvf_models[[5]]))
  openness_llvf_final <- openness_llvf_models[[5]]
  print_model_summary(openness_llvf_final, "Openness ~ LLVF")
  
  # Fascination
  agg_fascination_llvf <- prepare_llvf_data(combined_data, "likert-faszination")
  fascination_llvf_models <- fit_llvf_models(agg_fascination_llvf)
  print(anova(fascination_llvf_models[[1]], fascination_llvf_models[[2]], fascination_llvf_models[[3]], 
              fascination_llvf_models[[4]], fascination_llvf_models[[5]]))
  fascination_llvf_final <- fascination_llvf_models[[3]]
  print_model_summary(fascination_llvf_final, "Fascination ~ LLVF")
  
  # Stress
  agg_stress_llvf <- prepare_llvf_data(combined_data, "likert-stress")
  stress_llvf_models <- fit_llvf_models(agg_stress_llvf)
  print(anova(stress_llvf_models[[1]], stress_llvf_models[[2]], stress_llvf_models[[3]], 
              stress_llvf_models[[4]], stress_llvf_models[[5]]))
  stress_llvf_final <- stress_llvf_models[[5]]
  print_model_summary(stress_llvf_final, "Stress ~ LLVF")
  }

# ==============================================================================
# SECTION 4: ERP COMPONENTS ~ LLVF FEATURES
# ==============================================================================

if (RUN_LLVF_ERP) {

  # Create clusters if not already done
  if (!exists("erp_per_sc")) {
    erp_per_sc <- combined_data %>%
      mutate(
        P1_cluster = rowMeans(across(c(
          amplitude_O1_145_185, amplitude_O2_145_185,
          amplitude_Oz_145_185, amplitude_POz_145_185
        )), na.rm = TRUE),
        N1_cluster = rowMeans(across(c(
          amplitude_O1_192_232, amplitude_O2_192_232,
          amplitude_Oz_192_232, amplitude_POz_192_232
        )), na.rm = TRUE)
      ) %>%
      group_by(Subject, Condition, Stimulus) %>%
      summarise(
        mean_P3 = if (all(is.na(ParietalP3))) NA_real_ else mean(ParietalP3, na.rm = TRUE),
        mean_LPP = if (all(is.na(ParietalLPP))) NA_real_ else mean(ParietalLPP, na.rm = TRUE),
        mean_P1 = if (all(is.na(P1_cluster))) NA_real_ else mean(P1_cluster, na.rm = TRUE),
        mean_N1 = if (all(is.na(N1_cluster))) NA_real_ else mean(N1_cluster, na.rm = TRUE),
        .groups = "drop"
      )
  }
  
  # Add LLVF features
  llvf_src <- combined_data %>%
    select(Subject, Condition, gvHue, rcHue, HSVHue, Brightness, sdBrightness,
           Saturation, sdSaturation, StraightEdgeDensity, NonStraightEdgeDensity,
           NonStraightEdgeDensityWithGreen, Entropy, FractalD,
           depth_metric_mean, depth_metric_sd, GreenMaskWThresholdedDensity) %>%
    distinct()
  
  dup_llvf <- llvf_src %>% count(Subject, Condition) %>% filter(n > 1)
  if (nrow(dup_llvf) > 0) stop("Duplicate keys in LLVF features")
  
  erp_per_sc_llvf <- erp_per_sc %>%
    left_join(llvf_src, by = c("Subject", "Condition"))
  
  # Add personal predictors if not present
  if (!"Gender_binary" %in% names(erp_per_sc_llvf)) {
    erp_per_sc_llvf <- erp_per_sc_llvf %>%
      left_join(
        combined_data %>%
          select(Subject, Gender_binary, City_15, extrov, emostab) %>%
          distinct(),
        by = "Subject"
      )
  }
  
  # Scale LLVF predictors
  erp_per_sc_llvf <- erp_per_sc_llvf %>%
    mutate(
      z_green_mask_density = as.numeric(scale(GreenMaskWThresholdedDensity)),
      z_gvHue = as.numeric(scale(gvHue)),
      z_Saturation = as.numeric(scale(Saturation)),
      z_sdSaturation = as.numeric(scale(sdSaturation)),
      z_StraightEdgeD = as.numeric(scale(StraightEdgeDensity)),
      z_NonStraightED = as.numeric(scale(NonStraightEdgeDensity)),
      z_Entropy = as.numeric(scale(Entropy)),
      z_FractalD = as.numeric(scale(FractalD)),
      z_depth_metric_mean = as.numeric(scale(depth_metric_mean)),
      Gender_binary = relevel(as.factor(Gender_binary), ref = "Male")
    )
  
  # P1 Component ---------------------------------------------------------------
  P1_llvf_models <- fit_llvf_models(erp_per_sc_llvf, "mean_P1")
  print(anova(P1_llvf_models[[1]], P1_llvf_models[[2]], P1_llvf_models[[3]], 
              P1_llvf_models[[4]], P1_llvf_models[[5]]))
  P1_llvf_final <- P1_llvf_models[[2]]
  print_model_summary(P1_llvf_final, "P1 ~ LLVF")
  
  # N1 Component ---------------------------------------------------------------
  N1_llvf_models <- fit_llvf_models(erp_per_sc_llvf, "mean_N1")
  print(anova(N1_llvf_models[[1]], N1_llvf_models[[2]], N1_llvf_models[[3]], 
              N1_llvf_models[[4]], N1_llvf_models[[5]]))
  N1_llvf_final <- N1_llvf_models[[2]]
  print_model_summary(N1_llvf_final, "N1 ~ LLVF")
  
  # P3 Component ---------------------------------------------------------------
  P3_llvf_models <- fit_llvf_models(erp_per_sc_llvf, "mean_P3")
  print(anova(P3_llvf_models[[1]], P3_llvf_models[[2]], P3_llvf_models[[3]], 
              P3_llvf_models[[4]], P3_llvf_models[[5]]))
  P3_llvf_final <- P3_llvf_models[[2]]
  print_model_summary(P3_llvf_final, "P3 ~ LLVF")
  
  # LPP Component --------------------------------------------------------------
  LPP_llvf_models <- fit_llvf_models(erp_per_sc_llvf, "mean_LPP")
  print(anova(LPP_llvf_models[[1]], LPP_llvf_models[[2]], LPP_llvf_models[[3]], 
              LPP_llvf_models[[4]], LPP_llvf_models[[5]]))
  LPP_llvf_final <- LPP_llvf_models[[2]]
  print_model_summary(LPP_llvf_final, "LPP ~ LLVF")
}

# ==============================================================================
# EXPORT TABLES
# ==============================================================================
# library(sjPlot)
 
# if (RUN_SEMSEG_SUBJECTIVE) {
#   tab_model(arousal_semseg_final, valence_semseg_final, dominance_semseg_final,
#             safety_semseg_final, beauty_semseg_final, hominess_semseg_final,
#             openness_semseg_final, fascination_semseg_final, stress_semseg_final,
#             show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE,
#             file = file.path(OUTPUT_DIR, "semseg_subjective_models.doc"))
# }
# 
# if (RUN_SEMSEG_ERP) {
#   tab_model(P1_semseg_final, N1_semseg_final, P3_semseg_final, LPP_semseg_final,
#             show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE,
#             file = file.path(OUTPUT_DIR, "semseg_erp_models.doc"))
# }
# 
# if (RUN_LLVF_SUBJECTIVE) {
#   tab_model(arousal_llvf_final, valence_llvf_final, dominance_llvf_final,
#             safety_llvf_final, beauty_llvf_final, hominess_llvf_final,
#             openness_llvf_final, fascination_llvf_final, stress_llvf_final,
#             show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE,
#             file = file.path(OUTPUT_DIR, "llvf_subjective_models.doc"))
# }
# 
# if (RUN_LLVF_ERP) {
#   tab_model(P1_llvf_final, N1_llvf_final, P3_llvf_final, LPP_llvf_final,
#             show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE,
#             file = file.path(OUTPUT_DIR, "llvf_erp_models.doc"))