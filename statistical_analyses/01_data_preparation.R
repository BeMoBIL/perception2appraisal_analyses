# ==============================================================================
# Script: 01_data_preparation.R
# Purpose: Merge ERP peak data, subjective ratings, sociodemographic data,
#          and semantic segmentation data into a comprehensive dataset
# ==============================================================================

# Load required packages -------------------------------------------------------
library(dplyr)
library(readr)
library(readxl)
library(tidyr)

# Configuration ----------------------------------------------------------------
# Set base paths (modify these to match your directory structure)
BASE_DIR <- "/path/to/project" 
ERP_DIR <- file.path(BASE_DIR, "peaks")
RATING_DIR <- file.path(BASE_DIR, "subjective_ratings")
SOCIODEM_FILE <- file.path(BASE_DIR, "sociodemographics", "final_sociodem.xlsx")
SEGMENTATION_FILE <- file.path(BASE_DIR, "semantic_segmentation", "segmentation_pixel_sums_incl.csv")
OUTPUT_FILE <- file.path(BASE_DIR, "combined_data_full.csv")

# Define ERP peak files to load
ERP_FILES <- c(
  "peak_CPz_404_444.csv", "peak_Pz_404_444.csv", "peak_P3_404_444.csv", "peak_P4_404_444.csv",
  "peak_CPz_575_615.csv", "peak_O1_145_185.csv", "peak_O1_192_232.csv", "peak_O2_145_185.csv",
  "peak_O2_192_232.csv", "peak_Oz_145_185.csv", "peak_Oz_192_232.csv", "peak_P3_575_615.csv",
  "peak_P4_575_615.csv", "peak_POz_145_185.csv", "peak_POz_192_232.csv", "peak_Pz_575_615.csv"
)

# Helper functions -------------------------------------------------------------

#' Standardize string format (lowercase and trimmed)
#' @param x Character vector
#' @return Standardized character vector
clean_str <- function(x) {
  trimws(tolower(as.character(x)))
}

# Question label mapping (standardizes German/English labels) -----------------
QUESTION_MAPPING <- c(
  "likert-sch" = "likert-schonheit",
  "likert-beauty" = "likert-schonheit",
  "likert-schonheit" = "likert-schonheit",
  "likert-faszination" = "likert-faszination",
  "likert-fascination" = "likert-faszination",
  "likert-heimeligkeit" = "likert-heimeligkeit",
  "likert-hominess" = "likert-heimeligkeit",
  "likert-offenheit" = "likert-offenheit",
  "likert-openness" = "likert-offenheit",
  "likert-sicherheit" = "likert-sicherheit",
  "likert-safety" = "likert-sicherheit",
  "likert-stress" = "likert-stress",
  "sam-arousal" = "sam-arousal",
  "sam-dominance" = "sam-dominance",
  "sam-valence" = "sam-valence"
)

# Load and process ERP data ----------------------------------------------------
erp_data_list <- lapply(ERP_FILES, function(f) {
  df <- read_csv(file.path(ERP_DIR, f), show_col_types = FALSE) %>%
    mutate(
      Subject = as.numeric(Subject),
      Condition = clean_str(Condition),
      Question = clean_str(Question)
    )
  
  # Create descriptive column names from filename
  base_name <- gsub("peak_|\\.csv", "", f)
  df %>%
    rename_with(~ paste0("amplitude_", base_name), starts_with("PeakAmplitude")) %>%
    rename_with(~ paste0("latency_", base_name), starts_with("PeakLatency"))
})

# Merge all ERP datasets
erp_data <- Reduce(
  function(x, y) full_join(x, y, by = c("Subject", "Condition", "Question")),
  erp_data_list
) %>%
  group_by(Subject, Condition, Question) %>%
  summarise(
    across(starts_with("amplitude_"), ~ mean(.x, na.rm = TRUE)),
    across(starts_with("latency_"), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

# Standardize question labels
erp_data <- erp_data %>%
  mutate(Question = dplyr::recode(Question, !!!QUESTION_MAPPING, .default = Question))

# Load and process rating data -------------------------------------------------
rating_files <- list.files(RATING_DIR, pattern = "^results_sub-.*\\.xlsx$", full.names = TRUE)
rating_data <- bind_rows(lapply(rating_files, read_xlsx)) %>%
  rename(
    Subject = participantID,
    Condition = stimulusID,
    Question = scaleType,
    Rating = response
  ) %>%
  mutate(
    Subject = as.numeric(gsub("sub-", "", Subject)),
    Condition = clean_str(Condition),
    Question = clean_str(Question)
  )

# Standardize question labels
rating_data <- rating_data %>%
  mutate(Question = dplyr::recode(Question, !!!QUESTION_MAPPING, .default = Question))

# Load sociodemographic data ---------------------------------------------------
sociodem_data <- read_xlsx(SOCIODEM_FILE) %>%
  mutate(Subject_Code = as.numeric(Subject_Code)) %>%
  filter(!is.na(Subject_Code)) %>%
  distinct(Subject_Code, .keep_all = TRUE)


# Load semantic segmentation data ----------------------------------------------
segmentation_data <- read_delim(SEGMENTATION_FILE, delim = ";", show_col_types = FALSE) %>%
  mutate(StimulusId = clean_str(StimulusId))


# Create complete dataset structure --------------------------------------------
# Complete ERP grid (all combinations of Subject x Condition x Question)
erp_complete <- erp_data %>%
  complete(
    Subject = unique(rating_data$Subject),
    Condition = unique(rating_data$Condition),
    Question = unique(rating_data$Question)
  )

# Merge all data sources
combined_full <- erp_complete %>%
  left_join(sociodem_data, by = c("Subject" = "Subject_Code")) %>%
  left_join(segmentation_data, by = c("Condition" = "StimulusId")) %>%
  left_join(rating_data, by = c("Subject", "Condition", "Question")) %>%
  distinct()

# Save output ------------------------------------------------------------------
write_csv(combined_full, OUTPUT_FILE)