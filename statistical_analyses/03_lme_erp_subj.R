## Analysis Script for LMMs using ERP amplitudes to predict Subjective Ratings


# Read dataset
data_LMMs_erps_subj =  read.csv("~/data_LMMs_erps_subj.csv")


# Fit 1 LMM per rating scale
m1 = lmer(dominance ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
            +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)

m2  = lmer(arousal ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
             +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)

m3 = lmer(stress ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
            +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)
m4 = lmer(valence ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
            +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)

m5 = lmer(beauty ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
            +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)

m6  = lmer(fascination ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
             +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)

m7 = lmer(`likert-heimeligkeit` ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
            +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)

m8 = lmer(open ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
            +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)

m9 = lmer(safety ~ ParietalLPP+ParietalP3+N1+P1 + Gender_binary + City_15 + extrov + emostab +
            +                  (1 | Condition) + (1 | Subject), data =mediation_data_sub, REML = TRUE)


# Define fuunction to extract p-values for FDR correction
extract_pvals <- function(model) {
  coefs <- summary(model)$coefficients
  # Remove intercept row
  coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
  # Extract p-values
  pvals <- coefs[, "Pr(>|t|)"]
  return(pvals)
}

# Apply to each model
p_list <- list(
  m1 = extract_pvals(m1),
  m2 = extract_pvals(m2),
  m3 = extract_pvals(m3),
  m4 = extract_pvals(m4),
  m5 = extract_pvals(m5),
  m6 = extract_pvals(m6),
  m7 = extract_pvals(m7),
  m8 = extract_pvals(m8),
  m9 = extract_pvals(m9)
)

# Bind into a dataframe
p_df <- do.call(rbind, lapply(names(p_list), function(name) {
  data.frame(
    Model = name,
    Term = names(p_list[[name]]),
    Raw_p = p_list[[name]],
    stringsAsFactors = FALSE
  )
}))

# Add FDR adjusted p values to df
p_df$FDR_p <- p.adjust(p_df$Raw_p, method = "fdr")

#Export models in table
tab_model(m2,m4,m1,m3,m9,m5,m7,m8,m6, show.se = TRUE, p.adjust = "fdr", string.se = "SE", file = "erp_ratings_models.html")