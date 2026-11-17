# 只使用合成数据验证 WeMix 两层权重输入，不读取任何真实研究数据。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("需要一个 JSON 输出路径")
}

suppressPackageStartupMessages(library(WeMix))
suppressPackageStartupMessages(library(lme4))

set.seed(20260913)
participant_n <- 60L
visit_n <- rep(2:5, length.out = participant_n)
participant <- factor(rep(seq_len(participant_n), times = visit_n))
time <- unlist(lapply(visit_n, function(n) seq.int(0L, n - 1L)), use.names = FALSE)
person_weight <- seq(0.45, 2.35, length.out = participant_n)
weight_raw <- rep(person_weight, times = visit_n)
random_intercept <- rnorm(participant_n, sd = 0.70)
x <- rnorm(length(participant))
y <- 1.10 + 0.35 * x + 0.06 * time +
  random_intercept[as.integer(participant)] + rnorm(length(participant), sd = 0.60)

synthetic <- data.frame(
  participant = participant,
  time = time,
  x = x,
  y = y,
  observation_conditional_weight = 1,
  participant_weight = weight_raw,
  observation_unconditional_weight = weight_raw,
  unit_weight = 1
)

formula <- y ~ x + time + (1 | participant)
conditional <- mix(
  formula,
  data = synthetic,
  weights = c("observation_conditional_weight", "participant_weight"),
  cWeights = TRUE,
  max_iteration = 50
)
unconditional <- mix(
  formula,
  data = synthetic,
  weights = c("observation_unconditional_weight", "participant_weight"),
  cWeights = FALSE,
  max_iteration = 50
)
unit_wemix <- mix(
  formula,
  data = synthetic,
  weights = c("unit_weight", "unit_weight"),
  cWeights = TRUE,
  max_iteration = 50
)
unit_lme4 <- lmer(formula, data = synthetic, REML = FALSE)

person_weight_scaled <- person_weight / mean(person_weight)
synthetic$participant_weight_scaled <- rep(person_weight_scaled, times = visit_n)
scaled <- mix(
  formula,
  data = synthetic,
  weights = c("observation_conditional_weight", "participant_weight_scaled"),
  cWeights = TRUE,
  max_iteration = 50
)

max_abs <- function(left, right) {
  max(abs(as.numeric(left) - as.numeric(right)))
}

conditional_unconditional_coef_max_abs <- max_abs(conditional$coef, unconditional$coef)
conditional_unconditional_var_max_abs <- max_abs(conditional$vars, unconditional$vars)
unit_lme4_coef_max_abs <- max_abs(unit_wemix$coef, fixef(unit_lme4))
unit_lme4_sigma_abs <- abs(unit_wemix$sigma - sigma(unit_lme4))

if (conditional_unconditional_coef_max_abs > 1e-8) {
  stop("conditional 与等价 unconditional 的固定效应不一致")
}
if (conditional_unconditional_var_max_abs > 1e-8) {
  stop("conditional 与等价 unconditional 的方差分量不一致")
}
if (unit_lme4_coef_max_abs > 1e-8 || unit_lme4_sigma_abs > 1e-8) {
  stop("unit-weight WeMix 与 lme4 ML 对账失败")
}

result <- data.frame(
  metric = c(
    "status",
    "generated_utc",
    "WeMix_version",
    "lme4_version",
    "R_version",
    "participant_n",
    "observation_n",
    "minimum_visits",
    "maximum_visits",
    "unequal_followup_present",
    "participant_weight_divided_by_visit_count",
    "conditional_observation_level",
    "conditional_participant_level",
    "conditional_cWeights",
    "unconditional_observation_level",
    "unconditional_participant_level",
    "unconditional_cWeights",
    "conditional_unconditional_coef_max_abs",
    "conditional_unconditional_var_max_abs",
    "unit_wemix_lme4_coef_max_abs",
    "unit_wemix_lme4_sigma_abs",
    "raw_participant_weight_mean",
    "scaled_participant_weight_mean",
    "raw_scaled_fixed_effect_max_abs_difference",
    "raw_scaled_variance_component_max_abs_difference",
    "raw_scaled_standard_error_max_abs_difference",
    "raw_scaled_objective_difference",
    "robust_score_top_level",
    "limitations"
  ),
  value = c(
    "PASS_SYNTHETIC_SEMANTICS_ONLY",
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    as.character(packageVersion("WeMix")),
    as.character(packageVersion("lme4")),
    R.version.string,
    participant_n,
    nrow(synthetic),
    min(visit_n),
    max(visit_n),
    length(unique(visit_n)) > 1L,
    FALSE,
    "1",
    "w_i",
    TRUE,
    "w_i repeated on that participant's rows",
    "w_i",
    FALSE,
    conditional_unconditional_coef_max_abs,
    conditional_unconditional_var_max_abs,
    unit_lme4_coef_max_abs,
    unit_lme4_sigma_abs,
    mean(person_weight),
    mean(person_weight_scaled),
    max_abs(conditional$coef, scaled$coef),
    max_abs(conditional$vars, scaled$vars),
    max_abs(conditional$SE, scaled$SE),
    as.numeric(conditional$lnl - scaled$lnl),
    "participant in this two-level synthetic model",
    paste(
      "Does not validate household dependence; does not validate PSU or strata",
      "variance estimation; does not lock scaling or a final model implementation"
    )
  ),
  stringsAsFactors = FALSE
)

write.table(
  result,
  file = args[[1]],
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = TRUE,
  fileEncoding = "UTF-8"
)
