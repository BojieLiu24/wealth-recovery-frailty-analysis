# 使用冻结的Python预处理模型矩阵输入拟合正式WeMix加权敏感性模型。
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("需要输入CSV与输出JSON路径")

suppressPackageStartupMessages(library(WeMix))

d <- read.csv(args[[1]], check.names = FALSE, stringsAsFactors = FALSE)
d$person_id <- factor(d$person_id)
d$sex <- relevel(factor(d$sex), ref = "Male")
d$education <- relevel(factor(d$education), ref = "High")
d$marital_status <- relevel(factor(d$marital_status), ref = "Married/partnered")
d$labour_force_status <- relevel(factor(d$labour_force_status), ref = "Working")
d$current_smoking <- relevel(factor(d$current_smoking), ref = "False")
d$wealth_quintile <- relevel(factor(d$wealth_quintile), ref = "5")

form <- fi_primary ~ recovery * time_years + age_z + household_size_z +
  household_income_rank_z + wealth_loss_fraction_z + fi_t_z + sex + education +
  marital_status + labour_force_status + current_smoking + wealth_quintile +
  age_z:time_years + household_size_z:time_years + household_income_rank_z:time_years +
  wealth_loss_fraction_z:time_years + fi_t_z:time_years + sex:time_years +
  education:time_years + marital_status:time_years + labour_force_status:time_years +
  current_smoking:time_years + wealth_quintile:time_years + (1 | person_id)

captured <- character()
fit <- withCallingHandlers(
  WeMix::mix(form, data = d,
             weights = c("level1_weight", "participant_weight_scaled"),
             cWeights = TRUE, max_iteration = 200L),
  warning = function(w) {
    captured <<- c(captured, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
term <- "recovery:time_years"
if (!(term %in% names(fit$coef)) || !(term %in% names(fit$SE))) stop("找不到recovery:time_years")
beta <- unname(fit$coef[[term]])
se <- unname(fit$SE[[term]])
if (!is.finite(beta) || !is.finite(se) || se <= 0) stop("WeMix估计或CR0 SE无效")
if (max(abs(fit$SE - sqrt(diag(fit$cov_mat)))) > 1e-10) stop("CR0 SE与cov_mat不一致")

out <- data.frame(
  status = "PASS",
  estimate = beta,
  se = se,
  ci_lower = beta - qnorm(0.975) * se,
  ci_upper = beta + qnorm(0.975) * se,
  participants = length(unique(d$person_id)),
  fi_observations = nrow(d),
  fixed_effect_count = length(fit$coef),
  coefficient_names = paste(names(fit$coef), collapse = "|"),
  warning_messages = paste(captured, collapse = "|"),
  scaled_weight_mean_unique = mean(tapply(d$participant_weight_scaled, d$person_id, unique)),
  scaled_weight_min = min(d$participant_weight_scaled),
  scaled_weight_max = max(d$participant_weight_scaled),
  WeMix_version = as.character(packageVersion("WeMix")),
  R_version = R.version.string,
  covariance = "participant_level_CR0",
  secondary_meta = FALSE,
  stringsAsFactors = FALSE
)
write.csv(out, args[[2]], row.names = FALSE, fileEncoding = "UTF-8")
