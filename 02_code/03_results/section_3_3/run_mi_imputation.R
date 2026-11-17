# 按冻结predictor matrix执行participant-level多重插补；completed records仅写入临时目录。
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) stop("需要input CSV、output CSV、cohort和seed")
suppressPackageStartupMessages(library(mice))

input_path <- args[[1]]
output_path <- args[[2]]
cohort <- args[[3]]
seed <- as.integer(args[[4]])
d <- read.csv(input_path, check.names = FALSE, stringsAsFactors = FALSE,
              na.strings = c("", "NA"), colClasses = c(person_id = "character"))

d$sex <- factor(d$sex, levels = c("Male", "Female"))
d$education <- factor(d$education, levels = c("High", "Medium", "Low"))
d$marital_status <- factor(d$marital_status, levels = c("Married/partnered", "Separated/divorced", "Widowed", "Never married"))
d$labour_force_status <- factor(d$labour_force_status, levels = c("Working", "Unemployed", "Retired or partly retired", "Other economically inactive"))
d$current_smoking <- factor(d$current_smoking, levels = c("False", "True"))
d$wealth_quintile <- factor(d$wealth_quintile, levels = c("5", "1", "2", "3", "4"))
d$recovery <- as.numeric(d$recovery)

targets <- c("fi_t", "education", "marital_status", "current_smoking", "labour_force_status")
methods <- rep("", ncol(d)); names(methods) <- names(d)
requested <- c(fi_t = "pmm", education = "polyreg", marital_status = "polyreg", current_smoking = "logreg", labour_force_status = "polyreg")
for (name in targets) if (anyNA(d[[name]])) methods[[name]] <- requested[[name]]

predictor <- matrix(0L, nrow = ncol(d), ncol = ncol(d), dimnames = list(names(d), names(d)))
excluded_predictors <- c("person_id", "wealth_loss_fraction")
for (target in targets) {
  if (methods[[target]] != "") {
    predictors <- setdiff(names(d), c(target, excluded_predictors))
    predictor[target, predictors] <- 1L
  }
}

visit <- mice(d, m = 50L, maxit = 20L, method = methods, predictorMatrix = predictor,
              seed = seed, printFlag = FALSE, donors = 5L,
              remove.collinear = FALSE, remove.constant = FALSE)
completed <- complete(visit, action = "long", include = FALSE)
write.csv(completed, output_path, row.names = FALSE, fileEncoding = "UTF-8")

diagnostics <- data.frame(
  cohort = cohort,
  seed = seed,
  imputations = visit$m,
  iterations = visit$iteration,
  logged_events = if (is.null(visit$loggedEvents)) 0L else nrow(visit$loggedEvents),
  method_fi_t = methods[["fi_t"]],
  method_education = methods[["education"]],
  method_marital_status = methods[["marital_status"]],
  method_current_smoking = methods[["current_smoking"]],
  method_labour_force_status = methods[["labour_force_status"]],
  predictor_matrix_nonzero = sum(predictor != 0L),
  stringsAsFactors = FALSE
)
write.csv(diagnostics, paste0(output_path, ".diagnostics.csv"), row.names = FALSE, fileEncoding = "UTF-8")
