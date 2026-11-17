# 仅提取指定结构阶段所需 Stata 列，并显式保存 tagged missing。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
  stop("用法：extract_spine_source.R <dta> <mapping_csv> <cohort> <domains> <parquet_out> <schema_csv_out>")
}

dta_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
mapping_path <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
cohort <- args[[3]]
domains <- strsplit(args[[4]], ";", fixed = TRUE)[[1]]
parquet_out <- args[[5]]
schema_out <- args[[6]]

if (!requireNamespace("haven", quietly = TRUE)) stop("缺少已锁定的 haven 包")
if (!requireNamespace("arrow", quietly = TRUE)) stop("缺少已锁定的 arrow 包")
if (!requireNamespace("tidyselect", quietly = TRUE)) stop("缺少 haven 所需的 tidyselect 包")

mapping <- read.csv(mapping_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
mapping <- mapping[mapping$cohort == cohort & mapping$domain %in% domains, , drop = FALSE]
if (nrow(mapping) == 0L) stop("找不到 cohort/domain mapping：", cohort, "/", paste(domains, collapse = ";"))
if (any(mapping$verification_status != "verified")) {
  stop("结构 mapping 含未 verified 的记录：", cohort)
}

expand_template <- function(template, wave_min, wave_max) {
  if (grepl("\\{wave\\}", template)) {
    return(vapply(
      seq.int(wave_min, wave_max),
      function(wave) gsub("\\{wave\\}", as.character(wave), template),
      character(1)
    ))
  }
  template
}

variables <- unique(unlist(Map(
  expand_template,
  mapping$source_variable_template,
  mapping$wave_min,
  mapping$wave_max
)))

data <- haven::read_dta(dta_path, col_select = tidyselect::all_of(variables))
missing_variables <- setdiff(variables, names(data))
if (length(missing_variables) > 0L) {
  stop("DTA 缺少已验证的结构变量：", paste(missing_variables, collapse = ", "))
}

escape_json <- function(value) {
  value <- gsub("\\\\", "\\\\\\\\", value)
  value <- gsub('"', '\\"', value, fixed = TRUE)
  value <- gsub("\r", "\\r", value, fixed = TRUE)
  value <- gsub("\n", "\\n", value, fixed = TRUE)
  value
}

label_value_text <- function(value) {
  if (haven::is_tagged_na(value)) return(paste0(".", haven::na_tag(value)))
  if (is.na(value)) return("system_na")
  format(value, scientific = FALSE, trim = TRUE)
}

value_labels_json <- function(x) {
  labels <- attr(x, "labels", exact = TRUE)
  if (is.null(labels) || length(labels) == 0L) return("[]")
  entries <- vapply(seq_along(labels), function(index) {
    paste0(
      '{"value":"', escape_json(label_value_text(labels[[index]])),
      '","label":"', escape_json(names(labels)[[index]]), '"}'
    )
  }, character(1))
  paste0("[", paste(entries, collapse = ","), "]")
}

encoded <- list()
schema <- vector("list", length(variables))
for (index in seq_along(variables)) {
  variable <- variables[[index]]
  original <- data[[variable]]
  tagged <- if (is.numeric(original)) haven::is_tagged_na(original) else rep(FALSE, length(original))
  ordinary_missing <- is.na(original) & !tagged
  missing_tag <- rep(NA_character_, length(original))
  if (any(tagged)) {
    missing_tag[tagged] <- vapply(
      original[tagged],
      function(value) as.character(haven::na_tag(value)),
      character(1)
    )
  }
  missing_tag[ordinary_missing] <- "system_na"

  value <- haven::zap_labels(original)
  value[is.na(value)] <- NA
  encoded[[paste0(variable, "__value")]] <- value
  encoded[[paste0(variable, "__missing_tag")]] <- missing_tag

  tags_observed <- sort(unique(missing_tag[!is.na(missing_tag) & missing_tag != "system_na"]))
  mapping_row <- mapping[vapply(
    seq_len(nrow(mapping)),
    function(row_index) {
      variable %in% expand_template(
        mapping$source_variable_template[[row_index]],
        mapping$wave_min[[row_index]],
        mapping$wave_max[[row_index]]
      )
    },
    logical(1)
  ), , drop = FALSE]
  schema[[index]] <- data.frame(
    cohort = cohort,
    source_variable = variable,
    storage_type = typeof(original),
    stata_variable_label = ifelse(is.null(attr(original, "label", exact = TRUE)), "", attr(original, "label", exact = TRUE)),
    value_label_json = value_labels_json(original),
    stata_missing_tags_observed = paste(tags_observed, collapse = ";"),
    codebook_version = mapping_row$codebook_version[[1]],
    codebook_page = mapping_row$codebook_page[[1]],
    codebook_locator = mapping_row$codebook_locator[[1]],
    verification_status = mapping_row$verification_status[[1]],
    stringsAsFactors = FALSE
  )
}

encoded_frame <- as.data.frame(encoded, check.names = FALSE, stringsAsFactors = FALSE)
dir.create(dirname(parquet_out), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(schema_out), recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(encoded_frame, parquet_out)
write.csv(do.call(rbind, schema), schema_out, row.names = FALSE, fileEncoding = "UTF-8")

cat(
  "cohort=", cohort,
  " domains=", paste(domains, collapse = ";"),
  " rows=", nrow(encoded_frame),
  " variables=", length(variables),
  "\n",
  sep = ""
)
