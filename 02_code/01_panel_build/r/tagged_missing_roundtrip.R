# 在系统临时目录验证 Stata tagged missing 与 value labels 的可逆转换契约。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("用法：tagged_missing_roundtrip.R <temporary_output_directory>")

output_dir <- args[[1]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!requireNamespace("haven", quietly = TRUE)) stop("缺少 haven")
if (!requireNamespace("arrow", quietly = TRUE)) stop("缺少 arrow")

raw_value <- haven::labelled(
  c(
    haven::tagged_na("d"),
    haven::tagged_na("r"),
    haven::tagged_na("m"),
    haven::tagged_na("u"),
    haven::tagged_na("v"),
    haven::tagged_na("p"),
    NA_real_,
    0,
    1,
    -1
  ),
  labels = c(
    "zero" = 0,
    "positive" = 1,
    "negative" = -1,
    "do not know" = haven::tagged_na("d"),
    "refused" = haven::tagged_na("r"),
    "missing" = haven::tagged_na("m"),
    "unmarried" = haven::tagged_na("u"),
    "spouse nonresponse" = haven::tagged_na("v"),
    "proxy" = haven::tagged_na("p")
  ),
  label = "synthetic tagged missing fixture"
)

dta_path <- file.path(output_dir, "tagged_missing_fixture.dta")
parquet_path <- file.path(output_dir, "tagged_missing_roundtrip.parquet")
haven::write_dta(data.frame(row_id = seq_along(raw_value), value = raw_value), dta_path)
roundtrip <- haven::read_dta(dta_path)
x <- roundtrip$value

label_value_text <- function(value) {
  if (haven::is_tagged_na(value)) return(paste0(".", haven::na_tag(value)))
  if (is.na(value)) return("system_na")
  format(value, scientific = FALSE, trim = TRUE)
}
observed_labels <- attr(x, "labels")
observed_label_map <- setNames(
  vapply(observed_labels, label_value_text, character(1)),
  names(observed_labels)
)
expected_label_map <- c(
  "zero" = "0",
  "positive" = "1",
  "negative" = "-1",
  "do not know" = ".d",
  "refused" = ".r",
  "missing" = ".m",
  "unmarried" = ".u",
  "spouse nonresponse" = ".v",
  "proxy" = ".p"
)
if (!identical(
  observed_label_map[sort(names(observed_label_map))],
  expected_label_map[sort(names(expected_label_map))]
)) {
  stop("value labels 在 DTA -> haven round-trip 后发生变化")
}

tagged <- haven::is_tagged_na(x)
missing_tag <- rep(NA_character_, length(x))
missing_tag[tagged] <- vapply(x[tagged], function(value) as.character(haven::na_tag(value)), character(1))
missing_tag[is.na(x) & !tagged] <- "system_na"
clean_value <- haven::zap_labels(x)
clean_value[is.na(clean_value)] <- NA

arrow::write_parquet(
  data.frame(
    row_id = seq_along(x),
    value__value = clean_value,
    value__missing_tag = missing_tag,
    stringsAsFactors = FALSE
  ),
  parquet_path
)
cat("tagged_missing_roundtrip=PASS\n")
