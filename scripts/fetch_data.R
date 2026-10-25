# Alternative zu scripts/fetch_data.py – für den Fall, dass Python keine Yahoo-Daten bekommt,
# R (quantmod) aber schon. Erzeugt dasselbe Datenpaket in data/<name>/.
#
#   install.packages(c("quantmod", "jsonlite"))
#   Rscript scripts/fetch_data.R templates/eu_ai_act.json [name]
#
# FF3-Faktoren lädt dieses Skript nicht; dafür die ZIP-Datei im Browser von der French Data Library
# holen und nach data/factors/ legen (siehe data/factors/README.md).

suppressPackageStartupMessages({
  library(quantmod)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Aufruf: Rscript scripts/fetch_data.R <config.json> [name]")
cfg <- fromJSON(args[1], simplifyVector = FALSE)

slug <- if (length(args) >= 2) args[2] else gsub("(^-|-$)", "", gsub("[^a-z0-9]+", "-", tolower(cfg$name)))
out <- file.path("data", slug)
dir.create(out, recursive = TRUE, showWarnings = FALSE)

tickers <- unique(c(unlist(lapply(cfg$groups, function(g) unlist(g$tickers))),
                    unlist(lapply(cfg$groups, function(g) g$market_index))))

# Zeitraum wie im Tool: großzügig vor dem frühesten und nach dem spätesten Event
dates <- as.Date(sapply(cfg$events, function(e) e$date))
lead <- abs(cfg$estimation_window[[1]]) * 1.6 + 30 + if (isTRUE(cfg$placebo$enabled)) 400 else 0
lag <- max(sapply(cfg$windows, function(w) w[[2]])) * 1.6 + 30 + if (isTRUE(cfg$placebo$enabled)) 200 else 0
from <- min(dates) - lead
to <- min(max(dates) + lag, Sys.Date())
cat(sprintf("Lade %d Ticker von %s bis %s …\n", length(tickers), from, to))

series <- list()
for (t in tickers) {
  x <- tryCatch(getSymbols(t, src = "yahoo", from = from, to = to, auto.assign = FALSE, warnings = FALSE),
                error = function(e) NULL)
  if (!is.null(x) && nrow(x) > 0) {
    s <- Ad(x)
    colnames(s) <- t
    series[[t]] <- s
  } else {
    cat("  keine Daten:", t, "\n")
  }
  Sys.sleep(0.2)  # Yahoo nicht überlasten
}
if (length(series) == 0) stop("Keine Kurse erhalten.")

prices <- do.call(merge, series)
df <- data.frame(Date = index(prices), coredata(prices), check.names = FALSE)
write.csv(df, file.path(out, "prices.csv"), row.names = FALSE, na = "")
write.csv(data.frame(ticker = names(series), source = "R quantmod (Yahoo)"),
          file.path(out, "sources.csv"), row.names = FALSE)
write(toJSON(list(study = cfg$name, created = format(Sys.time()), period = c(format(from), format(to)),
                  tickers = length(tickers), loaded = length(series),
                  missing = setdiff(tickers, names(series))), auto_unbox = TRUE, pretty = TRUE),
      file.path(out, "manifest.json"))
cat(sprintf("\nDatenpaket: %s (%d/%d Ticker)\n", out, length(series), length(tickers)))
