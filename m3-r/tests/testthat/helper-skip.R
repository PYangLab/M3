# Engine tests provision a basilisk env (torch etc.) and need ~minutes the first
# time, so they are opt-in: set M3_TEST_ENGINE=1 to run them.
skip_if_no_engine <- function() {
  if (!nzchar(Sys.getenv("M3_TEST_ENGINE"))) {
    testthat::skip("engine test (set M3_TEST_ENGINE=1 to run)")
  }
  if (!requireNamespace("basilisk", quietly = TRUE)) {
    testthat::skip("basilisk not installed")
  }
}
