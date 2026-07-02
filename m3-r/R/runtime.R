# The runtime: one persistent Python session for the R session, holding the live
# Python m3 models in a process-local registry (see inst/python/_r_api.py).
#
# A trained m3.M3 holds live torch state that cannot be serialised back into R,
# so we keep it alive in the Python session and address it by an integer handle.
# Every R verb calls a flat helper in _r_api and gets back R-native values
# (matrices / data.frames / lists).
#
# Two backends:
#   * "basilisk" (default): a self-contained env (torch 2.1.2 etc.), provisioned
#     on first use. Reproducible anywhere, no system Python needed.
#   * "reticulate": reuse an EXISTING Python that already has torch + m3's deps.
#     Enable with options(m3.python = "/path/to/python") or env M3_PYTHON. Useful
#     to match a specific environment exactly (e.g. the one the Python tutorials
#     ran in) and to skip provisioning.

# package-local handle to the running session + its fixed device/backend
.m3_runtime <- new.env(parent = emptyenv())

#' Path to the vendored Python (inst/python: holds the `m3` package + `_r_api`).
#' @keywords internal
.pkg_py <- function() {
  p <- system.file("python", package = "m3")
  if (!nzchar(p)) stop("vendored python not found (inst/python).")
  p
}

#' Which Python backend to use ("basilisk" or a system "reticulate" python).
#' @keywords internal
.m3_backend <- function() {
  py <- getOption("m3.python", Sys.getenv("M3_PYTHON", ""))
  if (nzchar(py)) list(mode = "reticulate", python = py) else list(mode = "basilisk")
}

#' Start (once) or fetch the persistent m3 Python session.
#'
#' `device` is fixed for the session at first use, because torch reads CUDA
#' availability when it is first imported. Call \code{\link{m3_shutdown}} to
#' release it (and every live model) and pick a different device.
#' @keywords internal
.m3_proc <- function(device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  if (!is.null(.m3_runtime$started)) return(.m3_runtime$proc)
  be <- .m3_backend()
  .m3_runtime$mode <- be$mode

  if (be$mode == "reticulate") {
    if (identical(device, "cpu")) Sys.setenv(CUDA_VISIBLE_DEVICES = "")
    # RETICULATE_PYTHON is honoured at reticulate init; use_python alone is
    # ignored if reticulate already bound an interpreter. Set both, and fail
    # loudly if something already bound a different python.
    Sys.setenv(RETICULATE_PYTHON = be$python)
    if (reticulate::py_available(initialize = FALSE)) {
      cur <- tryCatch(reticulate::py_config()$python, error = function(e) "")
      if (nzchar(cur) && normalizePath(cur, mustWork = FALSE) !=
          normalizePath(be$python, mustWork = FALSE)) {
        stop("reticulate is already bound to '", cur, "' but m3.python wants '",
             be$python, "'. Set M3_PYTHON / RETICULATE_PYTHON before the first ",
             "reticulate use (restart R).")
      }
    }
    reticulate::use_python(be$python, required = TRUE)
    sys <- reticulate::import("sys", convert = FALSE)
    sys$path$insert(0L, normalizePath(.pkg_py()))
    reticulate::import("_r_api")                    # warm import (loads torch + m3)
    .m3_runtime$proc <- NULL
  } else {
    proc <- basilisk::basiliskStart(m3_env)
    basilisk::basiliskRun(proc, function(pkgpy, device) {
      os <- reticulate::import("os", convert = FALSE)
      if (identical(device, "cpu")) {
        os$environ$update(reticulate::dict(CUDA_VISIBLE_DEVICES = ""))
      }
      sys <- reticulate::import("sys", convert = FALSE)
      sys$path$insert(0L, normalizePath(pkgpy))     # resolve `import m3`, `import _r_api`
      reticulate::import("_r_api")                    # warm import (loads torch + m3)
      invisible(NULL)
    }, pkgpy = .pkg_py(), device = device)
    .m3_runtime$proc <- proc
  }
  .m3_runtime$started <- TRUE
  .m3_runtime$device <- device
  .m3_runtime$proc
}

#' Call a function in inst/python/_r_api in the persistent session.
#'
#' @param fn name of the `_r_api` function.
#' @param ... named arguments forwarded to it (NULLs are dropped so the Python
#'   defaults apply).
#' @param .device session device on first use.
#' @keywords internal
.m3_call <- function(fn, ..., .device = "auto") {
  .m3_proc(.device)
  args <- Filter(Negate(is.null), list(...))
  if (identical(.m3_runtime$mode, "reticulate")) {
    rapi <- reticulate::import("_r_api")            # in-process; cached after warm import
    do.call(rapi[[fn]], args)
  } else {
    basilisk::basiliskRun(.m3_runtime$proc, function(fn, args, pkgpy) {
      sys <- reticulate::import("sys", convert = FALSE)
      sys$path$insert(0L, normalizePath(pkgpy))
      rapi <- reticulate::import("_r_api")          # cached after the warm import
      do.call(rapi[[fn]], args)
    }, fn = fn, args = args, pkgpy = .pkg_py())
  }
}

#' Shut down the m3 Python session.
#'
#' Releases the basilisk process (basilisk backend) and every live model held in
#' it. The next \code{\link{m3_train}} starts fresh. Trained models are
#' session-bound (a live torch object cannot persist across R sessions);
#' re-train to continue.
#'
#' @return invisible \code{NULL}.
#' @examples
#' \donttest{ m3_shutdown() }
#' @export
m3_shutdown <- function() {
  if (isTRUE(.m3_runtime$started) && !is.null(.m3_runtime$proc)) {
    try(basilisk::basiliskStop(.m3_runtime$proc), silent = TRUE)
  }
  .m3_runtime$proc <- NULL
  .m3_runtime$started <- NULL
  .m3_runtime$device <- NULL
  .m3_runtime$mode <- NULL
  invisible(NULL)
}

.onUnload <- function(libpath) {
  m3_shutdown()
}
