#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python environment missing. Run bash scripts/setup.sh or set PYTHON_BIN.\n' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
case "${1:-core}" in
  core) shift "$(( $# > 0 ? 1 : 0 ))"; "$PYTHON_BIN" -B -m pytest tests -q -p no:cacheprovider --ignore=tests/test_rq3_kdpp_prepared.py --ignore=tests/test_rq3_pinned_ppr.py "$@" ;;
  graph) shift; "$PYTHON_BIN" -B -m pytest tests/test_rq3_kdpp_prepared.py tests/test_rq3_pinned_ppr.py -q -p no:cacheprovider "$@" ;;
  --help|-h) printf 'Usage: bash scripts/run_tests.sh [core|graph] [pytest options]\n' ;;
  *) printf 'Unknown test group; use core or graph.\n' >&2; exit 2 ;;
esac
