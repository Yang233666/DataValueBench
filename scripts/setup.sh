#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-install}" in
  --help|-h) printf 'Usage: bash scripts/setup.sh [--check|--help]\nInstall creates .venv; --check validates the selected existing PYTHON_BIN without downloading.\n'; exit 0 ;;
  --check)
    PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
    "$PYTHON_BIN" -B -c 'import sys,importlib.metadata as m; assert sys.version_info >= (3,10); expected={"numpy":"1.26.4","pandas":"2.2.2","scipy":"1.15.3","pyarrow":"15.0.2","scikit-learn":"1.7.2","pytest":"8.2.2","numba":"0.61.2"}; bad={k:(m.version(k),v) for k,v in expected.items() if m.version(k)!=v}; assert not bad,bad; print("Core environment: PASS")'
    exit 0 ;;
  install) ;;
  *) printf 'Unknown option. Use --help.\n' >&2; exit 2 ;;
esac
if [[ -e "$ROOT/.venv" ]]; then printf '.venv already exists; use --check or select an existing interpreter.\n' >&2; exit 2; fi
"${PYTHON_BOOTSTRAP:-python3}" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements/core.txt"
"$ROOT/.venv/bin/python" -m pip install --no-deps -e "$ROOT"
printf 'Installed locally. Run bash scripts/run_tests.sh and bash scripts/verify_release.sh.\n'
