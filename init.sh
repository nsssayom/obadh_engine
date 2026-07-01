#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

AUTOCORRECT_PATH="data/autocorrect"
AUTOSUGGEST_PATH="data/autosuggest"
AUTOSUGGEST_GENERATOR_PREFIX="$AUTOSUGGEST_PATH/models/neural/autosuggest-generator-gru256-topk128-c64-balanced"

info() {
  printf '==> %s\n' "$1"
}

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

ensure_git_lfs() {
  if git lfs version >/dev/null 2>&1; then
    return
  fi

  if command -v brew >/dev/null 2>&1; then
    info "Installing git-lfs with Homebrew"
    brew install git-lfs
  fi

  git lfs version >/dev/null 2>&1 || die "git-lfs is required; install it, then rerun ./init.sh"
}

path_has_git_metadata() {
  [ -d "$1/.git" ] || [ -f "$1/.git" ]
}

path_is_empty_dir() {
  [ -d "$1" ] && [ -z "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]
}

prepare_submodule_path() {
  local path="$1"

  if [ ! -e "$path" ]; then
    return
  fi

  if path_has_git_metadata "$path"; then
    return
  fi

  if path_is_empty_dir "$path"; then
    rm -rf "$path"
    return
  fi

  local backup="${path}.backup.$(date +%Y%m%d%H%M%S)"
  info "Moving existing non-submodule $path to $backup"
  mv "$path" "$backup"
}

pull_lfs() {
  local path="$1"
  [ -d "$path" ] || die "missing submodule path: $path"
  info "Resolving LFS objects in $path"
  git -C "$path" lfs pull
}

verify_file() {
  [ -s "$1" ] || die "expected file is missing or empty: $1"
}

verify_resolved_file() {
  verify_file "$1"
  if head -n 1 "$1" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
    die "Git LFS pointer was not resolved: $1"
  fi
}

require_command git
require_command npm
ensure_git_lfs

[ -f ".gitmodules" ] || die ".gitmodules is missing; the data submodules are not configured"

info "Initializing data submodules"
git lfs install
prepare_submodule_path "$AUTOCORRECT_PATH"
prepare_submodule_path "$AUTOSUGGEST_PATH"
git submodule sync --recursive -- "$AUTOCORRECT_PATH" "$AUTOSUGGEST_PATH"
git submodule update --init --recursive -- "$AUTOCORRECT_PATH" "$AUTOSUGGEST_PATH"

pull_lfs "$AUTOCORRECT_PATH"
pull_lfs "$AUTOSUGGEST_PATH"

verify_resolved_file "$AUTOCORRECT_PATH/models/bn.fst"
verify_resolved_file "$AUTOCORRECT_PATH/models/en_bn_loanwords.fst"
verify_file "$AUTOSUGGEST_PATH/corpus/manifest.json"
verify_file "$AUTOSUGGEST_PATH/models/ngram/vocab.manifest.json"
verify_resolved_file "$AUTOSUGGEST_PATH/models/ngram/vocab.tsv"
verify_file "$AUTOSUGGEST_PATH/models/ngram/autosuggest-ngram.manifest.json"
verify_resolved_file "$AUTOSUGGEST_PATH/models/ngram/autosuggest-ngram.bin"
verify_file "$AUTOSUGGEST_PATH/models/ngram/autosuggest-ngram-c64.manifest.json"
verify_resolved_file "$AUTOSUGGEST_PATH/models/ngram/autosuggest-ngram-c64.bin"
verify_file "$AUTOSUGGEST_GENERATOR_PREFIX.manifest.json"
verify_resolved_file "$AUTOSUGGEST_GENERATOR_PREFIX.onnx"
verify_resolved_file "$AUTOSUGGEST_GENERATOR_PREFIX.int8.onnx"
verify_file "$AUTOSUGGEST_GENERATOR_PREFIX.mlpackage/Manifest.json"
verify_file "$AUTOSUGGEST_GENERATOR_PREFIX.mlpackage/Data/com.apple.CoreML/model.mlmodel"
verify_resolved_file "$AUTOSUGGEST_GENERATOR_PREFIX.mlpackage/Data/com.apple.CoreML/weights/weight.bin"

info "Installing web dependencies"
npm --prefix www install

info "Obadh workspace is initialized"
