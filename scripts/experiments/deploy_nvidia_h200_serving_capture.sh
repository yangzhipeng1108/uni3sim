#!/bin/bash
# Usage: deploy_nvidia_h200_serving_capture.sh HOST
# A caller may supply MIDLAYER_H200_HOST instead of the positional host.
# Deploy a clean committed snapshot, with LFS payloads, under the chosen root.
set -euo pipefail
if [ "$#" -gt 1 ]; then
    echo "usage: $0 [HOST] (or set MIDLAYER_H200_HOST)" >&2
    exit 2
fi
CAPTURE_HOST=${1:-${MIDLAYER_H200_HOST:?supply HOST or set MIDLAYER_H200_HOST}}
case "$CAPTURE_HOST" in
    ''|-*|*[[:space:]]*) echo "invalid SSH host" >&2; exit 2 ;;
esac
PUBLIC_ROOT=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT to the capture root}
case "$PUBLIC_ROOT" in
    /*) ;;
    *) echo "capture root must be absolute" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../.."
if [ -n "$(git status --porcelain)" ]; then
    echo "worktree dirty" >&2
    exit 1
fi
SHA=$(git rev-parse HEAD)
SHORT=${SHA:0:9}
DEST=$PUBLIC_ROOT/src/midlayer_h200_serving_capture/$SHORT
ARCHIVE=$(mktemp "${TMPDIR:-/tmp}/midlayer_capture.XXXXXX.tgz")
trap 'rm -f -- "$ARCHIVE"' EXIT
git ls-files -z | tar --exclude=.git --null -T - -czf "$ARCHIVE"

# Quote every remote argument for the login shell, including embedded quotes.
shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}
REMOTE_SCRIPT=$(cat <<'REMOTE'
set -euo pipefail
destination=$1
commit=$2
mkdir -p -- "$destination"
tar xzf - -C "$destination"
printf '%s\n' "$commit" > "$destination/SOURCE_COMMIT"
mkdir -p -- "$destination/build"
cc -O3 -shared -fPIC -o "$destination/build/lru_sim.so" "$destination/src/kernel/csrc/lru_sim.c"
ln -sfn -- "$destination" "${destination%/*}/current"
REMOTE
)
REMOTE_COMMAND="bash -c $(shell_quote "$REMOTE_SCRIPT") -- $(shell_quote "$DEST") $(shell_quote "$SHA")"
timeout 600 ssh "$CAPTURE_HOST" "$REMOTE_COMMAND" < "$ARCHIVE"
printf '%s\n' "$DEST"
