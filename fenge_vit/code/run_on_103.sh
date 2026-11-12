#!/usr/bin/env bash
# Preserve the canonical NAS paths used by the immutable training manifests.
# This mount namespace exists only for this command and its children.
set -euo pipefail
FENGE_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FENGE_NAS_IMAGES=/media/nas/lzc/fenge_result
if [ "$#" -eq 0 ]; then
    echo 'Usage: bash code/run_on_103.sh COMMAND [ARGS...]' >&2
    exit 2
fi
test -d "$FENGE_NAS_IMAGES"
exec bwrap --bind / / --dev-bind /dev /dev --proc /proc \
    --tmpfs /home --bind /home/lzc /home/lzc \
    --ro-bind /media/nas /home/nas \
    --bind "$FENGE_NAS_IMAGES" /home/nas/lzc/fenge_result \
    --tmpfs /media --bind /media/common /media/common \
    --symlink /home/nas /media/nas \
    --chdir "$FENGE_PROJECT" -- "$@"
