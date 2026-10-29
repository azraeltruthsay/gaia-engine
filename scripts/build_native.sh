#!/usr/bin/env bash
# Put gaia_cpp's native artifacts next to the engine source, once, for every
# container that mounts this package directory.
#
#   gaia_engine/cpp/gaia_cpp.cpython-311-x86_64-linux-gnu.so   the extension
#   gaia_engine/cpp/lib/libllama.so.0, libggml*.so.0           the llama.cpp it was
#                                                              compiled against
#
# Bundling the llama.cpp libraries matters: images ship their own, newer
# llama.cpp for llama-server, and gaia_cpp must link the build it was compiled
# against. gaia_engine/cpp/__init__.py loads the bundled copies first. The CUDA
# runtime (libcudart / libcublas) comes from each image.
#
# Usage:
#   scripts/build_native.sh                      # extract from the gaia-prime image
#   scripts/build_native.sh --image <tag>        # extract from another built image
#   scripts/build_native.sh --rebuild            # rebuild the image first (slow), then extract
#
# Both artifacts are git-ignored; rerun this after changing gaia_cpp.cpp.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONOREPO="$(cd "$HERE/.." && pwd)"
IMAGE="localhost:5000/gaia-prime:gaia-engine"
REBUILD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --rebuild) REBUILD=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ "$REBUILD" = 1 ]; then
    echo "Rebuilding $IMAGE from gaia-prime/Dockerfile.gaia-engine ..."
    (cd "$MONOREPO" && docker build -f gaia-prime/Dockerfile.gaia-engine -t "$IMAGE" .)
fi

SITE=/usr/local/lib/python3.11/site-packages/gaia_engine/cpp
DEST="$HERE/gaia_engine/cpp"
STAGE="$(mktemp -d)"
CID="$(docker create "$IMAGE")"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true; rm -rf "$STAGE"' EXIT

echo "Extracting gaia_cpp from $IMAGE ..."
docker cp "$CID:$SITE/gaia_cpp.cpython-311-x86_64-linux-gnu.so" "$STAGE/"
mkdir -p "$STAGE/lib"
for lib in libllama.so.0 libggml.so.0 libggml-base.so.0 libggml-cpu.so.0 libggml-cuda.so.0; do
    # docker cp follows the soname symlink to the real file
    docker cp -L "$CID:/usr/local/lib/$lib" "$STAGE/lib/$lib"
done

# Install by rename, never by overwriting: running engines have these files
# mapped, and rewriting a mapped .so in place crashes them (it segfaulted
# gaia-prime). A rename leaves them on the old inode until their next restart.
install_atomic() {  # <src> <dest-file>
    cp "$1" "$2.new.$$" && mv -f "$2.new.$$" "$2"
}
mkdir -p "$DEST/lib"
install_atomic "$STAGE/gaia_cpp.cpython-311-x86_64-linux-gnu.so" "$DEST/gaia_cpp.cpython-311-x86_64-linux-gnu.so"
for f in "$STAGE"/lib/*.so.0; do
    install_atomic "$f" "$DEST/lib/$(basename "$f")"
done
echo "Running engines keep the previous build until restarted."
echo "Installed into $DEST:"
ls -la "$DEST"/gaia_cpp*.so "$DEST"/lib/
