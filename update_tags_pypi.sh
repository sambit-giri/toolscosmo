#!/usr/bin/env bash
#
# Release toolscosmo: create the git tag v<version> and upload to PyPI.
#
# The version is read from pyproject.toml. Bump it there (via a PR) and merge
# to main before running this script.
#
# Usage:
#   ./update_tags_pypi.sh               # test, build, check, tag, upload to PyPI, push tag
#   ./update_tags_pypi.sh --dry-run     # do every check and build, but no tag and no upload
#   ./update_tags_pypi.sh --test-pypi   # upload to TestPyPI instead (no tag)
#   ./update_tags_pypi.sh --tag-only    # only create and push the tag (version already on PyPI)
#   ./update_tags_pypi.sh --yes         # skip the confirmation prompt
#   ./update_tags_pypi.sh --skip-tests  # skip the test suite
#
# Environment:
#   PYTHON   interpreter used to build and test (default: python3). The wheel is
#            built for this interpreter's version and platform, e.g. use a
#            Python 3.11 interpreter to get a cp311 wheel.
#
# Requirements: git, curl, and for PYTHON: build, twine.
# PyPI credentials are read by twine from ~/.pypirc or TWINE_* variables.
#
# Order of operations: checks -> clean build -> twine check -> install the
# wheel in a fresh venv and run the tests there -> confirm -> local tag ->
# upload -> push tag.
# If the upload fails, the local tag is deleted, so nothing is half-released.

set -euo pipefail

DRY_RUN=0; TEST_PYPI=0; TAG_ONLY=0; ASSUME_YES=0; SKIP_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=1 ;;
        --test-pypi)  TEST_PYPI=1 ;;
        --tag-only)   TAG_ONLY=1 ;;
        --yes|-y)     ASSUME_YES=1 ;;
        --skip-tests) SKIP_TESTS=1 ;;
        -h|--help)    sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
    esac
done

PYTHON="${PYTHON:-python3}"
REMOTE="origin"
BRANCH="main"
PKG="toolscosmo"

die()  { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

cd "$(git rev-parse --show-toplevel)"

# --- Checks ------------------------------------------------------------------
step "Checking repository state"
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "not on '$BRANCH'."
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "working tree has uncommitted changes."
git fetch -q "$REMOTE" "$BRANCH" --tags
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$REMOTE/$BRANCH")" ] \
    || die "local '$BRANCH' differs from '$REMOTE/$BRANCH'; pull or push first."

VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' pyproject.toml | head -1)"
[ -n "$VERSION" ] || die "could not read version from pyproject.toml."
TAG="v$VERSION"
COMMIT="$(git rev-parse --short HEAD)"
echo "Version $VERSION, tag $TAG, commit $COMMIT ($(git log -1 --format=%s))"

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
   || git ls-remote --exit-code --tags "$REMOTE" "refs/tags/$TAG" >/dev/null 2>&1; then
    die "tag $TAG already exists. Bump the version in pyproject.toml."
fi

PYPI_STATUS="$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/$PKG/$VERSION/json" || true)"
if [ "$TAG_ONLY" = 1 ]; then
    [ "$PYPI_STATUS" = 200 ] || die "--tag-only, but $PKG $VERSION is not on PyPI (HTTP $PYPI_STATUS)."
    echo "$PKG $VERSION is on PyPI; only the tag will be created."
elif [ "$TEST_PYPI" = 0 ] && [ "$PYPI_STATUS" = 200 ]; then
    [ "$DRY_RUN" = 1 ] && echo "WARNING: $PKG $VERSION is already on PyPI; a real run would stop here." || \
    die "$PKG $VERSION is already on PyPI (versions cannot be re-uploaded). Bump the version, or use --tag-only."
fi

if [ "$TAG_ONLY" = 0 ]; then
    "$PYTHON" -c "import build, twine" 2>/dev/null \
        || die "$PYTHON needs 'build' and 'twine' ($PYTHON -m pip install build twine)."
    echo "Building with $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

    WORK="$(mktemp -d)"
    trap 'git worktree remove --force "$WORK/src" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

    # --- Build from a clean checkout of HEAD ---------------------------------
    # A clean worktree keeps untracked files (downloaded data, compiled .so,
    # stale egg-info) out of the sdist and wheel.
    step "Building sdist and wheel from a clean checkout of $COMMIT"
    git worktree add -q --detach "$WORK/src" HEAD
    "$PYTHON" -m build --outdir "$WORK/dist" "$WORK/src" > "$WORK/build.log" 2>&1 \
        || { tail -30 "$WORK/build.log"; die "build failed (log above)."; }
    ls -l "$WORK/dist"

    step "Checking distributions"
    "$PYTHON" -m twine check --strict "$WORK"/dist/*
    tar tzf "$WORK"/dist/*.tar.gz | grep -q '\.pyx$' || die "sdist does not contain the .pyx sources."
    if tar tzf "$WORK"/dist/*.tar.gz | grep -qE '\.cov$|\.part$'; then die "sdist contains downloaded data."; fi

    step "Installing the wheel (with its dependencies) into a fresh venv"
    "$PYTHON" -m venv "$WORK/venv"
    "$WORK/venv/bin/pip" install -q "$WORK"/dist/*.whl pytest
    (cd "$WORK" && "$WORK/venv/bin/python" -c "
import importlib.metadata as m, toolscosmo
from toolscosmo import cython_ParkinsonColeHelly2008
assert m.version('$PKG') == '$VERSION', m.version('$PKG')
print('import OK:', '$PKG', m.version('$PKG'))
")

    # Run from inside tests/ so the installed wheel is imported, not src/.
    if [ "$SKIP_TESTS" = 0 ]; then
        step "Running the test suite against the installed wheel"
        (cd "$WORK/src/tests" && "$WORK/venv/bin/python" -m pytest -q -p no:cacheprovider .)
    fi
fi

if [ "$DRY_RUN" = 1 ]; then
    step "Dry run finished: all checks passed. No tag was created and nothing was uploaded."
    exit 0
fi

# --- Confirm -----------------------------------------------------------------
if [ "$TEST_PYPI" = 1 ]; then
    ACTION="upload $PKG $VERSION to TestPyPI (no tag)"
elif [ "$TAG_ONLY" = 1 ]; then
    ACTION="create and push tag $TAG at $COMMIT"
else
    ACTION="create tag $TAG at $COMMIT, upload $PKG $VERSION to PyPI, and push the tag"
fi
if [ "$ASSUME_YES" = 0 ]; then
    echo
    echo "About to $ACTION."
    [ "$TEST_PYPI" = 1 ] || [ "$TAG_ONLY" = 1 ] || echo "A PyPI upload cannot be undone: this version number can never be reused."
    read -r -p "Continue? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || die "aborted."
fi

# --- Release -----------------------------------------------------------------
if [ "$TEST_PYPI" = 1 ]; then
    step "Uploading to TestPyPI"
    "$PYTHON" -m twine upload --repository testpypi "$WORK"/dist/*
    exit 0
fi

step "Creating tag $TAG"
git tag -a "$TAG" -m "$PKG $VERSION"

if [ "$TAG_ONLY" = 0 ]; then
    step "Uploading to PyPI"
    if ! "$PYTHON" -m twine upload "$WORK"/dist/*; then
        git tag -d "$TAG" >/dev/null
        die "upload failed; local tag $TAG removed. Nothing was released."
    fi
fi

step "Pushing tag $TAG"
git push "$REMOTE" "refs/tags/$TAG"

step "Done: $ACTION."
echo "  PyPI: https://pypi.org/project/$PKG/$VERSION/"
echo "  Tag:  $(git remote get-url "$REMOTE" | sed 's/\.git$//')/releases/tag/$TAG"
