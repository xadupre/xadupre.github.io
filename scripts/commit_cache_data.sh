#!/usr/bin/env bash

set -euo pipefail

readonly EXPECTED_REMOTE='xadupre/cache_data'
readonly TARGET_BRANCH='main'
readonly MAX_ATTEMPTS=5
readonly RETRY_DELAY_SECONDS="${CACHE_DATA_RETRY_DELAY_SECONDS:-5}"

usage() {
    echo "Usage: $0 DATA_REPOSITORY COMMIT_MESSAGE" >&2
}

if [[ $# -ne 2 || -z $1 || -z $2 ]]; then
    usage
    exit 2
fi

data_repo=$1
commit_message=$2

if [[ ! -d $data_repo ]]; then
    echo "Data repository path does not exist: ${data_repo}" >&2
    exit 1
fi

data_repo=$(cd -- "$data_repo" && pwd -P)
repo_root=$(git -C "$data_repo" rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z $repo_root ]]; then
    echo "Not a Git checkout: ${data_repo}" >&2
    exit 1
fi
repo_root=$(cd -- "$repo_root" && pwd -P)
if [[ $repo_root != "$data_repo" ]]; then
    echo "Data repository path must be its checkout root: ${data_repo}" >&2
    exit 1
fi

origin_url=$(git -C "$data_repo" config --get remote.origin.url 2>/dev/null || true)
case "$origin_url" in
    https://github.com/xadupre/cache_data|\
    https://github.com/xadupre/cache_data.git|\
    git@github.com:xadupre/cache_data|\
    git@github.com:xadupre/cache_data.git|\
    ssh://git@github.com/xadupre/cache_data|\
    ssh://git@github.com/xadupre/cache_data.git)
        ;;
    *)
        echo "Refusing to commit unexpected repository '${origin_url:-<no origin>}'." >&2
        echo "Expected GitHub repository: ${EXPECTED_REMOTE}" >&2
        exit 1
        ;;
esac

git -C "$data_repo" config user.name "github-actions[bot]"
git -C "$data_repo" config user.email \
    "41898282+github-actions[bot]@users.noreply.github.com"
git -C "$data_repo" add -A -- .

if git -C "$data_repo" diff --cached --quiet; then
    echo "No cache data changes to commit."
    exit 0
fi

git -C "$data_repo" commit -m "$commit_message"

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if push_output=$(git -C "$data_repo" push origin "HEAD:${TARGET_BRANCH}" 2>&1); then
        printf '%s\n' "$push_output"
        echo "Cache data push succeeded on attempt ${attempt}."
        exit 0
    else
        push_status=$?
    fi
    printf '%s\n' "$push_output" >&2

    if grep -Eiq \
        'HTTP[^[:digit:]]*403|error: 403|returned error: 403|Permission to .* denied' \
        <<<"$push_output"; then
        echo "Cache data push was rejected with HTTP 403 or permission denied." >&2
        echo "BOT_TOKEN must have Contents read/write access to xadupre/cache_data." >&2
        exit "$push_status"
    fi

    if ((attempt == MAX_ATTEMPTS)); then
        break
    fi

    echo "Cache data push attempt ${attempt} failed; rebasing on origin/main."

    # Include any files written after the initial commit before rebasing.
    git -C "$data_repo" add -A -- .
    if ! git -C "$data_repo" diff --cached --quiet; then
        git -C "$data_repo" commit --amend --no-edit
    fi

    if grep -Eiq \
        'non-fast-forward|fetch first|stale info|cannot lock ref|failed to update ref' \
        <<<"$push_output"; then
        git -C "$data_repo" fetch origin "$TARGET_BRANCH"
        if ! git -C "$data_repo" rebase -X theirs "origin/${TARGET_BRANCH}"; then
            echo "Failed to rebase cache data changes on origin/main." >&2
            git -C "$data_repo" rebase --abort || true
            exit 1
        fi
    fi
    sleep $((attempt * RETRY_DELAY_SECONDS))
done

echo "Failed to push cache data after ${MAX_ATTEMPTS} attempts." >&2
exit 1
