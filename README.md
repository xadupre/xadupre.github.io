# xadupre.github.io

![Repo size](https://img.shields.io/github/repo-size/xadupre/xadupre.github.io)

A kind of dashboard.

The dashboard data are maintained in the separate public
[`xadupre/cache_data`](https://github.com/xadupre/cache_data) repository, with
the former `cache_data/` contents at that repository's root. Its GitHub Pages
project serves those files under `https://xadupre.github.io/cache_data/`, so
the existing dashboard URLs remain unchanged.

## Branch protection and the auto-updating workflows

Most of the workflows under [`.github/workflows`](.github/workflows) commit and
push to a default branch (`main`) on their own. Documentation workflows update
`docs/` in this repository. Data workflows check out `xadupre/cache_data` at
`cache_data/` (or `site/cache_data/`) and push generated CSV/JSON files to that
repository instead. A classic branch protection rule that requires pull
requests or status checks will block those pushes and break publication.

The `CACHE_DATA_SSH_KEY` secret contains a writable deploy key for
`xadupre/cache_data`. It is used for data publication because a workflow's
`GITHUB_TOKEN` cannot push to another repository. `BOT_TOKEN` remains available
for documentation workflows that push to `xadupre/xadupre.github.io`.

There is no way for a workflow to push to a branch that requires a pull
request unless the actor performing the push is explicitly allowed to bypass
the rule. The following setups are known to work with this repository:

1. **Leave `main` unprotected (default).** This is the simplest option and
   the one currently assumed by the workflows. Anyone with write access can
   push directly, which is required for the bot commits to land. Combine it
   with the *Require signed commits* setting only if you also follow option
   2 or 3 below, otherwise the bot pushes will be rejected.

2. **Use a repository ruleset with a bypass list (recommended).**
   In each repository's `Settings → Rules → Rulesets`, create a ruleset
   targeting `main` with the protections you want (for example *Require a pull
   request before merging* and *Require status checks*) and configure its
   *Bypass list*:

   - for documentation pushes to this repository, allow the **GitHub Actions**
     bypass actor when using the default `GITHUB_TOKEN`, or allow the
     `BOT_TOKEN` identity;
   - for data pushes to `xadupre/cache_data`, allow its writable deploy key
     because the source repository's `GITHUB_TOKEN` cannot be used there;
   - the repository owner, so that manual maintenance pushes keep working.

   Rulesets (unlike the legacy *Branch protection rules*) support bypassing
   per app and per role, which is what makes them the right tool here.

3. **Push with a dedicated identity that is allowed to bypass protection.**
   If your plan does not let GitHub Actions bypass a branch protection rule
   directly, configure the `cache_data` deploy key as a bypass actor. The
   private key is stored as the `CACHE_DATA_SSH_KEY` repository secret in this
   repository, while the public key is registered with write access on
   `xadupre/cache_data`. A writable checkout uses it as follows:

   ```yaml
   - uses: actions/checkout@v6
     with:
       repository: xadupre/cache_data
       path: cache_data
       ref: main
       fetch-depth: 0
       ssh-key: ${{ secrets.CACHE_DATA_SSH_KEY }}
   ```

   The data checkout uses v6 and its stored SSH credential. Primary
   checkouts of this site intentionally remain on v5 because later
   documentation push steps rely on its credential storage behavior.

In every case the workflows retry after rebasing on `origin/main`, so
transient races with other commits do not require additional configuration.

### Troubleshooting "none of the actions can update main"

If every workflow that pushes to `main` fails with a job log that looks like

```
remote: Permission to <owner>/<target-repo>.git denied to <user>.
fatal: unable to access 'https://github.com/<owner>/<repo>/': The requested URL returned error: 403
Push attempt N failed; rebasing on origin/main and retrying...
...
Failed to push after 5 attempts.
```

the workflow code itself is fine — the rebase loop is succeeding, but the
HTTPS push is being rejected by GitHub with HTTP **403**. The actor named on
the `denied to <user>` line tells you which credential is being used and
therefore which setting is missing:

- **`denied to github-actions[bot]`** — a documentation push is using the
  default `GITHUB_TOKEN` and cannot write this repository's `main`. Check:
    1. *Settings → Actions → General → Workflow permissions* must be set to
       **Read and write permissions**. The per-workflow `permissions:
       contents: write` block only lifts permissions up to the repository
       cap; if the cap is read-only the workflow token is read-only too.
    2. If `main` is covered by a branch protection rule or a ruleset
       (*Settings → Branches* or *Settings → Rules → Rulesets*), add the
       **GitHub Actions** bypass actor as described in option 2 above.
       Plain *Branch protection rules* cannot grant this bypass; convert
       the rule to a *Ruleset* if needed.
- **`denied to <your-username>`** — `BOT_TOKEN` cannot write
  `xadupre.github.io` for documentation updates. Check:
    1. The token has not expired (fine-grained PATs expire by default
       after a short period).
    2. For a fine-grained PAT: the token grants **Contents: Read and write**
       on this repository.
    3. For a classic PAT: the token has the `repo` scope (or at least
       `public_repo` for a public repository).
    4. The account that owns the PAT is on the target repository's *Bypass
       list* for every rule that targets `main`.

After updating the setting, re-run a failing workflow from the *Actions*
tab. The push should succeed on the first attempt and no rebase loop
should be printed.
