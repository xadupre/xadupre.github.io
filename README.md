# xadupre.github.io

A kind of dashboard.

## Branch protection and the auto-updating workflows

Most of the workflows under [`.github/workflows`](.github/workflows) commit and
push to the default branch (`main`) on their own. They refresh the cached CSV
files under `cache_data/` or rebuild the documentation under `docs/`. A
classic branch protection rule that requires pull requests or status checks
will block those pushes and break the dashboard.

There is no way for a workflow to push to a branch that requires a pull
request unless the actor performing the push is explicitly allowed to bypass
the rule. The following setups are known to work with this repository:

1. **Leave `main` unprotected (default).** This is the simplest option and
   the one currently assumed by the workflows. Anyone with write access can
   push directly, which is required for the bot commits to land. Combine it
   with the *Require signed commits* setting only if you also follow option
   2 or 3 below, otherwise the bot pushes will be rejected.

2. **Use a repository ruleset with a bypass list (recommended).**
   In `Settings → Rules → Rulesets`, create a ruleset targeting `main` with
   the protections you want (for example *Require a pull request before
   merging* and *Require status checks*) and add the following entries to
   *Bypass list*:

   - the **GitHub Actions** bypass actor, so that pushes made with the
     default `GITHUB_TOKEN` from the workflows in this repository are
     allowed. On public repositories and on paid plans this is exposed
     directly as the *GitHub Actions* entry in the bypass picker; on plans
     where that entry is not available, add a repository role of *Maintain*
     or higher and run the workflows under an account with that role
     (see option 3 below);
   - the repository owner, so that manual maintenance pushes keep working.

   Rulesets (unlike the legacy *Branch protection rules*) support bypassing
   per app and per role, which is what makes them the right tool here.

3. **Push with a dedicated identity that is allowed to bypass protection.**
   If your plan does not let GitHub Actions bypass a branch protection rule
   directly, generate a fine-grained Personal Access Token (or a GitHub App
   installation token) for an account that is on the bypass list, store it
   as a repository secret (for example `BOT_TOKEN`), and replace the
   checkout/push steps in the workflows with that token, e.g.:

   ```yaml
   - uses: actions/checkout@v6
     with:
       fetch-depth: 0
       token: ${{ secrets.BOT_TOKEN }}
   ```

   The `git push` step then uses `BOT_TOKEN` instead of the default
   `GITHUB_TOKEN` and the protection rule lets it through.

In every case the workflows already retry the push after rebasing on top of
`origin/main`, so transient races with other commits do not require any
additional configuration.
