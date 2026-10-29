Development
===========

For requirements and launching the extension from source, see
:doc:`getting_started`.

Local checks
------------

.. code-block:: bash

   npm ci
   npm run check
   npm run coverage
   npm run package

``check`` runs Prettier, ESLint, TypeScript compilation and the unit tests.
``coverage`` prints a text summary and writes ``lcov.info`` under
``build/coverage/typescript``. Coverage includes unexercised extension code;
unit coverage is not evidence that the interactive VS Code UI has been tested.
None of these commands sends a model request.

``npm run smoke`` sends three real requests through the authenticated local CLI
and checks the result, Markdown renderer and a file read in a temporary Git
repository, including live tool execution/result events. It stops the CLI,
resumes the same session in a new process and verifies the earlier context.
It uses your Copilot allowance.
Override the executable with ``ARRANGED_AGENTS_CLI_PATH`` if needed.

Use **Test Arranged Agents activation** in the Run and Debug view for an F5
activation/tab-opening check. A successful run writes host versions and the loaded extension
path to ``dist/scripts/extension-host-smoke-result.json``. It tests the development
extension, not VSIX installation.

Manual validation
-----------------

Install the packaged VSIX, then check:

* Automatic development-tab opening and **Arranged Agents: Open** after VSIX installation.
* Empty left panel, repository list on the right and the chat in the center.
* Pointer/keyboard panel resizing and activity icons during startup and generation.
* Direct-child discovery, lazy startup and context isolation while switching repositories.
* Repository file access, Git/GitHub queries and permission confirmation/rejection.
* Enter to send, Shift+Enter for new lines, and live command/output visibility.
* Category checkboxes hide/restore history without affecting execution or drafts.
* Markdown, fenced code and inline/display equations, without remote math resources.
* Cancellation, concurrent requests in different repositories, tab closure and deactivation.
* Close/reopen history and draft restoration, lazy resume and interrupted-operation notices.
* Missing CLI session recovery through an explicit **New session**, never a silent reset.
* Visible persistence errors and no automatic rerun of interrupted commands on resume.
* Missing/login-invalid CLI and invalid models, with visible errors and clean restart.
* Blocking in an untrusted workspace, escaped HTML and blocked remote images.
* HTTP/HTTPS links open externally; command, script and local-file links stay blocked.

Building documentation
-----------------------

Python is only needed for documentation, not the extension:

.. code-block:: bash

   python -m venv .venv
   .venv/bin/python -m pip install -r docs/requirements.txt
   .venv/bin/python -m sphinx -n -W --keep-going -b html docs build/docs/html

On Windows, use ``.venv/Scripts/python.exe`` instead. Open
``build/docs/html/index.html`` after a successful build. Sphinx warnings fail the
build, including broken internal references.

Refreshing feature screenshots
------------------------------

The feature guide uses committed PNG screenshots of the actual application
webview with fixed, fictional repositories, conversations and activity. They do
not require VS Code, GitHub authentication or a Copilot request.

.. code-block:: bash

   npx playwright install chromium
   npm run docs:screenshots

On Linux, Playwright may also require browser system libraries; install them
with ``npx playwright install --with-deps chromium`` if Chromium reports missing
libraries. The generator uses the current renderer, browser script and stylesheet
with a fixed dark theme, viewport, locale and timezone. It writes seven images
to ``docs/images`` and fails on browser script errors or unexpected network
requests. Review the images after UI changes, then rebuild the documentation.
The Sphinx build uses the committed images and does not need a browser.

GitHub Actions
--------------

* **Tests**: formatting, lint, unit tests, high-severity dependency audit and VSIX
  packaging on Linux and Windows for pushes to ``main``, PRs and manual runs.
* **Nightly coverage**: TypeScript coverage artifacts and Codecov uploads.
  Configure Codecov with the ``CODECOV_TOKEN`` repository secret.
* **Weekly documentation**: strict Sphinx builds and HTML artifacts; also runs
  for documentation PRs. Scheduled and manual failures open an issue.
* **Release**: tests and packages ``v*`` tags or published GitHub releases and
  attaches the VSIX to the release.

Actions are pinned to commits. Only release publication has write access to
repository contents, and only the documentation-failure job can create issues.
No workflow runs the authenticated Copilot smoke test.

For a release, first update ``package.json`` and ``package-lock.json`` to the
intended version, commit the change, then tag that commit with the corresponding
``v``-prefixed version and push the tag. A published GitHub release rebuilds and
replaces the asset for its tag. No workflow publishes to the Marketplace.

Dependency and packaging policy
-------------------------------

Keep optional dependencies when running ``npm ci``. Koffi, used by the SDK,
distributes its prebuilt native binaries through optional platform packages.
``--omit=optional`` forces a source build, which fails with the Windows runner's
MSVC toolchain. The SDK's bundled runtime packages are excluded from the VSIX
by ``.vscodeignore`` instead; the application still uses the installed Copilot CLI.

The project has no selected license. ``npm run package`` uses ``--skip-license``
while the package remains ``UNLICENSED``; this is a packaging policy, not a license
grant.

The Zod, Azure Identity and KaTeX overrides in ``package.json`` address dependency
advisories. Before changing them, run ``npm audit`` and
``npm audit --omit=dev``, then rerun the tests.
