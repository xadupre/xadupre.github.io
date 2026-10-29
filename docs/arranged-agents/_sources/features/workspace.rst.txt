Workspace
=========

Arranged Agents opens as a single editor tab with three resizable panels:

* repository activity on the left;
* the selected repository's Copilot conversation in the center;
* repositories beneath the chosen parent directory on the right.

.. figure:: ../images/workspace.png
   :alt: GitHub issues on the left, Markdown conversation in the center and active or saved repositories on the right.

   Three resizable panels, shown with illustrative demo data.

Drag either divider to resize adjacent panels. A focused divider also responds
to the left and right arrow keys.

Repository discovery
--------------------

The right panel lists immediate child directories that contain a ``.git``
directory or file, including Git worktrees. Discovery is intentionally one level
deep and does not follow directory symlinks.

No Copilot process starts when the application opens. Selecting a repository
starts its session lazily and makes its activity and conversation visible.
Open or busy repositories move into **Active sessions**; the remaining entries
stay under **Other repositories**. Selection, open-session state and running work
have separate visual indicators.

Use **Refresh** to rescan the parent directory. Existing sessions remain open
when their repositories are still present. Switching repositories preserves
each conversation and unfinished draft.

Navigation
----------

Issue, pull request, comment and GitHub-backed commit links open in the external
browser. Only validated HTTP and HTTPS links are accepted. Command, script and
local-file URL schemes are blocked.

See :doc:`sessions` for session isolation and persistence, and
:doc:`github_activity` for the left panel.
