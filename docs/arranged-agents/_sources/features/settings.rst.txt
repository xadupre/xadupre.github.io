Settings
========

Select **Settings** at the top right of the conversation to open the Arranged
Agents settings page in VS Code. A repository session does not need to be
running.

Request timeout
---------------

``arrangedAgents.requestTimeoutMinutes`` sets the total deadline for a Copilot
operation and defaults to 60 minutes. The deadline includes model generation,
tool execution, permission and question waits, and crash-recovery delays.
Activity does not reset it.

The setting applies to main and parallel agents and each PostMortem analysis
request. Changes affect subsequent operations, including operations started from
an existing session. An operation already in progress retains its original
deadline. **Cancel** remains available at any time.

Copilot executable
------------------

Set ``arrangedAgents.cliPath`` to the absolute Copilot CLI path when the extension
cannot find ``copilot`` on ``PATH``. This is a machine-scoped setting.

In Remote SSH, WSL and dev containers, the executable must be installed and
authenticated on the machine that runs the workspace extension.

Model
-----

Set ``arrangedAgents.model`` to a supported model identifier. Leaving it empty
uses the Copilot CLI default.
