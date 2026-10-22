Development
===========

Build against onnx-light main
-----------------------------

The default CMake configuration fetches onnx-light's ``main`` branch.
Python bindings, upstream tests, benchmarks and built-in operator kernels
are disabled in the dependency configuration. The outgoing common library compiles only
the existing LeakyRelu kernel source alongside the core; unused targets are
excluded from the default build.

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

On Windows or other multi-configuration generators, also pass
``--config Release`` to the build and ``-C Release`` to CTest.
On macOS use ``sysctl -n hw.logicalcpu`` in place of ``nproc``.
Keep one build tree; reconfigure it rather than maintaining stale parallel
compiled copies.

Build either direction independently:

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_ORT_BUILD_HOST=ON \
     -DONNX_LIGHT_ORT_BUILD_PLUGIN=OFF
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

Reverse the two options for a plugin-only build; both are enabled by default.
The plugin option selects ``custom_ops``, both ``*_light_to_ort`` bridges and
``common_light_to_ort``. The host option selects both ``*_ort_to_light`` bridges
and ``common_ort_to_light``. Unimplemented bridges remain interface targets;
they do not acquire fake runtime implementations just by being enabled.
The EP entry-point library will have its own target when implemented.
``ONNX_LIGHT_ORT_BUILD_TESTS=OFF`` disables native integration tests.
When switching to a host-only build, also turn
``ONNX_LIGHT_ORT_TEST_WITH_ORT=OFF`` if previously enabled.

To use an existing checkout explicitly:

.. code-block:: bash

   git -C /path/to/onnx-light fetch origin main
   cmake -S . -B build \
     -DFETCHCONTENT_SOURCE_DIR_ONNX_LIGHT=/path/to/onnx-light

The override uses the checkout exactly as supplied and does not switch
branches. Ensure it contains the desired main revision; CMake logs its HEAD.
CI explicitly checks out upstream ``main`` and passes this override so its
resolved commit appears in the job logs. Fresh CI runs and the daily schedule
exercise new upstream main changes without a release.

Component tests and examples
-----------------------------

Each implemented component owns its native tests. Select CTest labels
``custom_ops``, ``bridge_ops_light_to_ort``, ``common_light_to_ort`` or
``common_ort_to_light`` to run a component's tests, or ``unit`` for the three
implemented outgoing
unit suites. These use a minimal explicit C API host, not a mocking framework
or an ORT installation. The incoming common library currently has dependency-
contract smoke tests only; unimplemented bridges and the EP entry-point
library have no executable tests.
See :doc:`components/index` for each component's page and precise coverage.

The C++ custom-op example is built by default when plugins are enabled,
without linking ORT. ``ONNX_LIGHT_ORT_BUILD_EXAMPLES=OFF`` disables it.
Both :doc:`components/custom_ops` examples run inference in a real ORT host.
To additionally execute the C++ example through CTest, pass an existing runtime:

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON \
     -DONNX_LIGHT_ORT_RUNTIME_LIBRARY=/absolute/path/to/libonnxruntime.so
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build -L example --output-on-failure

Test in a real ORT host
-----------------------

The native build never needs an ORT runtime. Enable the separate integration
tests explicitly with Python 3.12 or later:

.. code-block:: bash

   python -m pip install -r requirements/ort-tests.txt
   cmake -S . -B build -DONNX_LIGHT_ORT_BUILD_PLUGIN=ON \
     -DONNX_LIGHT_ORT_BUILD_TESTS=ON -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

The tests build models with custom domain ``onnx_light``, opset 1, operator
``LeakyRelu`` and an explicit float ``alpha``. They load the produced shared
library with ``SessionOptions.register_custom_ops_library`` and verify repeated
dynamic/scalar/empty shapes, read-only input preservation, preallocated output
buffers, independent sessions, and invalid attributes, arity, types, rank and
alignment or overlapping buffers. Expected values come from NumPy, not a
substitute native kernel.

On Linux the loadable library is
``build/onnx_light_ort/custom_ops/libonnx_light_ort_custom_ops.so``.
CTest resolves the platform-specific path from the custom-op target, including
multi-configuration build directories.

Only this opt-in path installs ORT test dependencies. The CI job uses ORT
1.18.1 with NumPy below 2 to exercise the requested API-18 baseline.

Documentation and style
-----------------------

.. code-block:: bash

   python -m pip install -r requirements/style.txt -r docs/requirements.txt
   black . && ruff check .
   git ls-files -z '*.cc' '*.cpp' '*.h' '*.hpp' | xargs -0 clang-format -i
   git ls-files -z '*.cc' '*.cpp' '*.h' '*.hpp' | xargs -0 clang-format --dry-run --Werror
   python -m sphinx -b html -W --keep-going docs build/docs

The build and style tools ignore downloaded dependencies and generated output.
Native tests use the real onnx-light core without mocking libraries or an
installed Python package. Reverse-direction units use explicit test-owned
opaque values and API callbacks. Nothing is installed into site-packages by
the native build.

CI and documentation publishing
-------------------------------

Native CI covers Linux, Windows and macOS, including host-only, plugin-only
and combined configurations. It uses a compiler cache, cancels superseded
branch runs, and builds against upstream main. A Linux check rejects an ORT
shared-library dependency in the native test executable and plugin library.
Real-host tests run in a separate Linux job; host-only jobs do not install ORT.

Daily coverage
---------------

The ``Coverage`` workflow runs every day at 07:00 UTC on ``main`` and can
also be started manually. It builds both native directions with GCC coverage
instrumentation and runs all native CTest tests. Coverage includes the C++
implementation and headers under ``onnx_light_ort/``, excluding test sources,
downloaded dependencies, generated files and examples. Python and opt-in
real-ORT tests are not included.

The workflow uploads the Cobertura XML report to
`Codecov <https://codecov.io/gh/xadupre/onnx-light-ort>`_ using GitHub OIDC.
Enable this repository in Codecov before the first upload; no
``CODECOV_TOKEN`` secret is required. Upload failures fail the workflow.
The README and documentation badges show the latest ``main`` coverage.
HTML and XML reports are also retained as the ``coverage`` Actions artifact
for 14 days. Relevant pull requests generate these artifacts without
publishing to Codecov, so they do not replace the daily ``main`` report.

To reproduce the native report locally, reconfigure the existing build tree:

.. code-block:: bash

   python -m pip install -r requirements/coverage.txt
   CC=gcc CXX=g++ cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
     -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
     -DCMAKE_SHARED_LINKER_FLAGS="--coverage" \
     -DONNX_LIGHT_ORT_BUILD_HOST=ON -DONNX_LIGHT_ORT_BUILD_PLUGIN=ON \
     -DONNX_LIGHT_ORT_BUILD_TESTS=ON -DONNX_LIGHT_ORT_TEST_WITH_ORT=OFF
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure
   mkdir -p build/coverage
   python -m gcovr build --root . \
     --filter 'onnx_light_ort/' --exclude '.*/tests/.*' \
     --xml-pretty --xml build/coverage/coverage.xml \
     --html-details build/coverage/index.html --print-summary

Fuzzing onnx-light
------------------

The optional fuzz targets are provided by the fetched onnx-light dependency
and require Clang with libFuzzer support:

.. code-block:: bash

   CC=clang CXX=clang++ cmake -S . -B build-fuzz \
     -DONNX_LIGHT_ORT_BUILD_FUZZERS=ON \
     -DONNX_LIGHT_ORT_BUILD_TESTS=OFF
   cmake --build build-fuzz --parallel "$(nproc)"
   ./build-fuzz/_deps/onnx_light-build/fuzz_checker -runs=2000

Fuzzing is disabled by default. The ``fuzz`` GitHub Actions workflow builds
the available upstream harnesses and runs short smoke campaigns; it does not
replace longer coverage-guided fuzzing.

Documentation builds on pull requests, main pushes and manual dispatch.
Only successful main-branch documentation builds deploy to GitHub Pages.
Workflow tokens are read-only by default. Pages write permission is restricted
to documentation deployment; OIDC write permission is granted to documentation
deployment and the coverage job for Codecov authentication.
