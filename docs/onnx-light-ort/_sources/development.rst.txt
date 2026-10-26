Development
===========

Build against onnx-light main
-----------------------------

The default CMake configuration fetches onnx-light's ``main`` branch.
Python bindings are disabled for native-only builds; Python test options
automatically enable the matching onnx-light model bindings. Upstream tests,
benchmarks remain disabled. Built-in operator kernels and the onnx-light-cpu
Python bindings are enabled for the Python gallery and real-ORT tests so the
benchmark can also run native onnx-light + onnx-light-cpu graphs.
The outgoing common library compiles only
the existing LeakyRelu kernel source alongside the core; unused targets are
excluded from the default build.
Plugin builds additionally fetch onnx-light-cpu at
``054cdd984d9f85a7692dc4f678b2f13275e8ede1`` and configure only its standalone
``ml_cpu_compute`` subproject. Gallery and real-ORT test builds additionally
compile its Python bindings and kernel registration sources against the same
onnx-light runtime, placing the Python package in ``build/python``. No additional
onnx-light build is needed. Its runtime SIMD dispatch remains enabled; do not
add target-wide native/AVX compiler flags. To supply that same source checkout,
set ``FETCHCONTENT_SOURCE_DIR_ONNX_LIGHT_CPU=/path/to/onnx-light-cpu``.
Dependency headers are treated as system headers; warnings remain errors in
this project's own sources.
Set ``Python3_EXECUTABLE`` to select an interpreter explicitly. The same
interpreter is forwarded to the onnx-light bindings build.

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

On Windows or other multi-configuration generators, also pass
``--config Release`` to the build and ``-C Release`` to CTest.
On macOS use ``sysctl -n hw.logicalcpu`` in place of ``nproc``.
Keep one build tree; reconfigure it rather than maintaining stale parallel
compiled copies.

The Python source-build frontend follows onnx-light's command:

.. code-block:: bash

   python -m pip install -r requirements/host-tests.txt
   python setup.py build_ext --inplace --cpp-tests
   export PYTHONPATH="$PWD:$PWD/build/python:$PYTHONPATH"

It configures Release libraries and matching model bindings in the existing
``build`` tree, without installing or copying compiled artifacts.
``--cpp-tests`` enables, builds and runs the native CTest suites; omit it to
build without tests. Native unit, dependency and consumer tests are selected,
not opt-in Python/ORT or gallery tests. ``--parallel N`` or
``CMAKE_BUILD_PARALLEL_LEVEL`` controls concurrency (all available CPUs by
default).
``CMAKE_ARGS`` forwards additional configuration options and existing
dependency-source overrides are preserved. The frontend enforces Release
even if ``CMAKE_ARGS`` contains another build type, matching its build and
CTest configuration. For example:

.. code-block:: bash

   CMAKE_ARGS="-DONNX_LIGHT_ORT_TEST_PYTHON=ON -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON" \
     python setup.py build_ext --inplace --cpp-tests
   ctest --test-dir build -C Release --output-on-failure

To use a locally built onnx-light runtime, as with onnx-light-cpu:

.. code-block:: bash

   PYTHONPATH="/path/to/onnx-light:$PYTHONPATH" \
     python setup.py build_ext --inplace --cpp-tests --onnx-light-source

``--onnx-light-source`` uses ``onnx_light.get_cpp_build_info()`` from the
package selected by Python (including ``PYTHONPATH``). It links against that
runtime's existing core, proto and threading shared libraries, without
fetching or rebuilding onnx-light or copying its Python bindings.
The checkout must already be built. If its model package lives outside the
source checkout, set ``ONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR`` to that matching
checkout. Windows also requires the matching import libraries reported by
``get_cpp_build_info()``. Missing headers or libraries fail the build explicitly.
If the import libraries live in another build tree, supply their exact paths
with ``ONNX_LIGHT_ORT_ONNX_LIGHT_CORE_IMPLIB``,
``ONNX_LIGHT_ORT_ONNX_LIGHT_PROTO_IMPLIB`` and
``ONNX_LIGHT_ORT_ONNX_LIGHT_THREADS_IMPLIB``.
If external runtime DLLs are found through ``PATH``, launch the frontend with
``python cmake/run_python_with_dll_paths.py setup.py build_ext --inplace --cpp-tests --onnx-light-source``.
This registers the DLL directories in that Python process; registrations in a
parent process are not inherited by subprocesses.
This mode disables installation and does not support upstream fuzzers;
release packages continue to use the fetched dependency build.
Omit the flag to restore the fetched dependency configuration in the same tree.
The onnx-light-cpu compute dependency still uses its existing source override.

The default Python APIs read ``build/CMakeCache.txt`` and resolve libraries
using the manifest generated from CMake's actual targets: the active build
type for single-configuration generators, or
``build/Release/python_libraries.json`` for multi-configuration generators.
An unset single-config build type uses ``build/python_libraries.json``.
Explicit library paths remain available for other multi-config configurations
or custom build locations. See :doc:`python_api`.

Build either direction independently:

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_ORT_BUILD_HOST=ON \
     -DONNX_LIGHT_ORT_BUILD_PLUGIN=OFF
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

Reverse the two options for a plugin-only build; both are enabled by default.
The plugin option selects ``custom_ops``, both ``*_light_to_ort`` bridges and
``common_light_to_ort``. The host option selects both ``*_ort_to_light`` bridges
and ``common_ort_to_light``. Unimplemented EP bridges remain interface targets;
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
``custom_ops``, ``bridge_ops_light_called_by_ort``, ``common_light_to_ort`` or
``common_ort_to_light`` or ``bridge_ops_ort_called_by_light`` to run a component's tests.
The ``unit`` label selects native unit suites. Outgoing units use a minimal
explicit C API host, not a mocking framework or an ORT installation. Incoming
native tests load an independent ORT ``LeakyRelu`` custom-op library into the real standalone
host; the dependency-contract smoke test is retained. EP bridges and the EP
entry-point library have no executable tests.
See :doc:`components/index` for each component's page and precise coverage.

The ``gallery_benchmark_log_custom_ops`` CTest exercises every supported
backend and both graph topologies with 1, 7 and 33 elements, including plotting
and all correctness checks. It is a smoke test, not a performance measurement:
the full million-element gallery benchmark is too costly in an unoptimized
coverage build. Direct gallery execution and documentation builds retain the
full input-size range.

The C++ custom-op example is built by default when plugins are enabled,
without linking ORT. ``ONNX_LIGHT_ORT_BUILD_EXAMPLES=OFF`` disables it.
Both :doc:`components/bridge_ops_light_called_by_ort` examples run inference in a real ORT host.
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
They also execute ``onnx_light_cpu::Log`` and check its special values and SIMD
tails against a NumPy float64 reference, rounded to float32, with relative
tolerance ``2e-6`` and zero absolute tolerance.
Both operators also run in float16, bfloat16 and float64 through real ORT and
the standalone host, preserving dtype. The float64 cases include values
that would lose precision or overflow if routed through float32.

On Linux the loadable library is
``build/onnx_light_ort/custom_ops/libonnx_light_ort_custom_ops.so``.
CTest resolves the platform-specific path from the custom-op target, including
multi-configuration build directories.

The real-host CI job resolves the latest stable ONNX Runtime release on each
run, installs its Python wheel, and downloads the matching C++ runtime archive
with SHA-256 verification against the release asset digest.
Models are created and checked with ``onnx_light.onnx``, never the ``onnx``
Python package.

Documentation and style
-----------------------

Sphinx builds the matching onnx-light Python bindings and native gallery
libraries in the existing ``build`` tree, then executes both gallery examples.
The standard command below also works from a fresh checkout without setting
``PYTHONPATH`` or library environment variables. It requires Python 3.12 or later,
a C++20 compiler, Git and network access to fetch the native dependencies.
No local project artifacts are installed into site-packages.

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

Standalone Python custom-op tests
---------------------------------

Use the source checkout through ``PYTHONPATH``; do not install the project.
The incoming host uses NumPy and the onnx-light Python model bindings,
not the ``onnx`` package or an ONNX Runtime installation:

.. code-block:: bash

   python -m pip install -r requirements/host-tests.txt
   cmake -S . -B build -DONNX_LIGHT_ORT_BUILD_HOST=ON \
     -DONNX_LIGHT_ORT_BUILD_PLUGIN=OFF -DONNX_LIGHT_ORT_TEST_WITH_ORT=OFF \
     -DONNX_LIGHT_ORT_TEST_PYTHON=ON
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build -L bridge_ops_ort_called_by_light --output-on-failure

The build places the onnx-light model package under ``build/python`` and CTest
adds it to ``PYTHONPATH``. It uses the same source revision as the native host;
no project artifacts are installed into site-packages. For direct Python use,
set ``PYTHONPATH="$PWD:$PWD/build/python:$PYTHONPATH"``.
On Windows, CTest additionally registers the native test ``PATH`` directories
with ``os.add_dll_directory`` for the lifetime of each Python script. Python
3.8 and later do not search ``PATH`` when loading extension-module dependencies.
This includes external DLLs needed by the onnx-light libraries, not just the
libraries placed under ``build/python``. Direct Windows invocations can use
``python cmake/run_python_with_dll_paths.py path/to/script.py ...`` as well.

``CustomOpSessionOrtCalledByLight`` accepts a model path, an onnx-light ``ModelProto`` or
serialized model bytes. Plugin/host paths default to the bundled libraries or
the source build and can be overridden explicitly.
Its audited single-node floating-point profile is documented in
:doc:`components/bridge_ops_ort_called_by_light`; it is not a general plugin host.

Executable Python gallery
--------------------------

The :doc:`auto_examples/index` gallery runs both custom-op directions.
Sphinx prepares its prerequisites automatically. To also run the integration
tests or execute the scripts directly on Linux, build both sides and set the
three library paths:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   cmake -S . -B build -DONNX_LIGHT_ORT_BUILD_HOST=ON \
     -DONNX_LIGHT_ORT_BUILD_PLUGIN=ON -DONNX_LIGHT_ORT_TEST_PYTHON=ON \
     -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure
   export PYTHONPATH="$PWD:$PWD/build/python:$PYTHONPATH"
   export ONNX_LIGHT_ORT_PLUGIN_LIBRARY="$PWD/build/onnx_light_ort/custom_ops/libonnx_light_ort_custom_ops.so"
   export ONNX_LIGHT_ORT_HOST_LIBRARY="$PWD/build/onnx_light_ort/bridge_ops_ort_called_by_light/libonnx_light_ort_bridge_ops_ort_called_by_light.so"
   export ONNX_LIGHT_ORT_INCOMING_CUSTOM_OP_LIBRARY="$PWD/build/onnx_light_ort/bridge_ops_ort_called_by_light/libonnx_light_ort_ort_custom_ops.so"
   python -m sphinx -b html -W --keep-going docs build/docs

On macOS use the corresponding ``.dylib`` paths; on Windows use the ``.dll``
paths under the selected build configuration. CTest resolves these paths
automatically. The ``gallery_custom_ops_light_called_by_ort`` and
``gallery_custom_ops_ort_called_by_light``
tests execute the same gallery scripts in separate Python processes.
The incoming example never imports ORT; the outgoing example uses a real
ORT session. No EP is involved, and failures are not skipped or hidden.

Sphinx builds the model/runtime bindings, native CPU bindings and the three
gallery libraries, not the native tests. Existing CMake dependency-source
overrides are preserved,
and explicitly supplied gallery library environment variables take precedence.

The ``gallery_benchmark_log_custom_ops`` test executes the Log benchmark gallery.
It compares ORT's built-in Log and the onnx-light-cpu custom op called by ORT,
both using I/O binding with preallocated buffers, with the ORT-ABI CPU custom op
called by the standalone onnx-light single-node host. A fourth path uses
onnx-light's ``ReferenceEvaluator`` with ``onnx_light_cpu.register_kernels()``
and standard-domain Log nodes, without ONNX Runtime or an ORT-ABI bridge.
This native path supports both graph topologies and uses one CPU thread.
NumPy is used only for input preparation and correctness checks, not as a
timed backend. It uses one
Log or ten consecutive Logs, checks results before and after timing, and
reports median microseconds per graph and speedup relative to ORT's built-in
Log over eleven input sizes from 1 to 1,048,576 elements. Latency and speedup
are plotted separately for each topology. Ten Logs
produce NaNs in float32; this measures the literal chain, including special
values. The standalone host cannot execute that topology and is marked
unavailable. Measurements run in separate processes to keep ORT out of the
standalone host and native CPU evaluator. Use the same library environment
variables shown above.
All benchmark code lives in ``docs/examples/plot_benchmark_log_custom_ops.py``,
including model construction, timing, isolated workers and the CLI entry point.
The command-line benchmark also accepts ``--nodes 1|10`` and ``--sizes``.
ORT always uses I/O binding; the standalone host and native CPU evaluator use ``run``.
To run only the native CPU worker, without ORT or either bridge library:

.. code-block:: bash

   python docs/examples/plot_benchmark_log_custom_ops.py --backend native --nodes 10

CI and documentation publishing
-------------------------------

Native CI covers Linux, Windows and macOS, including host-only, plugin-only
and combined configurations. It uses a compiler cache, cancels superseded
branch runs, and builds against upstream main. A Linux check rejects an ORT
shared-library dependency in the native test executable and plugin library.
Real-host tests run in a separate Linux job; host-only jobs do not install ORT.
Release publication is handled separately by ``.github/workflows/releases.yml``;
ordinary native CI does not create releases.

Tagged releases
---------------

The ``Releases`` workflow starts automatically when a tag is pushed.
It calls the reusable ``Python release package`` workflow
(``.github/workflows/release-python.yml``) to build wheels and a source
distribution. That workflow can also be dispatched independently to build
downloadable Actions artifacts without publishing or creating a tag.
For version 0.2.0, both ``0.2.0`` and ``v0.2.0`` are accepted. It checks that the
tag, CMake project, Python metadata and ``__version__`` agree before building.
It resolves onnx-light's main revision once and uses that immutable commit in
every release job; onnx-light-cpu remains independently pinned. Each package
includes ``release-build-info.txt`` and the dependency licenses.

The targets match onnx-light's release matrix:

.. list-table::
   :header-rows: 1

   * - Platform
     - Architectures
     - CPython wheels
   * - Linux, manylinux 2.28
     - x86_64, aarch64
     - 3.12, 3.13, 3.14
   * - Windows
     - AMD64, ARM64
     - 3.12, 3.13, 3.14
   * - macOS 13.3+
     - universal2 (arm64 + x86_64)
     - 3.12, 3.13, 3.14

There are fifteen wheels, one Python source distribution and five native C++
archives. Native archives use
Ubuntu 24.04 on Linux; they are not the portable manylinux wheels.
Windows archives are ZIP files; Linux and macOS archives are gzip-compressed
tar files. All builds enable both custom-op directions and use all available
processors. No ORT runtime or ``onnx`` Python package is bundled.

Native archive builds disable Python and optional OpenSSL support. They contain
the two shared libraries with statically linked onnx-light/CPU dependencies,
public host and ORT API headers, and a relocatable CMake package. Each job moves
the installation and builds/runs an independent downstream consumer:

.. code-block:: cmake

   find_package(onnx_light_ort 0.2 CONFIG REQUIRED)
   target_link_libraries(my_host PRIVATE onnx_light_ort::bridge_ops_ort_called_by_light)

``onnx_light_ort::custom_ops`` is the installed plugin target.

Wheels bundle the host, plugin and their native runtime dependencies under
``onnx_light_ort/_native``. Bundled dependency libraries use private names to
avoid ABI collisions with a separately installed onnx-light model package.
The matching onnx-light model bindings are built
in the same tree for wheel smoke tests, but are not copied into this package.
The tests load the installed libraries first and execute LeakyRelu and CPU Log
without ORT. Dependency repair uses auditwheel, delvewheel or delocate as
appropriate; Windows repair preserves already bundled DLLs.
Success on one platform does not publish a partial release: publication waits
for every wheel and native job, then attaches all artifacts and ``SHA256SUMS``.
A retry can replace artifacts on the same release.

``workflow_dispatch`` without a tag builds and tests artifacts without
publishing. With an existing tag it can retry a release. This workflow publishes
to GitHub Releases, not PyPI; pushing a tag is an explicit maintainer action.

To build a wheel locally without installing it, use the existing build tree.
Set ``CMAKE_GENERATOR`` to its current generator (``Unix Makefiles`` in this
Linux example), rather than creating another compiled tree:

.. code-block:: bash

   CMAKE_GENERATOR="Unix Makefiles" CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
     python -m pip wheel . --no-deps -w build/wheelhouse

For source-checkout work, continue using ``PYTHONPATH`` rather than installing
local artifacts. Wheel builds reconfigure ``build``; restore the desired native
options and ``-DSKBUILD=OFF`` before returning to a plain CMake development build.

Daily coverage
---------------

The ``Coverage`` workflow runs every day at 07:00 UTC on ``main`` and can
also be started manually. It builds both native directions with GCC coverage
instrumentation and runs native tests, Python tests in both custom-op
directions, and the Python gallery through CTest. Coverage includes the C++
implementation, headers and Python modules under ``onnx_light_ort/``, excluding
test sources, downloaded dependencies, generated files and examples.
Python coverage also follows subprocesses, including benchmark workers.

The workflow uploads the C++ and Python Cobertura XML reports to
`Codecov <https://codecov.io/gh/xadupre/onnx-light-ort>`_ using GitHub OIDC.
Enable this repository in Codecov before the first upload; no
``CODECOV_TOKEN`` secret is required. Upload failures fail the workflow.
The README and documentation badges show the latest ``main`` coverage.
HTML and XML reports are also retained as the ``coverage`` Actions artifact
for 14 days. Relevant pull requests generate these artifacts without
publishing to Codecov, so they do not replace the daily ``main`` report.

To reproduce both reports locally, reconfigure the existing build tree:

.. code-block:: bash

   python -m pip install -r requirements/coverage.txt
   CC=gcc CXX=g++ cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
     -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
     -DCMAKE_SHARED_LINKER_FLAGS="--coverage" \
     -DONNX_LIGHT_ORT_BUILD_HOST=ON -DONNX_LIGHT_ORT_BUILD_PLUGIN=ON \
     -DONNX_LIGHT_ORT_BUILD_TESTS=ON -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON \
     -DONNX_LIGHT_ORT_TEST_PYTHON=ON -DONNX_LIGHT_ORT_PYTHON_COVERAGE=ON
   cmake --build build --parallel "$(nproc)"
   mkdir -p build/coverage
   export COVERAGE_FILE="$PWD/build/coverage/.coverage"
   python -m coverage erase
   ctest --test-dir build --output-on-failure
   python -m gcovr build --root . \
     --filter 'onnx_light_ort/' --exclude '.*/tests/.*' \
     --xml-pretty --xml build/coverage/coverage.xml \
     --html-details build/coverage/index.html --print-summary
   python -m coverage combine
   python -m coverage xml -o build/coverage/python.xml
   python -m coverage html -d build/coverage/python
   python -m coverage report

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
