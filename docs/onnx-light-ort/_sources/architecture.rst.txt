Architecture
============

Four bridges and two common libraries
-------------------------------------

Bridge names state who provides the implementation and who calls it:
``ort_called_by_light`` means ORT-compatible operators or providers called by
onnx-light; ``light_called_by_ort`` means onnx-light implementations called by
ORT. ``ops`` and ``ep`` distinguish custom operators from execution providers.

See :doc:`newt_steps/2026/2026-10-start` for the detailed implementation roadmap and the
inventory of host functions and plugin callbacks.

Each component has its own directory under ``onnx_light_ort/``. The root
CMake file selects components and tests; it does not define a monolithic
plugin or keep implementation sources in a root ``src/`` directory.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Component / target
     - Responsibility
     - Current status
   * - ``onnx_light_ort::custom_ops``
     - ORT-loadable library owning ``RegisterCustomOps``, operator descriptors,
       domains and callback wiring.
     - Shared library.
   * - ``custom_ep`` (future target)
     - Independent ORT-loadable library owning ``CreateEpFactories``,
       ``ReleaseEpFactory`` and EP callbacks.
     - Reserved directory; not implemented.
   * - ``onnx_light_ort::common_light_to_ort``
     - Shared native kernel state, buffer validation and exception/status
       conversion for both bridges targeting ORT.
     - Static library.
   * - ``onnx_light_ort::common_ort_to_light``
     - Shared API-table, opaque-value and lifetime support for both
       compatibility bridges targeting onnx-light without ORT.
     - Static API-18 CPU subset; see the incoming custom-op profile.
   * - ``onnx_light_ort::bridge_ops_light_called_by_ort``
     - Custom-op construction, attributes, execution and destruction callbacks.
     - Static library.
   * - ``onnx_light_ort::bridge_ep_light_called_by_ort``
     - EP-specific adaptation for running onnx-light inside ORT.
     - Interface target; runtime planned.
   * - ``onnx_light_ort::bridge_ops_ort_called_by_light``
     - Loads and registers foreign custom ops in onnx-light.
     - Shared single-node float32 host with Python access.
   * - ``onnx_light_ort::bridge_ep_ort_called_by_light``
     - Hosts foreign EP factories, selection and execution in onnx-light.
     - Interface target; runtime planned.

The dependency graph is deliberately directional:

.. code-block:: text

   custom_ops -> bridge_ops_light_called_by_ort -> common_light_to_ort -> onnx-light
   custom_ep  -> bridge_ep_light_called_by_ort  -> common_light_to_ort -> onnx-light

   bridge_ops_ort_called_by_light -> common_ort_to_light -> onnx-light
   bridge_ep_ort_called_by_light  -> common_ort_to_light -> onnx-light

Common libraries are split **by direction**, not by ops/EP profile. Neither
owns the other direction's API or lifetime machinery. Bridges own their
profile-specific registration and execution contracts; they do not depend
on another bridge. EP-related runtimes remain planned. Incoming custom-op
hosting is implemented only for the audited subset described in
:doc:`components/bridge_ops_ort_called_by_light`.

The custom-op and EP libraries must remain independently loadable. Neither
may link the opposite-direction bridge, its API tables or its bootstrap
exports. Provider-neutral helpers can be shared later, never the host shim.
No component links ORT. The real consuming application supplies ORT to
``custom_ops`` and ``custom_ep`` through its API tables; only their opt-in
integration tests need an installed runtime.

The existing ``onnx_light_ort::plugin`` alias maps to ``::custom_ops`` and
``onnx_light_ort::host`` maps to ``::common_ort_to_light``. The existing
``BUILD_PLUGIN`` option selects custom ops, both outgoing bridges and their
common library; ``BUILD_HOST`` selects both incoming bridges and their common
library. These names do not
merge the planned EP into the custom-op library.

The dependency test compiles
ORT's public C/EP declarations alongside onnx-light headers, links onnx-light's
core, round-trips a custom-domain model, and executes a native custom kernel
with changing shapes, including scalar and empty tensors. A separate optional
test loads the plugin in real ORT; the default tests do not load ORT.

Implemented custom-op profile
-----------------------------

The shared library registers ``onnx_light::LeakyRelu`` at custom-domain opset 1.
It advertises one required float32 CPU input and output and requires a float
``alpha`` attribute; unlike standard ONNX LeakyRelu, it supplies no default
attribute. Dynamic runtime shapes, scalars and zero-element tensors are
accepted up to rank 16. No static shape-inference callback is registered yet.

The same library registers ``onnx_light_cpu::Log`` at custom-domain opset 1,
with the same tensor profile and no attributes. Its implementation calls
``ml_cpu_compute::LogFloat32`` from onnx-light-cpu's standalone computation
library. ISA selection happens at runtime; the bridge does not force native
compiler ISA flags or create a worker pool. The standalone incoming host can
load this very same API-18 plugin and execute it without ONNX Runtime.
No cross-direction CMake dependency is introduced.

The descriptor requests C API 18 and uses status-returning ``CreateKernelV2``
and ``KernelComputeV2`` callbacks. ``KernelDestroy`` releases each retained
native kernel. Registration uses the supplied ``OrtApiBase``, never imports
``OrtGetApiBase``, and rejects hosts that cannot supply API 18.

The outgoing common library compiles the existing onnx-light ``kernel_leakyrelu.cc``
with its core dependency, not the full kernel collection. It owns the retained
``LeakyReluKernel`` adapter. Both directions reuse the provider-neutral
C-boundary status-conversion helper without linking each other's common library.
``bridge_ops_light_called_by_ort`` owns the custom-op callbacks and required-attribute
contract; ``custom_ops`` wires them into its descriptors. The native kernel
is constructed once per ORT kernel instance. Each invocation borrows the
current input and ORT-allocated output directly, including preallocated outputs
provided through IO binding, without allocating or copying tensor payloads.
Borrowed views and context pointers are never retained across calls.

Before borrowing, the adapter validates CPU memory, float32 type, rank,
nonnegative dimensions, checked element/byte counts, and non-null, float-aligned
nonempty buffers. Input and output buffers must not overlap, including when
the caller supplies them through IO binding. It relies on the ORT host's valid
``OrtValue`` contract for actual allocation capacity and output shape/device;
API 18 does not expose
an independent buffer capacity. Native exceptions are translated to statuses
only at C ABI boundaries. ORT statuses are propagated unchanged.

Registration retains both mutex-protected domains and descriptors per supplied API
table, shared across sessions until library unload. Kernel attributes are
per-instance, not global. As required by the custom-op ABI, the application
must keep the library and its ORT host loaded until all consuming session
options, sessions and kernels have been destroyed.

Dependency boundary
-------------------

The dependency direction is this repository -> onnx-light. Keep only generic
kernel factories, executable-partition interfaces, and ownership/allocation
mechanisms in onnx-light. Do not add ``Ort*`` types to its public interfaces.
Any missing generic hooks require separate onnx-light PRs.

For now builds follow onnx-light's ``main`` branch. The public ORT headers are
pinned independently to commit
``db5f5eaff7c32e09e90520374ac19b78d89094ae``. Pinning headers does not claim
support for every function in their ABI.
The computation-only onnx-light-cpu dependency is pinned independently to
``054cdd984d9f85a7692dc4f678b2f13275e8ede1``. Only its ``ml_cpu_compute``
subproject is configured; its ONNX adapters, Python bindings and dependency
fetching are not enabled. Both bridges keep sharing the existing onnx-light core.

Implementation order
--------------------

The host workstream starts with ABI tables and ownership-aware adapters,
then CPU custom ops, kernel-based EPs, and finally compiling EPs. It must
never fall back to an installed ORT runtime.

The reverse workstream starts with a CPU custom-op library exporting
``RegisterCustomOps``, followed by a kernel-based plugin EP. A compiling EP
is a later milestone. Use the real host's supplied API table and manual
initialization rather than importing ``OrtGetApiBase`` at static initialization.
The host shim is not a prerequisite for the reverse target.

In both directions retain kernel state across calls, validate types, shapes,
devices and buffer extents, and release plugin-owned resources before unloading
libraries. Native plugins are trusted executable code, not sandboxed inputs.
Persistent compilation caches, shared prepacked weights and accelerator
streams require explicit later contracts.
