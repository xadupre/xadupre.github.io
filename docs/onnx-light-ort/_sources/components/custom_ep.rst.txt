Custom execution provider
=========================

``onnx_light_ort/custom_ep/`` reserves the independent ORT-loadable EP library.
There is no implementation, CMake target or executable unit test yet; the
directory README states that limitation explicitly.

The future library owns ``CreateEpFactories``, ``ReleaseEpFactory``, factory
and device callbacks, node selection and kernel-registry or compiled-partition
callbacks. It will use :doc:`bridge_ep_light_to_ort` and its
:doc:`common_light_to_ort` dependency, not :doc:`custom_ops` or the ops bridge.
It must never link :doc:`common_ort_to_light` or the incoming bridges, or
interpose their compatibility API tables.
The first EP can support selected standard ONNX nodes without rewriting their
domain/name, leaving unsupported nodes to the real ORT host.

Required tests with the first implementation
---------------------------------------------

Factory/device ownership, partial-registration failure, capability selection,
unsupported-node fallback, kernel-state reuse, changing shapes and teardown
must be unit-tested when their code exists. A separate real-ORT test must run
a mixed supported/unsupported graph and verify numerical output and placement.
Testing empty declarations now would not validate those behaviors.

See :doc:`../newt_steps/2026/2026-10-start` for the public callback inventory, CPU-first milestones
and the later compiling-EP extension.
