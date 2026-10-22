Shared support for hosting ORT-compatible plugins
==================================================

``onnx_light_ort/common_ort_to_light/`` defines the interface target
``onnx_light_ort::common_ort_to_light``. The runtime implementation is planned:
it will expose audited API tables, opaque-value adapters and shared lifetime
support to :doc:`bridge_ops_ort_to_light` and :doc:`bridge_ep_ort_to_light`.
Those bridges, not this common layer, own profile-specific plugin registration.
It must build, test and run without installing, loading or linking ORT.

This direction is independent of :doc:`common_light_to_ort`. In particular,
the minimal host in ``tests/ort_test_host.h`` is explicit unit-test
infrastructure for the reverse direction, not this production bridge.

Current dependency tests
------------------------

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_ORT_BUILD_HOST=ON \
     -DONNX_LIGHT_ORT_BUILD_PLUGIN=OFF -DONNX_LIGHT_ORT_TEST_WITH_ORT=OFF
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build -L common_ort_to_light --output-on-failure

``tests/test_dependency_contract.cc`` lives in this component and checks
custom-domain model serialization and a real onnx-light custom kernel across
vector/matrix/scalar/empty shapes, including input preservation. It is labelled
``dependency``, not a unit test of a loader that does not exist yet.

The first implementation must add API negotiation, opaque-value ownership,
registration failure, loader teardown and independent fixture-plugin tests
alongside the actual bridge code. See :doc:`../newt_steps/2026/2026-10-start` for the API inventory
and its no-ORT-runtime acceptance criteria.
