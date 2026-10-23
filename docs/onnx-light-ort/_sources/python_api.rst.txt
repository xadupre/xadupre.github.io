Python API
==========

``onnx_light_ort`` is the single Python package for the project's four
interoperability directions:

.. list-table::
   :header-rows: 1

   * - Mechanism
     - ONNX Runtime host
     - onnx-light host
   * - Custom ops
     - ``register_custom_ops_light_called_by_ort``
     - ``CustomOpSessionOrtCalledByLight``
   * - Execution providers
     - Not implemented
     - Not implemented

The package does not replace ``onnx_light`` or ``onnx_light_cpu``.
Importing it does not import ONNX Runtime or onnx-light's model bindings.
ONNX Runtime is optional and needed only for the outgoing direction.
Python models use ``onnx_light.onnx``, never the ``onnx`` Python package.

Names identify the caller explicitly: ``light_called_by_ort`` means ORT is
the host; ``ort_called_by_light`` means onnx-light is the host.

Custom ops called by ONNX Runtime
---------------------------------

.. code-block:: python

   import onnxruntime as ort
   from onnx_light_ort import register_custom_ops_light_called_by_ort

   options = register_custom_ops_light_called_by_ort()
   session = ort.InferenceSession("model.onnx", sess_options=options)
   outputs = session.run(None, {"X": values})

``register_custom_ops_light_called_by_ort(options=None, *, library=None)`` returns
``onnxruntime.SessionOptions`` with the plugin registered. If options are
provided, the same object is updated without changing its settings, including
thread counts. ``library`` selects a trusted external plugin instead of the
bundled one. Registration errors propagate; there is no fallback.
Install the optional ``ort`` extra or supply your own ONNX Runtime installation.
This helper calls ONNX Runtime's own
``SessionOptions.register_custom_ops_library``; it does not register anything
in onnx-light.

The model must explicitly use ``onnx_light::LeakyRelu`` or
``onnx_light_cpu::Log`` with custom-domain opset 1. Registration does not
replace standard ONNX nodes with custom nodes.

Custom ops called by onnx-light
-------------------------------

.. code-block:: python

   from onnx_light_ort import CustomOpSessionOrtCalledByLight

   with CustomOpSessionOrtCalledByLight("model.onnx") as session:
       outputs = session.run(None, {"X": values})

``CustomOpSessionOrtCalledByLight(model, *, library=None, host_library=None)`` accepts a
filesystem path (string or path-like), serialized bytes or an onnx-light
``ModelProto``. Both libraries default to this project's build or wheel.
For an external ORT custom-op library, provide ``library="/trusted/plugin.so"``.
``host_library`` overrides the standalone host path.

This is not a general ONNX session. It supports a single audited API-18 CPU
custom node with one floating-point input and output. Models are limited to
16 MiB, tensors to 64 MiB and rank to 16. File reads are bounded before parsing.
See :doc:`components/bridge_ops_ort_called_by_light` for the exact profile.
External native plugins execute trusted code and must be audited against it.
This direction neither imports nor loads ONNX Runtime.
``CustomOpSession`` remains an alias for backward compatibility; new code should
use ``CustomOpSessionOrtCalledByLight`` to make the host explicit.

Native library resolution
-------------------------

``get_library_path("custom_ops")`` and ``get_library_path("host")`` return
absolute ``pathlib.Path`` objects. Wheels use their own ``_native`` directory.
Source checkouts read ``build/CMakeCache.txt`` to select the manifest generated
by CMake from the actual native targets, including platform-specific filenames.
Single-configuration generators use the active build type, such as
``build/Debug/python_libraries.json`` or ``build/python_libraries.json`` when
no build type is set. Multi-configuration generators use
``build/Release/python_libraries.json``. Manifests from other configurations
are never used as fallbacks.
No system library search or paths embedded in models are used.
Missing libraries and disabled components raise explicit errors.
Other build directories and non-Release multi-config builds require explicit
library paths.

For source development:

.. code-block:: bash

   python -m pip install -r requirements/host-tests.txt
   python setup.py build_ext --inplace --cpp-tests
   export PYTHONPATH="$PWD:$PWD/build/python:$PYTHONPATH"

This uses the existing ``build`` tree; it does not install or copy compiled
project artifacts into site-packages or the source package.
The onnx-light model bindings and native runtime use the same fetched revision.
Alternatively, add ``--onnx-light-source`` to link against the already-built
onnx-light runtime selected by ``PYTHONPATH``, without rebuilding or copying it.
See :doc:`development` for integration tests and release packaging.
