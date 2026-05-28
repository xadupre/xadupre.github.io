Getting Started
===============

This page explains how to install and build the project, both for regular users
and for developers who want to contribute or extend it.

.. contents:: Table of Contents
    :local:
    :depth: 2

For Users
---------

Prerequisites
~~~~~~~~~~~~~

- Python 3.10 or later
- ``pip``

Installation
~~~~~~~~~~~~

Install the package directly from `PyPI <https://pypi.org/project/yet-another-onnxruntime-extensions/>`_:

.. code-block:: bash

    pip install yet-another-onnxruntime-extensions

Verify the installation:

.. code-block:: python

    import yaourt
    print(yaourt.__version__)

Using Custom Operators (ortops)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The custom ONNX Runtime operators (``ortops``) require a compiled shared
library.  The library is shipped as a pre-built binary inside the wheel on
supported platforms, so no extra build step is needed when installing from
PyPI.

To load the library and register the operators with ONNX Runtime:

.. code-block:: python

    import onnxruntime as ort
    from yaourt.ortops import get_ort_ext_libs

    opts = ort.SessionOptions()
    opts.register_custom_ops_library(get_ort_ext_libs()[0])

For Developers
--------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.10 or later
- ``git``
- A C++ compiler supported by CMake (GCC, Clang, or MSVC)
- `CMake <https://cmake.org/>`_ 3.25 or later

Clone the Repository
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/xadupre/yet-another-onnxruntime-extensions.git
    cd yet-another-onnxruntime-extensions

Create a Virtual Environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate        # Linux / macOS
    # .venv\Scripts\activate         # Windows

Install the Package in Editable Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the Python package together with all development dependencies:

.. code-block:: bash

    pip install -e .[dev]

Build the C++ Custom Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ortops`` module contains custom ONNX Runtime operators implemented in
C++.  Configure and build them with CMake:

.. code-block:: bash

    cmake -S cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release

The compiled shared library is placed under ``build/`` and is automatically
discovered by :func:`yaourt.ortops.get_ort_ext_libs`.

Alternatively, the ``setup.py`` ``build_ext`` command integrates the CMake
build into the standard Python toolchain and copies the compiled library
directly into the source tree so that it is importable without adding
``build/`` to any search path:

.. code-block:: bash

    python setup.py build_ext --inplace

This is equivalent to running the two ``cmake`` commands above but places the
shared library alongside the Python sources automatically.

Using a Local Build of ONNX Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, CMake downloads the official ONNX Runtime release matching the
version set in ``cmake/load_externals.cmake``.  When you need to build against
a custom or unreleased version of ONNX Runtime, pass the **path to the
directory that contains the compiled shared library** (``libonnxruntime.so`` on
Linux, ``onnxruntime.dll`` on Windows, ``libonnxruntime.dylib`` on macOS) as
the ``ORT_VERSION`` CMake variable:

.. code-block:: bash

    cmake -S cmake -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DORT_VERSION=/path/to/onnxruntime/build/Linux/Release
    cmake --build build --config Release

CMake recognises that ``ORT_VERSION`` is a filesystem path rather than a
version string and uses the following layout (standard for an in-tree
ONNX Runtime build):

- **Library directory** – the value of ``ORT_VERSION`` itself.
- **Include directory** – ``<ORT_VERSION>/../../../include/onnxruntime/core/session``
  (resolves to the ``include/`` tree at the root of the ONNX Runtime source).

The same option is available when building through ``setup.py`` by setting the
``ORT_VERSION`` environment variable before invoking the command:

.. code-block:: bash

    ORT_VERSION=/path/to/onnxruntime/build/Linux/Release \
        python setup.py build_ext --inplace

On Windows use ``set`` instead:

.. code-block:: bat

    set ORT_VERSION=C:\onnxruntime\build\Windows\Release
    python setup.py build_ext --inplace

Run the Tests
~~~~~~~~~~~~~

Run the pure-Python test suite:

.. code-block:: bash

    pytest unittests

Run the ``ortops`` tests (requires the C++ build above):

.. code-block:: bash

    pytest unittests/ortops

Run all tests with coverage:

.. code-block:: bash

    pytest --cov=yaourt --cov-report=term-missing unittests

Code Style and Linting
~~~~~~~~~~~~~~~~~~~~~~

The project uses `black <https://black.readthedocs.io/>`_ for formatting and
`ruff <https://docs.astral.sh/ruff/>`_ for linting.  Run both before
committing:

.. code-block:: bash

    black .
    ruff check .

Build the Documentation
~~~~~~~~~~~~~~~~~~~~~~~

The HTML documentation is generated by `Sphinx <https://www.sphinx-doc.org/>`_
and includes the C++ API rendered via
`Doxygen <https://www.doxygen.nl/>`_ and
`Breathe <https://breathe.readthedocs.io/>`_.

Install the system and Python dependencies first:

.. code-block:: bash

    # Debian / Ubuntu
    sudo apt-get install -y doxygen

    # macOS (Homebrew)
    brew install doxygen

    pip install -e .[docs]

Then build the HTML documentation:

.. code-block:: bash

    python -m sphinx docs dist/html -j auto

Or use the convenience script at the root of the repository:

.. code-block:: bash

    bash make_doc.sh

The generated documentation is written to ``dist/html/``.
Doxygen runs automatically as part of the Sphinx build; no separate
invocation is needed.

Build the Python Wheel
~~~~~~~~~~~~~~~~~~~~~~

To produce a distribution wheel (e.g. for local testing):

.. code-block:: bash

    pip install build
    python -m build

The wheel and source distribution are placed in the ``dist/`` directory.
