Getting Started
===============

Installation
------------

Install from source:

.. code-block:: bash

   pip install .

Or with Pixi:

.. code-block:: bash

   pixi install
   pixi run install

Or build with CMake:

.. code-block:: bash

   cmake -S . -B build
   cmake --build build

Quick Start (C++)
-----------------

Include the public header and call ``RegisterImageKernels()`` once before
running any model that uses the ``ImageDecoder`` operator:

.. code-block:: cpp

   #include <onnx_light_kernel_images/register_image_kernels.h>

   int main() {
       onnx_light_kernel_images::RegisterImageKernels();
       // ... run models ...
   }

Quick Start (Python)
--------------------

.. code-block:: python

   from onnx_light_kernel_images.onnx_py._imgpykernels import register_image_kernels

   register_image_kernels()
