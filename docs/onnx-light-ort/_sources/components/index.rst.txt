Projects
========

Each project has its own page, source directory and README. Implemented targets
own their CMake definitions and native tests; integration tests additionally
exercise the real ORT host. Planned components are identified explicitly,
without placeholder tests that would imply an implementation exists.

.. toctree::
   :maxdepth: 1

   custom_ops
   custom_ep
   bridge_ops_light_called_by_ort
   bridge_ep_light_called_by_ort
   bridge_ops_ort_called_by_light
   bridge_ep_ort_called_by_light
   common_light_to_ort
   common_ort_to_light
