Operators
=========

.. include:: ../../include/core/operators/README.md
   :parser: myst_parser.sphinx_

Note: For documentation on the join functions, see :cpp:func:`cdough::relational::EncodedTable::_join`.

Boolean Circuits
----------------

.. doxygenfile:: circuits.h

Shuffle
-------
.. doxygenfile:: shuffle.h

Sorting
-------
.. doxygenfile:: sorting.h
.. doxygenfile:: sorting_network.h
.. doxygenfile:: quicksort.h
.. doxygenfile:: radixsort.h

Merge
-----
.. doxygenfile:: merge.h

Relational
----------
.. doxygenfile:: aggregation.h
.. doxygenfile:: distinct.h
.. doxygenfile:: aggregation_selector.h
.. doxygenfile:: join.h
.. doxygenfile:: prefix_network.h

.. doxygenstruct:: cdough::relational::JoinOptions
.. doxygenstruct:: cdough::relational::AggregationOptions

Machine Learning
----------------
.. doxygenfile:: machine_learning.h

Streaming
---------
.. doxygenfile:: streaming.h

Common
------
.. doxygenfile:: common.h