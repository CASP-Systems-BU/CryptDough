Random
======

.. include:: ../../include/core/random/README.md
   :parser: myst_parser.sphinx_

Permutations
------------
.. doxygenfile:: permutation_manager.h

.. doxygenclass:: cdough::random::ShardedPermutation
   :members:
   :protected-members:
   :undoc-members:
   :private-members:

.. doxygenclass:: cdough::random::HMShardedPermutation
   :members:
   :protected-members:
   :undoc-members:
   :private-members:

.. doxygenclass:: cdough::random::DMShardedPermutation
   :members:
   :protected-members:
   :undoc-members:
   :private-members:

.. doxygenclass:: cdough::random::DMShardedPermutationGenerator
   :members:
   :protected-members:
   :undoc-members:
   :private-members:

.. doxygenclass:: cdough::random::DMDummyGenerator
   :members:
   :protected-members:
   :undoc-members:
   :private-members:

.. doxygenclass:: cdough::random::DMPermutationCorrelationGenerator
   :members:
   :protected-members:
   :undoc-members:
   :private-members:

.. doxygenfile:: zero_permutation_generator.h

Randomness Generators
---------------------

.. doxygenfile:: random_generator.h
.. doxygenfile:: prg_algorithm.h
.. doxygenfile:: common_prg.h
.. doxygenfile:: seeded_prg.h
.. doxygenfile:: committed_seeds_queue.h
.. doxygenfile:: zero_rg.h

Correlation Generators
-----------------------
.. doxygenfile:: correlation_generator.h
.. doxygenfile:: ole_generator.h
.. doxygenfile:: ot_generator.h
.. doxygenfile:: dummy_ole.h
.. doxygenfile:: zero_ole.h
.. doxygenfile:: silent_ot.h
.. doxygenfile:: gilboa_ole.h
.. doxygenfile:: gilboa_mod_p.h
.. doxygenfile:: gilboa_crt.h
.. doxygenfile:: dpf.h
.. doxygenfile:: shprg.h
.. doxygenfile:: beaver_triple_generator.h
.. doxygenfile:: dummy_auth_triple_generator.h
.. doxygenfile:: dummy_auth_random_generator.h
.. doxygenfile:: zero_sharing_generator.h
.. doxygenfile:: oprf.h
.. doxygenfile:: registry.h

Utilities
---------
.. doxygenfile:: pooled_generator.h
.. doxygenfile:: manager.h



