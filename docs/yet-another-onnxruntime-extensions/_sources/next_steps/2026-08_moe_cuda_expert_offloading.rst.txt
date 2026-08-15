.. _l-next-steps-moe-cuda-expert-offloading:

Adaptive CUDA expert offloading for Qwen 3.6 MoE
================================================

:Date: 2026-08

**discussion**

Objective
+++++++++

Determine, before implementing a CUDA cache, which expert-placement strategy is
worth implementing for Qwen 3.6 and other Mixture-of-Experts models. The first
deliverable is an expert-routing logger, followed by measured traces and an
offline cache simulator. Only the strategy selected from those simulations is
implemented in the CUDA operator.

The eventual operator must execute models whose expert weights do not all fit
in GPU memory. CPU memory keeps the canonical copy of every expert. A bounded
number of experts are also cached on CUDA, without removing their CPU copy, so
an evicted expert remains immediately available to the CPU path.

The first adaptive policy counts how often every expert is selected. During
each autoregressive inference step, which produces one new token, experts used
by the router have their counters incremented as soon as the routing decision
is available. Experts with the largest counters become candidates for CUDA
residency. When that decision changes the cache, the corresponding CPU weights
are copied to CUDA immediately. The transfer should overlap the remainder of
the current token computation and finish before the same layer processes the
next token. The experiment must first establish through trace simulation whether this
policy is at least equivalent to a static placement and whether expert-routing
sequences suggest a better policy.

This idea is close to the activation-aware caching explored by
`MoE-Infinity <https://arxiv.org/abs/2401.14361>`_. Its specific objective here
is to test the cheapest useful model-agnostic policy: integer counters, a
bounded ranking, and asynchronous copies triggered by observed routing. It
requires neither predictor training, model-specific calibration, a database of
past request traces, nor prior knowledge of which experts are important. The
same cache manager should work for any MoE model that exposes stable layer and
expert identifiers. Only the expert tensor layout and execution kernel remain
model-specific.

Scope
+++++

The implementation targets the Qwen 3.6 MoE graph and its top-k routing
pattern. Expert identity is the pair ``(layer_id, expert_id)`` because experts
from different layers do not share weights or statistics.

The investigation includes, in order:

* opt-in logging of the complete expert-selection sequence;
* reproducible benchmarks that collect routing traces before cache
  implementation;
* an offline simulator for static and cumulative adaptive placement;
* a capacity sweep over the number of experts allowed on CUDA;
* a literature review and trace-driven evaluation of better policies;
* implementation of the selected strategy, including the fused CUDA MoE
  operator, persistent CPU weights, fixed-capacity CUDA slots, and CPU fallback.

Training, changes to router logits, and expert-weight quantization are outside
the first implementation. The operator must preserve the model output within
the tolerance of the existing CPU or ONNX Runtime implementation.

Memory and execution model
++++++++++++++++++++++++++

Each expert has one permanent CPU allocation. The CUDA cache owns
``gpu_expert_capacity`` slots, where one slot stores all weights needed to
execute one expert. CPU weights are never moved or released when the expert is
copied to a slot.

.. code-block:: text

    CPU expert weights (canonical, always resident)
        expert 0 ───────────────────────────┐
        expert 1 ──────────────┐            │
        ...                    │ copy       │ copy
        expert N ───────┐      v            v
                        │  CUDA slot 0   CUDA slot 1  ...  CUDA slot C-1
                        └─ CPU fallback

The operator is configured with the number of experts that may reside on CUDA,
not with a placement policy. Cache policy, counters, and transfers belong to a
runtime cache manager. The execution path receives a current immutable snapshot
of the expert-to-slot mapping:

* a cache hit dispatches the token to the CUDA expert in its assigned slot;
* a cache miss executes the current token from the CPU weights;
* when the cache policy admits that expert, its CPU weights are copied to the
  reserved CUDA slot immediately on a dedicated transfer stream;
* a slot cannot be reused until all CUDA work referencing its previous expert
  has completed.

The cache manager must not block the current token merely to make an expert
resident. The CPU fallback and the host-to-device copy may read the same
immutable CPU weights concurrently. If the destination slot is still in use,
the transfer stream first waits for the previous expert's completion event and
then starts the copy as soon as the slot is safe.

The new expert is marked as ``loading`` while the copy is in flight. Recording
the transfer-completion event does not delay the current token. Before the next
token reaches that MoE layer, the manager queries the event:

* if the copy is complete, the manager publishes the new expert-to-slot mapping
  and the next token uses CUDA;
* if the copy is incomplete, the mapping remains unavailable and the next token
  uses the CPU fallback again rather than waiting;
* once the event completes, the following mapping snapshot publishes the slot.

The intended timeline is therefore:

.. code-block:: text

    token t, layer L router selects expert E
        -> increment count(L, E)
        -> adaptive policy admits E and reserves a CUDA slot
        -> enqueue CPU-to-CUDA weight copy immediately
        -> execute the current miss on CPU while the copy progresses
        -> finish the remaining layers of token t
    token t+1, before layer L
        -> publish E if the copy event is complete
        -> execute E on CUDA on a hit, otherwise use CPU without blocking

This gives the copy the largest practical overlap window: from the routing
decision at layer ``L`` for token ``t`` until layer ``L`` is reached for token
``t + 1``. A simpler synchronous transfer mode is retained only for correctness
tests and transfer-cost calibration.

Adaptive placement
++++++++++++++++++

The baseline adaptive strategy is cumulative least-frequently-used placement.
For token inference ``t``, every layer router produces its selected experts.
As soon as a layer router returns, the manager computes the number of
selections for that layer and updates:

.. math::

    count_t(e) = count_{t-1}(e) + uses_t(e)

Immediately after updating the counters, the manager compares resident and
non-resident experts. The most frequently used experts are admitted until the
CUDA capacity is reached. A replacement occurs only when a non-resident expert
has a strictly higher score than the least-used resident expert. Deterministic
``(layer_id, expert_id)`` ordering breaks ties.

Admission reserves a slot and enqueues the weight copy immediately; it does not
wait until the end of the token. Residency changes only when the copy event
completes, so kernels already in flight keep their immutable mapping snapshot.
This distinction between ``loading`` and ``resident`` prevents a partially
copied expert from becoming visible while still maximizing overlap with the
current token.

Counters and residency are session state. They can be reset before every
benchmark repetition, exported with the logs, and optionally initialized from
a previous trace. The following extensions are evaluated only after the
cumulative policy is measured:

* exponentially decayed frequency, to adapt when the workload changes;
* a sliding-window LFU policy;
* LRU or frequency-plus-recency scoring;
* separate budgets and counters per MoE layer;
* transition-aware prefetching based on recent expert sequences;
* prompt- or workload-conditioned placement.

Static baseline
+++++++++++++++

The static strategy places the same experts on CUDA for the entire run. It uses
the same CUDA capacity, CPU fallback, kernels, tensor types, batch size, and
transfer accounting as the adaptive strategy.

The primary baseline is deliberately computed *a posteriori* from the benchmark
being evaluated. It is an oracle-like simulated static baseline, not a
placement learned from a separate calibration set:

1. run the complete benchmark once with expert-sequence logging enabled;
2. after that run has finished, aggregate the number of uses of every
   ``(layer_id, expert_id)`` over the benchmark;
3. select the ``gpu_expert_capacity`` most frequently used experts, with
   deterministic layer/expert ordering for ties;
4. replay the trace in the simulator with exactly those experts resident;
5. keep that simulated placement unchanged for the complete trace.

This gives the static strategy knowledge of the complete future benchmark and
no in-run transfer cost. It is therefore an intentionally strong, optimistic
baseline: the adaptive policy should not appear better merely because static
experts were chosen poorly. A final end-to-end run with the selected strategy
is performed only after the simulation and decision phases.

A **cold static** placement using the first ``gpu_expert_capacity`` experts in
deterministic layer/expert order may also be reported as a sanity check, but it
is not the baseline used to decide whether adaptation is beneficial.

Expert-sequence logging
+++++++++++++++++++++++

Logging is opt-in and asynchronous. The inference path writes compact records
to a bounded in-memory buffer; a background writer serializes them as JSON
Lines for inspection and optionally as Parquet for larger analyses. Buffer
overflow is reported explicitly rather than silently dropping records.

Each routing record contains:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``run_id``, ``request_id``
     - Reproducible benchmark and inference identifiers.
   * - ``token_index``, ``layer_id``
     - Position of the routing decision in the generated sequence.
   * - ``expert_ids``, ``router_weights``
     - Ordered top-k experts and their router weights.
   * - ``resident_experts``
     - Placement snapshot used by this inference.
   * - ``cache_hit``
     - Per selected expert, whether CUDA executed it.
   * - ``admitted``, ``evicted``
     - Placement changes decided immediately after the routing update.
   * - ``copy_enqueued_ns``, ``copy_ready_ns``
     - Copy interval and whether the expert became ready before the next token.
   * - ``copy_bytes``, ``copy_duration_ns``
     - Host-to-device traffic attributable to cache changes.
   * - ``cpu_duration_ns``, ``cuda_duration_ns``
     - Execution time split by device.

The run metadata records the model revision, CUDA and ONNX Runtime versions,
GPU and CPU models, capacity, policy parameters, random seed, warm-up length,
and benchmark configuration. Prompts and generated text are not logged by
default; stable request identifiers are sufficient to join routing traces with
benchmark metadata when explicitly required.

Measurements on logged data
++++++++++++++++++++++++++++

The first analysis measures expert frequency, reuse distance, cache hit rate,
copies, evictions, CPU fallbacks, and estimated latency for every CUDA capacity.

It also compares cumulative counts with request resets, sliding windows, and
exponential decay. The goal is to determine the useful history length and detect
when the adaptive ranking converges to hindsight-static placement. Selection is
performed on a trace prefix and evaluated on held-out tokens and workloads.

The second analysis is separate. It measures:

* ``P(expert_n | expert_{n-1})`` and mutual information between adjacent layers;
* prediction accuracy and copy lead time for inter-layer prefetching;
* the relation between experts in layer ``n`` and the top-1 token predicted
  from layer ``n - 1``;
* the additional gain of the token signal over expert correlation alone.

The intermediate top-1 token requires an extra projection through the final
normalization and language-model head. It is logged only on sampled tokens or
computed offline, and its cost is excluded from policy timing.

Benchmarks
++++++++++

At least two workloads with different routing locality are required:

1. **Long single-request generation.** Run a long-context benchmark such as
   LongBench with batch size one and enough generated tokens to observe whether
   expert popularity stabilizes within a request.
2. **Heterogeneous multi-request generation.** Run a shuffled collection of
   unrelated conversational or instruction prompts, such as a ShareGPT-derived
   trace, to measure adaptation across changing topics and requests.

The benchmark phase does not require an adaptive cache implementation. It runs
the unmodified or instrumented model to collect complete routing traces for the
same prompts and generated-token limits. It also measures CPU expert time, CUDA
expert time when available, and host-to-device bandwidth so the simulator can
translate hits, misses, and copies into an estimated execution cost.

The simulator then replays every trace with both hindsight static and cumulative
adaptive placement. It sweeps ``gpu_expert_capacity`` from zero to all experts,
using a dense set of small capacities and representative larger capacities.
Both strategies receive exactly the same trace, capacity, expert sizes, and
measured transfer-cost model.

The benchmark and simulation report:

* expert-frequency distributions and complete routing sequences;
* simulated CUDA cache-hit rate per layer and overall;
* simulated host-to-device bytes, transfer count, and copy completion before
  the next token;
* simulated admissions, evictions, and CPU fallbacks;
* estimated latency and throughput from measured CPU, CUDA, and transfer costs;
* the capacity at which each strategy reaches a target hit rate;
* uncertainty or sensitivity when copy and execution costs vary.

After implementation, end-to-end validation adds time to first token,
inter-token latency, throughput, peak CPU and CUDA memory, and output agreement.
Kernel-only timing is reported separately but is not the decision metric.

Sequence analysis
+++++++++++++++++

The logs are analyzed offline before any cache policy is implemented. The
analysis measures:

* expert-frequency concentration and the CUDA capacity needed to cover a given
  fraction of selections;
* frequency drift between requests and between early and late generation;
* run lengths, reuse distance, and cache-hit rate achievable by LRU and LFU;
* per-layer differences in expert popularity;
* first-order and higher-order transitions between selected experts;
* correlation between router weight and near-future reuse;
* an offline optimal cache trace as an upper bound.

After the static/adaptive comparison, a literature review covers MoE expert
offloading, caching, prefetching, and sequence-aware replacement. Candidate
policies from that review are replayed against the same recorded sequences
without rerunning the model. Replay accounts for the configured capacity,
expert weight size, transfer cost, and the rule that CPU weights remain
resident. Policies that improve simulated hit rate but trigger excessive
transfers are rejected. Workload-specific improvements remain documented as
such.

What the literature says today
++++++++++++++++++++++++++++++

The literature does not identify one replacement policy that dominates for
every MoE model, workload, cache capacity, and hardware configuration. It does,
however, consistently show that expert routing contains exploitable structure
and that transfer scheduling is at least as important as replacement policy.

.. list-table::
   :header-rows: 1
   :widths: 24 31 45

   * - Work
     - Main mechanism
     - Implication for this plan
   * - `Fast Inference of Mixture-of-Experts Language Models with Offloading
       <https://arxiv.org/abs/2312.17238>`_ (Eliseev and Mazur, 2023)
     - Keeps expert weights in CPU memory, uses a GPU expert cache, and exploits
       temporal locality with LRU-style caching and speculative loading.
     - Recency is a necessary simulated baseline. Consecutive-token locality
       must be measured rather than assumed for Qwen 3.6.
   * - `MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE
       Serving <https://arxiv.org/abs/2401.14361>`_ (Xue et al., 2024)
     - Builds a layer-by-expert activation matrix for the current request,
       matches it by cosine distance against a bounded collection of historical
       request matrices, aggregates similar traces into per-expert likelihoods,
       weights predictions by layer proximity, and uses them for eviction and
       next-layer prefetching.
     - This is the closest published approach to the proposed idea. The present
       experiment deliberately studies a narrower and cheaper model-agnostic
       variant based only on online counters and immediate copies, without
       historical-trace storage, similarity search, aggregation, or a
       prediction matrix. MoE-Infinity is also training-free; the distinction
       is runtime complexity and state, not learned versus non-learned
       prediction. This plan also adds a hindsight-static comparison on the
       exact benchmark traces.
   * - `Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts
       Models <https://arxiv.org/abs/2402.07033>`_ (Kamahori et al., ICLR 2025)
     - Executes non-resident experts on CPU and transfers activations instead of
       synchronously transferring expert weights for every miss.
     - Supports the planned non-blocking CPU fallback. A simulator must compare
       CPU execution cost with weight-copy cost, not model every miss as a
       mandatory stall.
   * - `SiDA-MoE: Sparsity-Inspired Data-Aware Serving for Efficient and
       Scalable Large Mixture-of-Experts Models
       <https://arxiv.org/abs/2310.18859>`_ (MLSys 2024)
     - Predicts expert use from input-dependent information and prefetches
       experts before they are needed.
     - Expert popularity can depend on the workload. A policy learned from one
       benchmark may not generalize to another, which justifies the two distinct
       benchmark classes.
   * - `Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable
       Mixture-of-Expert Inference
       <https://arxiv.org/abs/2308.12066>`_ (ISCA 2024)
     - Adds a predictor that exposes routing decisions early enough to overlap
       expert transfer with preceding computation.
     - Correct prediction is insufficient without enough lead time. The trace
       simulator should model when a decision becomes available and whether the
       copy finishes before the target layer.
   * - `ExFlow: Exploiting Inter-Layer Expert Affinity for Mixture-of-Experts
       Model Inference <https://arxiv.org/abs/2401.08383>`_ (2024)
     - Profiles conditional affinities between experts selected in consecutive
       MoE layers and uses them for placement in a multi-GPU setting.
     - Logging must preserve layer order, not only global counts. Conditional
       transitions may support a better policy than cumulative frequency even
       though ExFlow itself targets GPU-to-GPU communication.
   * - `ProMoE: Fast MoE-based LLM Serving using Proactive Caching
       <https://arxiv.org/abs/2410.22134>`_ (Song et al., 2024)
     - Uses learned expert prediction, proactive caching, and chunked
       asynchronous transfers.
     - Chunked and cancellable copies are a candidate implementation refinement.
       A learned predictor has model-specific training and runtime costs that
       must be included in the comparison.
   * - `HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE
       Inference <https://arxiv.org/abs/2411.01433>`_ (Tang et al., 2024)
     - Combines token-, layer-, and sequence-level decisions with
       mixed-precision expert representations.
     - Frequency alone may miss useful structure at several time scales.
       Mixed-precision fallback is an alternative, but changes the numerical
       contract and is outside the initial exact-weight experiment.
   * - `Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware
       Multi-Batch Pipeline <https://arxiv.org/abs/2502.06888>`_ (2025)
     - Overlaps expert movement and computation across multiple batches and
       heterogeneous memory tiers.
     - Results obtained with several requests in flight may not predict
       single-request token latency. The two cases must remain separate in the
       benchmark report.
   * - `HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient
       MoE Inference <https://arxiv.org/abs/2504.05897>`_ (Zhong et al., 2025)
     - Combines CPU/GPU execution, impact-aware prefetching, and score-based
       cache management.
     - A replacement score should eventually account for miss cost and expected
       reuse, not frequency alone. This is a strong candidate family for
       trace-based simulation.

The following conclusions are sufficiently common across these works to guide
the experiment:

* **Expert use is skewed, but not universally stationary.** A small hot set
  often exists, which makes hindsight-static placement strong. Its composition
  may change with the model, layer, input domain, request, or generation phase.
  This is why a static placement computed after each benchmark is an important
  oracle baseline but not automatically a deployable policy.
* **Temporal locality matters.** Recency-based policies and prediction from
  nearby tokens can outperform a global frequency ranking when the active set
  changes. The simulator should therefore include LRU, cumulative LFU, and a
  windowed or exponentially decayed LFU in addition to the two required
  strategies.
* **Routing is layer-specific.** Global expert counts lose the ordering and
  conditional information exploited by inter-layer approaches. Statistics,
  capacities, and transition matrices must be available per layer even if the
  first policy uses one global capacity.
* **A cache hit is useful only when the weights are ready in time.** Predictive
  systems obtain their gains by starting copies early and overlapping them with
  useful work. Simulations must track enqueue time, available overlap, transfer
  completion, and fallback cost; hit rate alone is not a sufficient metric.
* **CPU execution is a credible miss path.** Fiddler and hybrid systems show
  that transferring activations and computing on CPU can be preferable to
  waiting for a large expert-weight transfer, especially at small batch size.
  This supports preserving CPU weights permanently.
* **More complex predictors have a portability cost.** Input-aware, learned,
  and inter-layer predictors can improve prefetching, but may require
  model-specific profiling or training. A cumulative or decayed adaptive policy
  remains attractive when it is statistically equivalent because it can be
  reused across models without determining a static expert set for each one.
* **Capacity changes the conclusion.** At large capacity, simple policies tend
  to converge because most useful experts fit. At small capacity, replacement
  and prediction matter more, while excessive churn can erase gains through
  host-to-device traffic. The capacity sweep is therefore a core experiment,
  not a secondary tuning exercise.

Current evidence makes cumulative LFU a reasonable generic baseline, not a
foregone implementation choice. The minimum literature-informed simulation set
is hindsight static, cumulative LFU, LRU, decayed or windowed LFU, and an
offline optimal replacement bound. Inter-layer transition prediction and
impact-aware scoring are the first more advanced candidates. The literature
review must be refreshed before implementation because this is an active area,
and preprints must be distinguished from peer-reviewed results.

The hypothesis under test is therefore not that frequency-aware caching has
never been proposed. It is that a particularly inexpensive implementation,
using only routing counters and immediate asynchronous copies, may be sufficient
to match or beat model-specific static placement across several models and
workloads. If confirmed, its value is operational simplicity and portability;
if rejected, the traces quantify when model-specific profiling or a more
expensive predictor becomes necessary.

Correctness and concurrency
+++++++++++++++++++++++++++

The cache manager owns all mutable state and exposes a mapping snapshot for one
inference. Concurrent requests may share immutable CPU weights but must not
mutate a session cache without synchronization. The initial implementation
serializes placement updates; parallel execution can later use versioned
snapshots and per-slot events.

Tests cover:

* capacity zero, partial capacity, and capacity covering every expert;
* deterministic admission and eviction, including ties;
* repeated hits without additional copies;
* eviction without releasing or modifying CPU weights;
* CPU fallback while an asynchronous copy is in flight;
* immediate copy enqueue when an adaptive update reserves a slot;
* publication before the next token when the copy event has completed;
* non-blocking CPU fallback when the copy misses the next-token deadline;
* safe slot reuse after CUDA completion;
* counter reset, export, and replay;
* complete logs and explicit buffer-overflow errors;
* numerical agreement for mixed CPU/CUDA expert execution.

Implementation plan
+++++++++++++++++++

First part: static versus adaptive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Implement expert-sequence logging without a cache.
2. Collect traces on both benchmarks.
3. Implement the offline cache simulator.
4. Sweep CUDA capacity and history length for hindsight-static and adaptive
   placement.
5. Keep adaptive as the generic candidate if it is better or equivalent;
   otherwise retain model-specific static placement.

Second part: predictive strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This part starts only after the first decision.

1. Extend logging with sampled top-1 tokens from intermediate layers.
2. Compute adjacent-layer expert correlations and token/expert correlations.
3. Simulate inter-layer prefetching on held-out traces, including copy lead
   time.
4. Review the literature and add relevant predictive policies to the simulator.
5. Select a predictive policy only if it consistently improves the first-part
   candidate.

Final implementation
^^^^^^^^^^^^^^^^^^^^

1. Implement only the selected policy.
2. Add fixed-capacity CUDA slots, persistent CPU weights, immediate asynchronous
   copies, next-token publication, CPU fallback, and the fused Qwen 3.6 kernel.
3. Rerun both benchmarks end to end and compare measurements with simulation.

Decision criteria
+++++++++++++++++

The experiment first succeeds by producing trustworthy traces and enough
evidence to choose a placement strategy before implementing it. Cumulative
adaptive placement is considered competitive when it is better than or
equivalent to hindsight static placement, within a predefined uncertainty
margin, across both benchmark classes and the useful capacity range. Its main
advantage is then portability: it does not require determining a different
expert set for every model.

If adaptive placement is worse, the result indicates that expert residency
must be profiled per model or workload unless the literature review identifies
a better generic policy. The final success criterion is a correct
bounded-memory MoE operator whose measured end-to-end behavior confirms the
simulation for the selected strategy.
