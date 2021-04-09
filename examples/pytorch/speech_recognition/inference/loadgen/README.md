# Overview {#mainpage}

*Note:* A compiled html version of this document is hosted online
[here](https://mlperf.github.io/inference/loadgen/index.html).

## Introduction

* The LoadGen is a *reusable* module that *efficiently* and *fairly* measures
  the performance of inference systems.
* It generates traffic for scenarios as formulated by a diverse set of experts
  in the [MLPerf working group](https://mlperf.org/about).
* The scenarios emulate the workloads seen in mobile devices,
  autonomous vehicles, robotics, and cloud-based setups.
* Although the LoadGen is not model or dataset aware, its strength is in its
  reusability with logic that is.

## Integration Example and Flow
The following is an diagram of how the LoadGen can be integrated into an
inference system, resembling how some of the MLPerf reference models are
implemented.
<div style="display:flex; flex-flow:row wrap; justify-content: space-evenly;">
<img src="loadgen_integration_diagram.svg" width="500px" style="padding: 20px">
<ol style="padding: 20px">
<li>Benchmark knows the model, dataset, and preprocessing.</li>
<li>Benchmark hands dataset sample IDs to LoadGen.</li>
<li>LoadGen starts generating queries of sample IDs.</li>
<li>Benchmark creates requests to backend.</li>
<li>Result is post processed and forwarded to LoadGen.</li>
<li>LoadGen outputs logs for analysis.<br>
</ol>
</div>

## Useful Links
* [FAQ](@ref ReadmeFAQ)
* [LoadGen Build Instructions](@ref ReadmeBuild)
* [LoadGen API](@ref LoadgenAPI)
* [Test Settings](@ref LoadgenAPITestSettings) -
  A good description of available scenarios, modes, and knobs.
* [MLPerf Inference Code](https://github.com/mlperf/inference) -
  Includes source for the LoadGen and reference models that use the LoadGen.
* [MLPerf Inference Rules](https://github.com/mlperf/inference_policies) -
  Any mismatch with this is a bug in the LoadGen.
* [MLPerf Website](www.mlperf.org)

## Scope of the LoadGen's Responsibilities

### In Scope
* **Provide a reusable** C++ library with python bindings.
* **Implement** the traffic patterns of the MLPerf Inference scenarios and
  modes.
* **Record** all traffic generated and received for later analysis and
  verification.
* **Summarize** the results and whether performance constraints were met.
* **Target high-performance** systems with efficient multi-thread friendly
  logging utilities.
* **Generate trust** via a shared, well-tested, and community-hardened
  code base.

### Out of Scope
The LoadGen is:
* **NOT** aware of the ML model it is running against.
* **NOT** aware of the data formats of the model's inputs and outputs.
* **NOT** aware of how to score the accuracy of a model's outputs.
* **NOT** aware of MLPerf rules regarding scenario-specific constraints.

Limitting the scope of the LoadGen in this way keeps it reusable across
different models and datasets without modification. Using composition and
dependency injection, the user can define their own model, datasets, and
metrics.

Additionally, not hardcoding MLPerf-specific test constraints, like test
duration and performance targets, allows users to use the LoadGen unmodified
for custom testing and continuous integration purposes.

## Submission Considerations

### Upstream all local modifications
* As a rule, no local modifications to the LoadGen's C++ library are allowed
for submission.
* Please upstream early and often to keep the playing field level.

### Choose your TestSettings carefully!
* Since the LoadGen is oblivious to the model, it can't enforce the MLPerf
requirements for submission. *e.g.:* target percentiles and latencies.
* For verification, the values in TestSettings are logged.
* To help make sure your settings are spec compliant, use
TestSettings::FromConfig in conjunction with the relevant config file provided
with the reference models.

## Responsibilities of a LoadGen User

### Implement the Interfaces
* Implement the SystemUnderTest and QuerySampleLibrary interfaces and pass
  them to the StartTest function.
* Call QuerySampleComplete for every sample received by
  SystemUnderTest::IssueQuery.

### Assess Accuracy
* Process the *mlperf_log_accuracy.json* output by the LoadGen to determine
  the accuracy of your system.
* For the official models, Python scripts will be provided by the MLPerf model
  owners for you to do this automatically.

For templates of how to do the above in detail, refer to code for the demos,
tests, and reference models.
