# LoadGen FAQ {#ReadmeFAQ}

## Q: The LoadGen does not match the MLPerf specification. Who is right?
**A:**
The MLPerf spec is *always* right.
Please file a LoadGen bug so it may be resolved.

## Q: How can I file a bug?
**A:**
On GitHub: https://github.com/mlperf/inference/issues/new

## Q: Can I make local modifications to the LoadGen for submission?
**A:**
No. To keep the playing field level, please upstream any local
modificiations you need to make. Ideally upstream such changes behind a runtime
flag or via an abstract interface the client can implement. This will help
with testability.

## Q: Where can I find the results of a test?
**A:**
By default, the loadgen will output an *mlperf_log_summary.txt* file
that summarizes the target metrics and constraints of the test, along with
other stats about the run.

*Note:* LogSettings also has a flag to forward the results to stdout and
there's an outstanding TODO to make this more programmable.

## Q: The reference implementation for \<*some_model*\> prints out results of its own. Are those for submission?
**A:**
They are not. The LoadGen results are the ground truth for submission
results since they will work even for systems that forgo the python bindings.
If you notice a bug in the LoadGen's results, please file a bug or submit a
patch.

## Q: I'm getting linker errors for LoadgenVersion definitions. Where is *version_generated.cc*?
**A:**
If you have a custom build setup, make sure you run the *version_generator.py*
script, which will create the cc file you are looking for. The official build
files that come with the LoadGen do this for you out of the box.

## Q: What is this *version_generator.py* script?
**A:**
The LoadGen records git stats (if available) and the SHA1 of all its
source files (always) at build time for verification purposes. This is easy
to circumvent, but try your best to run *version_generator.py* correctly;
ideally integrated with your build system if you have a custom build.
The intention is more to help with debugging efforts and detect accidental
version missmatches than to detect bad actors.

## Q: How do I view the *mlperf_log_trace.json* file?
**A:**
This file uses the [Trace Event Format]
(https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit)
to record a timeline of all the threads involved.
You can view the file by typing [chrome://tracing](chrome://tracing) into
Chrome's address bar and dragging the json file there.
This file zips well and you can drag the zip file directly into
[chrome://tracing](chrome://tracing) too.
Please include zipped traces (and the other logs) when filing bug reports.

## Q: What is the difference between the MultiStream and MultiStreamFree scenarios?
**A:**
MultiStream corresponds to the official MLPerf scenario for submissions;
it has a fixed query rate and allows only one outstanding query at a time.
MultiStreamFree is implemented for evaluation purposes only; it sends queries
as fast as possible and allows up to N outstanding queries at a time. You may
want to use MultiStreamFree for development purposes since small improvements
in performance will always be reflected in the results, whereas MultiStream's
results will be quantized.

## Q: Why is the code littered with so many lambdas? My eyes hurt.
**A:**
Lambdas are a convenient and efficient way to ship arbitrary data + deferred
logic over to the logging thread without much boilerplate.
Much of the loadgen is built on top of the logging utilities.
Thus the lambdas. (Sorry about the eyes.)

## Q: What C++ version does the LoadGen target?
**A:**
It currently targets and requires C++14. It should compile with recent
versions of clang, gcc, and msvc.

## Q: What dependencies does the LoadGen code have?
**A:**
The C++ code has no external dependencies. The loadgen itself, logging
utilities, and unit test utilities are built solely on the C++ Standard Library.
The python bindings, however, do require
[pybind11](https://github.com/pybind/pybind11).
