<div align="center">
<a href="https://k2-fsa.github.io/k2/">
  <img src="https://raw.githubusercontent.com/k2-fsa/k2/master/docs/source/_static/logo.png" width=88>
</a>

<br/>

[![Documentation Status](https://github.com/k2-fsa/k2/actions/workflows/build-doc.yml/badge.svg)](https://k2-fsa.github.io/k2/)

</div>

# k2

## Experimental ROCm/HIP Port

This repository includes an experimental ROCm/HIP port of `k2` for AMD GPUs.
The goal of this port is to make the existing CUDA-oriented `k2` codebase build
and run on ROCm with minimal changes to the higher-level Python and PyTorch
interfaces.

### Scope

- Build `k2` with ROCm/HIP while keeping the existing `torch.cuda`-style Python
  API expected by downstream projects.
- Preserve the main `k2` Python extension layout and `setup.py install`
  workflow.
- Keep the original CUDA path intact while adding a ROCm-specific build path.

### Validated Environment

- GPU: AMD Instinct MI210
- ROCm userland validated: `6.3.3`
- PyTorch: `2.9.1+rocm6.3`
- Python: `3.10`
- Compiler: ROCm `amdclang++`

### Main Porting Areas

- CMake and `setup.py` changes to detect ROCm PyTorch builds and configure a
  HIP toolchain.
- CUDA-to-HIP compatibility shims for runtime headers and common CUDA APIs.
- GPU source adaptations in `k2/csrc` and `k2/python/csrc` to compile with HIP.
- CUB/rocPRIM and iterator-related fixes needed by GPU kernels and reductions.
- Packaging fixes so `python setup.py install` produces a usable installed
  package in a ROCm PyTorch environment.

### Build Notes

The clean install path that was validated is:

```bash
export ROCM_PATH=/path/to/rocm
export K2_MAKE_ARGS=-j1
python setup.py install
```

Notes:

- `-j1` was used for a conservative, stable build during validation.
- The build compiles the `k2/torch/bin` targets as part of the normal install,
  which increases build time substantially.
- ROCm still uses the `torch.cuda` API surface in PyTorch, so Python-side code
  does not need to switch to a different device namespace.

### Validation Summary

The installed ROCm build was validated with:

- `import k2`
- `python -m k2.version`
- `k2.swoosh_l(...)` on GPU
- `k2.linear_fsa(...)` on GPU
- `k2.arc_sort(...)` on GPU
- `k2.ctc_topo(...)` on GPU
- `k2.mutual_information_recursion(...)` forward and backward on GPU

### Numerical Checks

CPU vs ROCm results from the current validation:

- `linear_fsa`: exact match
- `ctc_topo`: exact match for arcs and scores
- `swoosh_l`: max absolute difference `2.98e-08`
- `mutual_information_recursion`:
  - forward max absolute difference `0.0`
  - gradient max absolute differences about `6e-07`

These results are consistent with expected floating-point backend variance for
the tested kernels. This is good evidence that the port is numerically healthy
for the exercised paths, but it is not yet a full-library equivalence proof.

### Current Limitations

- This is not a full upstream test-suite validation.
- Validation was performed on a ROCm `6.3.3` userland, not on a direct ROCm
  `6.6.3` build target.
- The port has not yet been benchmarked against the original NVIDIA/CUDA path
  for performance.
- Multi-node and long-running production training validation were not part of
  this repository-level verification.

The vision of k2 is to be able to seamlessly integrate Finite State Automaton
(FSA) and Finite State Transducer (FST) algorithms into autograd-based machine
learning toolkits like PyTorch and TensorFlow.  For speech recognition
applications, this should make it easy to interpolate and combine various
training objectives such as cross-entropy, CTC and MMI and to jointly optimize a
speech recognition system with multiple decoding passes including lattice
rescoring and confidence estimation.  We hope k2 will have many other
applications as well.

One of the key algorithms that we have implemented is
pruned composition of a generic FSA with a "dense" FSA (i.e. one that
corresponds to log-probs of symbols at the output of a neural network).  This
can be used as a fast implementation of decoding for ASR, and for CTC and
LF-MMI training.  This won't give a direct advantage in terms of Word Error Rate when
compared with existing technology; but the point is to do this in a much more
general and extensible framework to allow further development of ASR technology.

## Implementation

 A few key points on our implementation strategy.

 Most of the code is in C++ and CUDA.  We implement a templated class `Ragged`,
 which is quite like TensorFlow's `RaggedTensor` (actually we came up with the
 design independently, and were later told that TensorFlow was using the same
 ideas).  Despite a close similarity at the level of data structures, the
 design is quite different from TensorFlow and PyTorch.  Most of the time we
 don't use composition of simple operations, but rely on C++11 lambdas defined
 directly in the C++ implementations of algorithms.  The code in these lambdas operate
 directly on data pointers and, if the backend is CUDA, they can run in parallel
 for each element of a tensor.  (The C++ and CUDA code is mixed together and the
 CUDA kernels get instantiated via templates).

 It is difficult to adequately describe what we are doing with these `Ragged`
 objects without going in detail through the code.  The algorithms look very
 different from the way you would code them on CPU because of the need to avoid
 sequential processing.  We are using coding patterns that make the most
 expensive parts of the computations "embarrassingly parallelizable"; the only
 somewhat nontrivial CUDA operations are generally reduction-type operations
 such as exclusive-prefix-sum, for which we use NVidia's `cub` library.  Our
 design is not too specific to the NVidia hardware and the bulk of the code we
 write is fairly normal-looking C++; the nontrivial CUDA programming is mostly
 done via the cub library, parts of which we wrap with our own convenient
 interface.

 The Finite State Automaton object is then implemented as a Ragged tensor templated
 on a specific data type (a struct representing an arc in the automaton).

## Autograd

 If you look at the code as it exists now, you won't find any references to
 autograd.  The design is quite different to TensorFlow and PyTorch (which is
 why we didn't simply extend one of those toolkits).  Instead of making autograd
 come from the bottom up (by making individual operations differentiable) we are
 implementing it from the top down, which is much more efficient in this case
 (and will tend to have better roundoff properties).

 An example: suppose we are finding the best path of an FSA, and we need
 derivatives.  We implement this by keeping track of, for each arc in the output
 best-path, which input arc it corresponds to.  (For more complex algorithms an arc
 in the output might correspond to a sum of probabilities of a list of input arcs).
 We can make this compatible with PyTorch/TensorFlow autograd at the Python level,
 by, for example, defining a Function class in PyTorch that remembers this relationship
 between the arcs and does the appropriate (sparse) operations to propagate back the
 derivatives w.r.t. the weights.

## Current state of the code

 We have wrapped all the C++ code to Python with [pybind11](https://github.com/pybind/pybind11)
 and have finished the integration with [PyTorch](https://github.com/pytorch/pytorch).

 We are currently writing speech recognition recipes using k2, which are hosted in a
 separate repository. Please see <https://github.com/k2-fsa/icefall>.

## Plans after initial release

 We are currently trying to make k2 ready for production use (see the branch
 [v2.0-pre](https://github.com/k2-fsa/k2/tree/v2.0-pre)).

## Quick start

Want to try it out without installing anything? We have setup a [Google Colab][1].
You can find more Colab notebooks using k2 in speech recognition at
<https://k2-fsa.github.io/icefall/recipes/Non-streaming-ASR/librispeech/conformer_ctc.html>.

[1]: https://colab.research.google.com/drive/1qbHUhNZUX7AYEpqnZyf29Lrz2IPHBGlX?usp=sharing

## Acknowledgements

Some contributors to this project were supported by Xiaomi Corporation. Others were supported by National Science Foundation CCRI award 2120435.  This is not an exhaustive list of sources of support.
