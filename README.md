# PortableNet.jl

A tiny neural network that trains and runs on **every GPU vendor and the CPU**
from a single source file. It is a teaching demo for the JuliaGPU stack: change
one line to move the whole program between NVIDIA, AMD, Intel, Apple, and the CPU.

The model is a one hidden layer MLP (multi layer perceptron) trained with full
batch gradient descent on the classic spiral dataset. It is small enough to run
in a couple of seconds on any laptop, with or without a GPU.

```
epoch     1   loss 1.2253   train acc 56.00%
epoch   200   loss 0.1292   train acc 98.00%
...
epoch  2000   loss 0.0329   train acc 99.33%

final train accuracy: 99.33%
```

## The one line that changes

```julia
backend = CPU()                         # works everywhere, no GPU package needed
# using CUDA;   backend = CUDABackend()   # NVIDIA
# using AMDGPU; backend = ROCBackend()    # AMD
# using oneAPI; backend = oneAPIBackend() # Intel
# using Metal;  backend = MetalBackend()  # Apple Silicon
```

Everything else in the program stays identical. That is the whole point of the
JuliaGPU stack.

## How it maps onto the JuliaGPU stack

The demo deliberately uses the two layers you would explain on a slide.

1. **KernelAbstractions.jl** is the portable kernel layer. The two kernels in
   `src/PortableNet.jl`, `relu_kernel!` and `sgd_kernel!`, are written once and
   compiled for CPU, CUDA, ROCm, oneAPI, and Metal. This is the layer that
   libraries such as AcceleratedKernels.jl are built on.

   ```julia
   @kernel function sgd_kernel!(W, @Const(dW), lr)
       i = @index(Global)
       @inbounds W[i] = W[i] - lr * dW[i]
   end
   ```

2. **The array layer (GPUArrays)** gives you generic matrix multiply (`*`),
   broadcasting (`.+`, `.*`, `exp.`), and reductions (`sum`, `maximum`) that
   already work on every backend array. The forward and backward passes are just
   ordinary Julia array code, and each backend swaps in its own fast routines
   underneath.

A single `backend` value selects both layers at once. `to_backend(backend, A)`
moves host data onto the device, kernels launch on that backend, and array
operations dispatch to the matching vendor library.

## Run it

Requires Julia 1.9 or newer.

```bash
git clone https://github.com/shreyas-omkar/PortableNet.jl
cd PortableNet.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/train.jl
```

That runs on the CPU out of the box. To run on a GPU, install the backend package
for your hardware and uncomment the matching line in `examples/train.jl`:

```bash
julia --project=. -e 'using Pkg; Pkg.add("CUDA")'    # or AMDGPU, oneAPI, Metal
```

## What is inside

| File | What it holds |
| --- | --- |
| `src/PortableNet.jl` | the model, the two portable kernels, data generator, and training loop |
| `examples/train.jl` | the runnable demo you present |

The public functions are `spiral_data`, `MLP`, `train!`, `predict`, `accuracy`,
and `to_backend`. The model is about 150 lines of commented code, meant to be
read start to finish during a talk.

## Why Julia for this

The same source compiles to native code for each vendor through
KernelAbstractions and the GPUArrays interface, so there is no separate CUDA C,
HIP, SYCL, or Metal shader to maintain. You write the math once and it runs
wherever a Julia GPU backend exists.

## License

MIT. See [LICENSE](LICENSE).
