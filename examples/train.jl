# Train a tiny MLP on the spiral dataset, on whatever backend you pick.
#
# Usage:
#   julia --project=. examples/train.jl            # CPU (default)
#   julia --project=. examples/train.jl cpu        # CPU
#   julia --project=. examples/train.jl cuda       # NVIDIA  (needs: Pkg.add("CUDA"))
#   julia --project=. examples/train.jl amd        # AMD     (needs: Pkg.add("AMDGPU"))
#   julia --project=. examples/train.jl intel      # Intel   (needs: Pkg.add("oneAPI"))
#   julia --project=. examples/train.jl metal      # Apple   (needs: Pkg.add("Metal"))
#
# The model, the kernels and the training loop never change between devices.
# Only the backend below does.

using PortableNet
using KernelAbstractions
using Printf

dev = lowercase(get(ARGS, 1, "cpu"))

# Load the vendor package for the chosen device (top level, so `using` is legal).
if dev == "cuda"
    @eval using CUDA
elseif dev in ("amd", "roc")
    @eval using AMDGPU
elseif dev in ("intel", "oneapi")
    @eval using oneAPI
elseif dev in ("apple", "metal")
    @eval using Metal
end

backend =
    dev == "cpu"              ? CPU() :
    dev == "cuda"             ? CUDABackend() :
    dev in ("amd", "roc")     ? ROCBackend() :
    dev in ("intel", "oneapi") ? oneAPIBackend() :
    dev in ("apple", "metal") ? MetalBackend() :
    error("unknown backend \"$dev\". use one of: cpu, cuda, amd, intel, metal")

classes = 3
Xh, y = spiral_data(; classes = classes, per_class = 100)
Yohh   = PortableNet.onehot(y, classes)

# Move the data onto the chosen backend once.
X   = to_backend(backend, Xh)
Yoh = to_backend(backend, Yohh)

model = MLP(backend, 2, 100, classes)

println("Training on ", backend, " with ", eltype(Xh), " ...")
@time train!(model, X, Yoh, y; epochs = 2000, lr = 1.0f0)
@printf("\nfinal train accuracy: %.2f%%\n", 100 * accuracy(model, X, y))
