# Train a tiny MLP on the spiral dataset, on whatever backend you pick.
#
# The ONLY line you change to move between vendors is `backend = ...`.
# CPU is the default and needs no GPU package installed.

using PortableNet
using KernelAbstractions
using Printf

# ---- pick your device (uncomment one) --------------------------------------
backend = CPU()                                  # works everywhere, no GPU needed
# using CUDA;   backend = CUDABackend()          # NVIDIA
# using AMDGPU; backend = ROCBackend()           # AMD
# using oneAPI; backend = oneAPIBackend()        # Intel
# using Metal;  backend = MetalBackend()         # Apple Silicon
# ----------------------------------------------------------------------------

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
