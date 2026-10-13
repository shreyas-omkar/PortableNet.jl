"""
    PortableNet

A tiny neural network that trains and runs on any JuliaGPU backend (CPU, CUDA,
ROCm, oneAPI, Metal) from a single source. It is a teaching demo for the
JuliaGPU stack, built on two layers:

  1. KernelAbstractions.jl  -> the `relu!` and `sgd!` kernels below. One source,
     compiled for every backend.
  2. The GPUArrays array layer -> generic matmul (`*`), broadcasting and
     reductions (`sum`, `maximum`) that already work on every backend array.

The model is a one hidden layer MLP trained with full batch gradient descent on
the classic two dimensional spiral dataset.
"""
module PortableNet

using KernelAbstractions
using LinearAlgebra
using Printf
using Random

export spiral_data, MLP, train!, accuracy, predict, to_backend

# ---------------------------------------------------------------------------
# Portable kernels (KernelAbstractions layer).
# The same two functions below compile to CPU, CUDA, ROCm, oneAPI and Metal.
# ---------------------------------------------------------------------------

@kernel function relu_kernel!(A)
    i = @index(Global)
    @inbounds A[i] = max(A[i], zero(eltype(A)))
end

@kernel function sgd_kernel!(W, @Const(dW), lr)
    i = @index(Global)
    @inbounds W[i] = W[i] - lr * dW[i]
end

"Apply ReLU in place, on whatever backend `A` lives on."
function relu!(backend, A)
    relu_kernel!(backend)(A; ndrange = length(A))
    return A
end

"In place SGD step: `W .-= lr .* dW`, as a portable kernel."
function sgd!(backend, W, dW, lr)
    sgd_kernel!(backend)(W, dW, eltype(W)(lr); ndrange = length(W))
    return W
end

# ---------------------------------------------------------------------------
# Data and small helpers
# ---------------------------------------------------------------------------

"""
    spiral_data(; classes=3, per_class=100, noise=0.2, T=Float32, rng)

Generate the classic spiral dataset. Returns `(X, y)` where `X` is a
`(2, classes*per_class)` matrix (one column per sample) and `y` is the integer
label vector with values in `1:classes`.
"""
function spiral_data(; classes = 3, per_class = 100, noise = 0.2,
                     T = Float32, rng = Random.default_rng())
    N, K = per_class, classes
    X = zeros(T, 2, N * K)
    y = zeros(Int, N * K)
    for j in 0:(K - 1)
        r = range(T(0), T(1); length = N)
        t = range(T(j * 4), T((j + 1) * 4); length = N) .+ T(noise) .* randn(rng, T, N)
        ix = (j * N + 1):(j * N + N)
        @. X[1, ix] = r * sin(t)
        @. X[2, ix] = r * cos(t)
        y[ix] .= j + 1
    end
    return X, y
end

"Build a `(classes, nsamples)` one hot matrix from label vector `y`."
function onehot(y, classes; T = Float32)
    Y = zeros(T, classes, length(y))
    for (i, c) in enumerate(y)
        Y[c, i] = one(T)
    end
    return Y
end

"Copy a host array onto `backend` and return the native backend array."
function to_backend(backend, A::AbstractArray)
    B = KernelAbstractions.allocate(backend, eltype(A), size(A))
    copyto!(B, A)
    return B
end

# softmax over the class dimension (dim 1). Generic: runs on any backend array.
function softmax(z)
    m = maximum(z; dims = 1)
    e = exp.(z .- m)
    return e ./ sum(e; dims = 1)
end

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

struct MLP{M, B}
    W1::M
    b1::M
    W2::M
    b2::M
    backend::B
end

"""
    MLP(backend, nin, nhidden, nout; T=Float32, rng)

Create a one hidden layer MLP with parameters allocated on `backend`. Weights
use He initialisation; biases start at zero.
"""
function MLP(backend, nin, nhidden, nout; T = Float32, rng = Random.default_rng())
    W1 = randn(rng, T, nhidden, nin) .* sqrt(T(2) / nin)
    W2 = randn(rng, T, nout, nhidden) .* sqrt(T(2) / nhidden)
    b1 = zeros(T, nhidden, 1)
    b2 = zeros(T, nout, 1)
    return MLP(to_backend(backend, W1), to_backend(backend, b1),
               to_backend(backend, W2), to_backend(backend, b2), backend)
end

# Forward pass. Returns the class probabilities and the hidden activation `a1`
# (kept so the backward pass can reuse it).
function forward(m::MLP, X)
    z1 = m.W1 * X .+ m.b1          # (H, N)  matmul + broadcast
    a1 = relu!(m.backend, z1)      # KernelAbstractions kernel, in place
    z2 = m.W2 * a1 .+ m.b2         # (C, N)
    probs = softmax(z2)            # (C, N)  reduction + broadcast
    return probs, a1
end

"Return class probabilities for inputs `X` (array on the model's backend)."
predict(m::MLP, X) = first(forward(m, X))

# One full batch gradient descent step. `Yoh` is the one hot target on the
# backend. Gradients are plain matmul / broadcast; the parameter updates use the
# portable `sgd!` kernel.
function step!(m::MLP, X, Yoh, lr)
    N = size(X, 2)
    probs, a1 = forward(m, X)

    dz2 = (probs .- Yoh) ./ eltype(probs)(N)   # (C, N)
    dW2 = dz2 * a1'                            # (C, H)
    db2 = sum(dz2; dims = 2)                   # (C, 1)

    da1 = m.W2' * dz2                          # (H, N)
    dz1 = da1 .* (a1 .> 0)                     # ReLU gradient
    dW1 = dz1 * X'                             # (H, nin)
    db1 = sum(dz1; dims = 2)                   # (H, 1)

    sgd!(m.backend, m.W1, dW1, lr)
    sgd!(m.backend, m.b1, db1, lr)
    sgd!(m.backend, m.W2, dW2, lr)
    sgd!(m.backend, m.b2, db2, lr)
    KernelAbstractions.synchronize(m.backend)

    # Cross entropy loss, reduced on host (cheap, once per reported epoch).
    return -sum(log.(max.(sum(probs .* Yoh; dims = 1), eltype(probs)(1e-7)))) / N
end

"""
    train!(m, X, Yoh, y; epochs=2000, lr=1.0f0, verbose=true, report_every=200)

Train the model in place. `X` and `Yoh` live on the model's backend; `y` is the
host label vector used only for the accuracy readout.
"""
function train!(m::MLP, X, Yoh, y; epochs = 2000, lr = 1.0f0,
                verbose = true, report_every = 200)
    for ep in 1:epochs
        loss = step!(m, X, Yoh, lr)
        if verbose && (ep == 1 || ep % report_every == 0 || ep == epochs)
            @printf("epoch %5d   loss %.4f   train acc %.2f%%\n",
                    ep, loss, 100 * accuracy(m, X, y))
        end
    end
    return m
end

"Fraction of samples in `X` classified correctly against host labels `y`."
function accuracy(m::MLP, X, y)
    probs = Array(predict(m, X))            # bring the small result to host
    preds = [argmax(view(probs, :, i)) for i in 1:size(probs, 2)]
    return count(preds .== y) / length(y)
end

end # module
