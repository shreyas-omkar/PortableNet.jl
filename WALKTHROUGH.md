# Walkthrough: setup and run

This guide takes you from a clean machine to a trained model, on CPU and on GPU,
for both Windows and Linux. The CPU path needs no GPU and works on any laptop.

## 1. Install Julia

The recommended way is `juliaup`, the official Julia version manager. It installs
Julia and keeps it updated.

### Linux

```bash
curl -fsSL https://install.julialang.org | sh
```

Close and reopen the terminal, then check it:

```bash
julia --version
```

If `julia` is not found, add the install directory to your shell: it is usually
`~/.juliaup/bin`. The installer prints the exact line to add.

### Windows

Open PowerShell and run:

```powershell
winget install julia -s msstore
```

If `winget` is not available, download the installer from
https://julialang.org/downloads and run it, or install juliaup directly with:

```powershell
winget install --name Julia --id Julia.Juliaup -e
```

Close and reopen PowerShell, then check it:

```powershell
julia --version
```

## 2. Get the package

You need `git`. On Windows, install it from https://git-scm.com/download/win if
you do not have it. On Linux use your package manager (for example
`sudo apt install git`).

```bash
git clone https://github.com/shreyas-omkar/PortableNet.jl
cd PortableNet.jl
```

## 3. Install the dependencies

This downloads KernelAbstractions and precompiles it. Run it once.

Linux (bash):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Windows (PowerShell uses double quotes):

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## 4. Run on CPU

This works on every machine, with no GPU package installed.

```bash
julia --project=. examples/train.jl cpu
```

Expected output (the first run is slower because it compiles; a second run is
fast):

```
Training on CPU(false) with Float32 ...
epoch     1   loss 1.2253   train acc 56.00%
epoch   200   loss 0.1292   train acc 98.00%
...
epoch  2000   loss 0.0329   train acc 99.33%

final train accuracy: 99.33%
```

## 5. Run on GPU

Pick the package that matches your hardware and install it once. Then pass the
device name to the same script. The code does not change.

| Hardware | Install once | Run command |
| --- | --- | --- |
| NVIDIA | `Pkg.add("CUDA")`   | `julia --project=. examples/train.jl cuda` |
| AMD    | `Pkg.add("AMDGPU")` | `julia --project=. examples/train.jl amd` |
| Intel  | `Pkg.add("oneAPI")` | `julia --project=. examples/train.jl intel` |
| Apple  | `Pkg.add("Metal")`  | `julia --project=. examples/train.jl metal` |

For example, on an NVIDIA machine:

Linux (bash):

```bash
julia --project=. -e 'using Pkg; Pkg.add("CUDA")'
julia --project=. examples/train.jl cuda
```

Windows (PowerShell):

```powershell
julia --project=. -e "using Pkg; Pkg.add(\"CUDA\")"
julia --project=. examples/train.jl cuda
```

The same program then reports the device it is running on, for example
`Training on CUDABackend(...) device=NVIDIA GeForce RTX 5080`, and reaches the
same accuracy as the CPU run.

## 6. Demo on cyclops

A ready copy lives at `~/PortableNet.jl` on cyclops, cloned from this repo with
CUDA already installed, so the GPU run starts with no setup.

```bash
ssh shreyas-omkar@cyclops.juliacomputing.io
cd ~/PortableNet.jl
git pull                                                      # get the latest code
/opt/julia-1.12/bin/julia --project=. examples/train.jl cuda  # same code on the RTX 5080
/opt/julia-1.12/bin/julia --project=. examples/train.jl cpu   # same code on the CPU
```

Running the two commands back to back is the clearest way to show one source
file on two very different devices.

## Troubleshooting

- **`julia` not found after install.** Reopen the terminal. If it still fails,
  the `juliaup` bin directory is not on your `PATH`; the installer prints the
  directory to add.
- **GPU run errors with a driver message.** Confirm the GPU and its driver are
  visible to the system first. For NVIDIA, `nvidia-smi` should list the card.
- **First run feels slow.** That is Julia compiling the code. The second run of
  the same command is fast.
- **Out of memory on a small GPU.** This demo is tiny and should fit anywhere,
  but if needed lower `per_class` in `examples/train.jl`.
