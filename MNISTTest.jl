include("CUDA/CuDNN.jl")
using CuDNN
using CUDA
CUDA.init()
CUDA.list_devices()
CuDNN.cudnnGetVersion()
device = CuDevice(0)