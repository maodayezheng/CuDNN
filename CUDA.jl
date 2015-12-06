
# Simplified version of CUDA.jl from https://github.com/JuliaGPU/CUDA.jl
export CUDA
module CUDA
export CuPtr
using Compat

# search CUDA library
const libcuda = Libdl.find_library(["libcuda"], ["/usr/lib/", "/usr/local/cuda/lib"])


# CUDA errors 
const driver_error_descriptions = (Int=>ASCIIString)[
    0 => "Success",
    1 => "Invalid value",
    2 => "Out of memory",
    3 => "Driver not initialized",
    4 => "Driver being shutdown",
    5 => "Profiler disabled",
    6 => "Profiler not initialized",
    7 => "Profiler already started",
    8 => "Profiler already stopped",
    100 => "No CUDA-capable device",
    101 => "Invalid device ordinal",
    200 => "Invalid kernel image",
    201 => "Invalid context",
    202 => "Context already current",
    205 => "Map operation failed",
    206 => "Unmap operation failed",
    207 => "Array mapped",
    208 => "Resource already mapped",
    209 => "No kernel image available/suitable for GPU",
    210 => "Resource already acquired",
    211 => "Resource not mapped",
    212 => "Resource not mapped as array",
    213 => "Resource not mapped as pointer",
    214 => "Uncorrectable ECC error detected",
    215 => "Unsupported limit",
    216 => "Context already in use",
    217 => "Peer access not supported",
    218 => "PTX JIT compilation failed",
    300 => "Invalid kernel source",
    301 => "File not found",
    302 => "Shared object symbol not found",
    303 => "Shared object initialization failed",
    304 => "OS call failed",
    400 => "Invalid handle",
    500 => "Named symbol not found",
    600 => "Not ready",
    700 => "Kernel launch failed",
    701 => "Launch out of resources",
    702 => "Launch timeout",
    703 => "Incompatible texturing mode",
    704 => "Peer access already enabled",
    705 => "Peer access not enabled",
    708 => "Primary context already active",
    709 => "Context destroyed",
    710 => "Assertion triggered failure",
    711 => "Too many peers",
    712 => "Host memory already registered",
    713 => "Host memory not registered",
    714 => "Device stack error",
    715 => "Device illegal instruction",
    716 => "Device load or store instruction on unaligned memory address",
    717 => "Device instruction on address not belonging to an allowed space",
    718 => "Device program counter wrapped its address space",
    719 => "Device exception while executing a kernel",
    800 => "Operation not permitted",
    801 => "Operation not supported",
    999 => "Unknown error"
]


immutable CuDriverError
    code::Int
end

description(err::CuDriverError) = driver_error_descriptions[err.code]
# CUDA errors end



#CUDA function name MAPPING

const cuInit = :cuInit
const cuDeviceGetCount = :cuDeviceGetCount
const cuDriverGetVersion = :cuDriverGetVersion
const cuDeviceGet = :cuDeviceGet
const cuDeviceGetAttribute = :cuDeviceGetAttribute
const cuDeviceGetCount = :cuDeviceGetCount
const cuDeviceGetName = :cuDeviceGetName
const cuLaunchKernel = :cuLaunchKernel
const cuModuleGetFunction = :cuModuleGetFunction
const cuModuleLoad = :cuModuleLoad
const cuModuleUnload = :cuModuleUnload
const cuStreamSynchronize = :cuStreamSynchronize

# when api version >= 3020
const cuDeviceTotalMem = :cuDeviceTotalMem_v2
const cuCtxCreate = :cuCtxCreate_v2
const cuModuleGetGlobal = :cuModuleGetGlobal_v2
const cuMemGetInfo = :cuMemGetInfo_v2
const cuMemAlloc = :cuMemAlloc_v2
const cuMemAllocPitch = :cuMemAllocPitch_v2
const cuMemFree = :cuMemFree_v2
const cuMemGetAddressRange = :cuMemGetAddressRange_v2
const cuMemAllocHost = :cuMemAllocHost_v2
const cuMemHostGetDevicePointer = :cuMemHostGetDevicePointer_v2
const cuMemcpyHtoD = :cuMemcpyHtoD_v2
const cuMemcpyDtoH = :cuMemcpyDtoH_v2
const cuMemcpyDtoD = :cuMemcpyDtoD_v2
const cuMemcpyDtoA = :cuMemcpyDtoA_v2
const cuMemcpyAtoD = :cuMemcpyAtoD_v2
const cuMemcpyHtoA = :cuMemcpyHtoA_v2
const cuMemcpyAtoH = :cuMemcpyAtoH_v2
const cuMemcpyAtoA = :cuMemcpyAtoA_v2
const cuMemcpyHtoAAsync = :cuMemcpyHtoAAsync_v2
const cuMemcpyAtoHAsync = :cuMemcpyAtoHAsync_v2
const cuMemcpy2D = :cuMemcpy2D_v2
const cuMemcpy2DUnaligned = :cuMemcpy2DUnaligned_v2
const cuMemcpy3D = :cuMemcpy3D_v2
const cuMemcpyHtoDAsync = :cuMemcpyHtoDAsync_v2
const cuMemcpyDtoHAsync = :cuMemcpyDtoHAsync_v2
const cuMemcpyDtoDAsync = :cuMemcpyDtoDAsync_v2
const cuMemcpy2DAsync = :cuMemcpy2DAsync_v2
const cuMemcpy3DAsync = :cuMemcpy3DAsync_v2
const cuMemsetD8 = :cuMemsetD8_v2
const cuMemsetD16 = :cuMemsetD16_v2
const cuMemsetD32 = :cuMemsetD32_v2
const cuMemsetD2D8 = :cuMemsetD2D8_v2
const cuMemsetD2D16 = :cuMemsetD2D16_v2
const cuMemsetD2D32 = :cuMemsetD2D32_v2
const cuArrayCreate = :cuArrayCreate_v2
const cuArrayGetDescriptor = :cuArrayGetDescriptor_v2
const cuArray3DCreate = :cuArray3DCreate_v2
const cuArray3DGetDescriptor = :cuArray3DGetDescriptor_v2
const cuTexRefSetAddress = :cuTexRefSetAddress_v2
const cuTexRefGetAddress = :cuTexRefGetAddress_v2
const cuGraphicsResourceGetMappedPointer = :cuGraphicsResourceGetMappedPointer_v2

# when api version >= 4000
const cuCtxDestroy = :cuCtxDestroy_v2
const cuCtxPopCurrent = :cuCtxPopCurrent_v2
const cuCtxPushCurrent = :cuCtxPushCurrent_v2
const cuStreamDestroy = :cuStreamDestroy_v2
const cuEventDestroy = :cuEventDestroy_v2


# CUDA functions call warpper
macro cucall(fv, argtypes, args...)
    f = eval(fv)
    quote
        _curet = ccall( ($(Meta.quot(f)), libcuda), Cint, $argtypes, $(args...) )
        if _curet != 0
            throw(CuDriverError(int(_curet)))
        end
    end
end

function initialize()
    @cucall(cuInit, (Cint,), 0)
    println("CUDA Driver Initialized")
end

initialize()


# Get driver version

function driver_version()
    a = Cint[0]
    @cucall(cuDriverGetVersion, (Ptr{Cint},), a)
    return Int(a[1])
end

const DriverVersion = driver_version()

if DriverVersion < 4000
    error("CUDA of version 4.0 or above is required.")
end


# box a variable into array

cubox{T}(x::T) = T[x]

end # module CUDA