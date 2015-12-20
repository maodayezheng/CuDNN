
# Simplified version of CUDA.jl from https://github.com/JuliaGPU/CUDA.jl
export CUDA
module CUDA
export CuPtr
using Compat

# search CUDA library
const libcuda = Libdl.find_library(["libcuda"], ["/usr/lib/", "/usr/local/cuda/lib"])

# CUDA errors 
const driver_error_descriptions = @compat(Dict(
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
))


immutable CuDriverError
    code::Int
end

description(err::CuDriverError) = driver_error_descriptions[err.code]
# CUDA errors end

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

#CUDA Device 
function devcount()
    # Get the number of CUDA-capable CuDevices
    a = Cint[0]
    @cucall(:cuDeviceGetCount, (Ptr{Cint},), a)
    return int(a[1])
end


immutable CuDevice
    ordinal::Cint
    handle::Cint

    function CuDevice(i::Int)
        ordinal = convert(Cint, i)
        a = Cint[0]
        @cucall(:cuDeviceGet, (Ptr{Cint}, Cint), a, ordinal)
        handle = a[1]
        new(ordinal, handle)
    end
end



# CUDA Context
immutable CuContext
    handle::Ptr{Void}
end

const CTX_SCHED_AUTO  = 0x00
const CTX_SCHED_SPIN  = 0x01
const CTX_SCHED_YIELD = 0x02
const CTX_SCHED_BLOCKING_SYNC = 0x04
const CTX_MAP_HOST = 0x08
const CTX_LMEM_RESIZE_TO_MAX = 0x10

function create_context(dev::CuDevice, flags::Integer)
    a = Array(Ptr{Void}, 1)
    @cucall(:cuCtxCreate_v2, (Ptr{Ptr{Void}}, Cuint, Cint), a, flags, dev.handle)
    return CuContext(a[1])
end

create_context(dev::CuDevice) = create_context(dev, 0)

function destroy(ctx::CuContext)
    @cucall(:cuCtxDestroy_v2, (Ptr{Void},), ctx.handle)
end


immutable CuCapability
    major::Int
    minor::Int
end

function name(dev::CuDevice)
    const buflen = 256
    buf = Array(Cchar, buflen)
    @cucall(:cuDeviceGetName, (Ptr{Cchar}, Cint, Cint), buf, buflen, dev.handle)
    bytestring(pointer(buf))
end

function totalmem(dev::CuDevice)
    a = Csize_t[0]
    @cucall(:cuDeviceTotalMem, (Ptr{Csize_t}, Cint), a, dev.handle)
    return int(a[1])
end

function attribute(dev::CuDevice, attrcode::Integer)
    a = Cint[0]
    @cucall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, Cint), a, attrcode, dev.handle)
    return int(a[1])
end

capability(dev::CuDevice) = CuCapability(attribute(dev, 75), attribute(dev, 76))

function list_devices()
    cnt = devcount()
    if cnt == 0
        println("No CUDA-capable CuDevice found.")
        return
    end

    for i = 0:cnt-1
        dev = CuDevice(i)
        nam = name(dev)
        tmem = iround(totalmem(dev) / (1024^2))
        cap = capability(dev)

        println("device[$i]: $(nam), capability $(cap.major).$(cap.minor), total mem = $tmem MB")
    end
end


# CUDA Execution control

get_dim_x(g::Int) = g
get_dim_x(g::(Int, Int)) = g[1]
get_dim_x(g::(Int, Int, Int)) = g[1]

get_dim_y(g::Int) = 1
get_dim_y(g::(Int, Int)) = g[2]
get_dim_y(g::(Int, Int, Int)) = g[2]

get_dim_z(g::Int) = 1
get_dim_z(g::(Int, Int)) = 1
get_dim_z(g::(Int, Int, Int)) = g[3]

typealias CuDim Union(Int, (Int, Int), (Int, Int, Int))

# Kernel management

function launch(f::CuFunction, grid::CuDim, block::CuDim, args::Tuple; shmem_bytes::Int=4, stream::CuStream=null_stream())
    gx = get_dim_x(grid)
    gy = get_dim_y(grid)
    gz = get_dim_z(grid)

    tx = get_dim_x(block)
    ty = get_dim_y(block)
    tz = get_dim_z(block)

    kernel_args = [cubox(arg) for arg in args]

    @cucall(cuLaunchKernel, (
        Ptr{Void},  # function
        Cuint,  # grid dim x
        Cuint,  # grid dim y
        Cuint,  # grid dim z
        Cuint,  # block dim x
        Cuint,  # block dim y
        Cuint,  # block dim z
        Cuint,  # shared memory bytes,
        Ptr{Void}, # stream
        Ptr{Ptr{Void}}, # kernel parameters,
        Ptr{Ptr{Void}}), # extra parameters
        f.handle, gx, gy, gz, tx, ty, tz, shmem_bytes, stream.handle, kernel_args, 0)

end # module CUDA