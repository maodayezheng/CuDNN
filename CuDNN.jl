# This module is based on CuDNN v3.0
# Detail information about each types and functions can be found in CuDNN LIBRARY USer Guide v3.0
# NVIDIA CuDNN v3.0 brief summary
# Features implemented in CuDNN: 
# 1. Convolution forward and backward, include cross correlation
# 2. Pooling forward and backward
# 3. Softmax forward and backward
# 4. Neuron activations forward and backward
#     i) Rectified linear
#     ii) Sigmoid
#     iii) Hyperbolic tangent
# 5. Tensor transformation function

# This is only a lower level wrapper of CuDNN
# The implementation of this module refers to following developments:
# 1. CUDA.jl (https://github.com/JuliaGPU/CUDA.jl)
# 2. cudnn.h v3.0 (NVIDIA)
# 3. mnistCUDNN.cpp (NVIDIA 2014)

export CuDNN
module CuDNN

using CUDA
using Compat
const libcudnn = Libdl.find_library(["libcudnn"], ["/usr/lib/", "/usr/local/cuda/lib"])


#Enumerated types reference to cudnnStatus_t

const  CUDNN_STATUS_SUCCESS          = 0
const  CUDNN_STATUS_NOT_INITIALIZED  = 1
const  CUDNN_STATUS_ALLOC_FAILED     = 2
const  CUDNN_STATUS_BAD_PARAM        = 3
const  CUDNN_STATUS_ARCH_MISMATCH    = 4
const  CUDNN_STATUS_MAPPING_ERROR    = 5
const  CUDNN_STATUS_EXECUTION_FAILED = 6
const  CUDNN_STATUS_INTERNAL_ERROR   = 7
const  CUDNN_STATUS_NOT_SUPPORTED    = 8
const  CUDNN_STATUS_LICENSE_ERROR    = 9


#CuDNN errors
immutable CuDNNError <: Exception
  code :: Int
end

const cudnnStatus_error = @compat(Dict(
	CUDNN_STATUS_SUCCESS => "The operation complete successfully",
	CUDNN_STATUS_NOT_INITIALIZED =>"The CuDNN library was not initialized",
	CUDNN_STATUS_ALLOC_FAILED => "Resurce allocation failed, to correct: deallocate previous allocated memory as much as possible",
	CUDNN_STATUS_BAD_PARAM =>"Incorrect value passed, to correct: ensure all the parameters being passed have valid values",
	CUDNN_STATUS_ARCH_MISMATCH =>"Feature absent from the GPU device, to correct: to compile and run the application on device with compute capabilities greater than 3.0",
	CUDNN_STATUS_MAPPING_ERROR => "An access to GPU memory space failed, to correct: unbind any previous bound textures",
	CUDNN_STATUS_EXECUTION_FAILED =>"GPU program fail to execute, to correct:ccheck the hardware, an appropriate version of the friver, and the cuDNN library are coreectly installed",
	CUDNN_STATUS_INTERNAL_ERROR =>"An internal cuDNN operation failed",
	CUDNN_STATUS_NOT_SUPPORTED =>"The functionality requested is not presently supported by cuDNN",
	CUDNN_STATUS_LICENSE_ERROR =>"The functionality requested requires license",
	))

import Base.show
show(io::IO, error::CuDNNError) = print(io, cudnnStatus_error[error.code])


macro cudnncheck(fv, argtypes, args...)
  f = eval(fv)
  quote
    _curet = ccall( ($(Meta.quot(f)), $libcudnn), Cint, $argtypes, $(args...)  )
    if round(Int, _curet) != CUDNN_STATUS_SUCCESS
      throw(CuDNNError(round(Int, _curet)))
    end
  end
end

#Check Version
# Error cudnnGetVersion can not find
#function cudnnGetVersion()
#version = Cint[0]
#@cudnncheck(:cudnnGetVerion,(Ptr{Void},),version)
#return version[1]
#end

#context pointer
typealias cudaStream_t Ptr{Void} # hold Cuda Stream
typealias cudnnHandle_t Ptr{Void} # hold cuDNN library context

function cudnnCreate()
handle = cudnnHandle_t[0]
@cudnncheck(:cudnnCreate, (Ptr{cudnnHandle_t},), handle)
return handle[1]
end

function cudnnDestroy(handle ::cudnnHandle_t)
@cudnncheck(:cudnnDestroy, (cudnnHandle_t,), handle)
end

function cudnnSetStream(handle,streamId)
@cudnncheck(:cudnnSetStream,(cudnnHandle_t,cudaStream_t),handle,streamId)
end

function cudnnGetStream(handle)
streamId = cudaStream_t[0]
@cudnncheck(:cudnnGetStream,(cudnnHandle_t,Ptr{cudaStream_t}),handle,streamId)
return streamId[1]
end

# Data type pointer

typealias cudnnConvolutionDescriptor_t Ptr{Void} # hold the description of a convolution operation
typealias cudnnPoolingDescriptor_t Ptr{Void}


#cudnnDataType_t
CUDNN_DATA_FLOAT = 0  # 32 bits 
CUDNN_DATA_DOUBLE = 1 # 64 bits
CUDNN_DATA_HALF = 2   # 16 bits

function cudnnDataTypeCheck{T<:AbstractFloat}(datatype::Type{T})

if datatype == Float32
	return CUDNN_DATA_FLOAT
elseif datatype == Float64
	return CUDNN_DATA_DOUBLE
elseif datatype == Float16
	return CUDNN_DATA_HALF
else
    error("CUDNN does not support data type $(datatype)")
end
end


function cudnnDataTypeConvert(dateType::Cint)
if dataType == CUDNN_DATA_FLOAT
	return Float32
elseif dataType == CUDNN_DATA_DOUBLE
	return Float64
elseif dataType == CUDNN_DATA_HALF
	return Float16
else
	error("CuDNN error data type:$(datatype)")
end
end

######### CuDNN Tensor Description ###########

# Tensor discriptor 
typealias cudnnTensorDescriptor_t Ptr{Void} # hold the description of generic n-D dataset 

#cudnnTensorFormat_t
const CUDNN_TENSOR_NCHW = 0 #data laid out order: image, features map, rows, columns
const CUDNN_TENSOR_NHWC = 1 #data laid out order: image, rows, columns, features map

function cudnnCreateTensorDescriptor()
tensorDesc = cudnnTensorDescriptor_t[0]
@cudnncheck(:cudnnCreateTensorDescriptor,(Ptr{cudnnTensorDescriptor_t},),tensorDesc)
return tensorDesc[1]
end

function cudnnSetTensor4dDescriptor{T<:AbstractFloat}(tensorDesc::cudnnTensorDescriptor_t,dataType::Type{T},n,c,h,w)
dtype = cudnnDataTypeCheck(dataType)
@cudnncheck(:cudnnSetTensor4dDescriptor,(cudnnTensorDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint),tensorDesc,CUDNN_TENSOR_NCHW,dtype,n,c,h,w)
end


function cudnnSetTensor4dDescriptorEx{T<:AbstractFloat}(tensorDesc::cudnnTensorDescriptor_t,dataType::Type{T},n,c,h,w,nStride,cStride,hStride,wStride)
dtype = cudnnDataTypeCheck(dataType)
@cudnncheck(:cudnnSetTensor4dDescriptorEx,(cudnnTensorDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),n,c,h,w,nStride,cStride,hStride,wStride)
end

function cudnnGetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t)
dataType = Cint[0]
n = Cint[0]
c = Cint[0]
h = Cint[0]
w = Cint[0]
nStride = Cint[0]
cStride = Cint[0]
hStride = Cint[0]
wStride = Cint[0]
@cudnncheck(:cudnnGetTensor4dDescriptor,(cudnnTensorDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
dtype = cudnnDataTypeConvert(dataType[1])
return (tensorDesc,dtype,(n[1],c[1],h[1],w[1]),(nStride[1],cStride[1],hStride[1],wStride[1]))
end


function cudnnSetTensorNdDescriptor{T<:AbstractFloat}(tensorDesc::cudnnTensorDescriptor_t,dataType::Type{T},nbDims::Unsigned,dimA::Array{Unsigned,1},strideA::Array{Unsigned,1})
dtype = cudnnDataTypeCheck(dataType)
@cudnncheck(:cudnnSetTensorNdDescriptor,(cudnnTensorDescriptor_t,Cint,Cint,Ptr{Cint},Ptr{Cint}),tensorDesc,dtype,nbDims,dimA,strideA)
end

function cudnnGetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t,nbDimsRequested::Unsigned)
dataType = Cint[0]
nbDims = Cint[0]
dimA = Cint[0]
strideA = Cint[0]
@cudnncheck(:cudnnGetTensorNdDescriptor,(cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,nbDimsRequested,datatype,nbDims,dimA,strideA)
dtype = cudnnDataTypeConvert(dataType[1])
return (tensorDesc,dtype,nbDims,dimA,strideA)
end

function cudnnDestroyTensorDescriptor(tensorDesc::cudnnTensorDescriptor_t)
@cudnncheck(:cudnnDestroyTensorDescriptor,(cudnnTensorDescriptor_t,),tensorDesc)
end

#TODO: what happens in Toensor transformation ?
function cudnnTransformTensor(handle::cudnnHandle_t,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CuPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CuPtr)
@cudnncheck(:cudnnTransformTensor,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData.p,beta,destDesc,destData.p)
end

#cudnnAddMode_t
const CUDNN_ADD_IMAGE = 0
const CUDNN_ADD_SAME_HW = 0
const CUDNN_ADD_FEATURE_MAP = 1
const CUDNN_ADD_SAM_CHW = 1
const CUDNN_ADD_SAME_C = 2
const CUDNN_ADD_FULL_TENSOR = 3

#TODO: if the version is less than v3.0 replace cudnnAddTensor_v3 by cudnnAddTensor
function cudnnAddTensor(handle::cudnnHandle_t,alpha,biasDesc::cudnnTensorDescriptor_t,biasData::CuPtr,beta,srcDestDesc::cudnnTensorDescriptor_t,srcDestData::CuPtr)
@cudnncheck(:cudnnAddensor_v3,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,biasDesc,biasData.p,beta,srcDestDesc,srcDestData.p)
end

function cudnnSetTensor(handle::cudnnHandle_t,srcDestDesc::cudnnTensorDescriptor_t,srcDestData::CuPtr,value)
@cudnncheck(:cudnnSetTensor,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,srcDestDesc,srcDestData.p,value)
end

function cudnnScaleTensor(handle::cudnnHandle_t,srcDestDesc::cudnnTensorDescriptor_t,srcDestData::CuPtr,alpha)
@cudnncheck(:cudnnScaleTensor,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,srcDestDesc,,srcDestData.p,alpha)
end

#########cudnn Tensor Description End ##########



######## cudnn Filter Description ###########
# filter pointer
typealias cudnnFilterDescriptor_t Ptr{Void} # hold the description of a filter dataset 

function cudnnCreateFilterDescriptor()
filterDesc = cudnnFilterDescriptor_t[0]
@cudnncheck(:cudnnCreateFilterDescriptor,(Ptr{cudnnFilterDescriptor_t},)filterDesc)
return filterDesc[1]
end

function cudnnSetFilter4dDescriptor{T<:AbstractFloat}(filterDesc::cudnnFilterDescriptor_t,dataType::Type{T},k,c,h,w)

end

######## cudnn Filter Description End ########

#cudnnConvolutionMode_t
const CUDNN_CONVOLUTION = 0
const CUDNN_CROSS_CORRELATION = 1

#cudnnConvolutionFwdPreference_t
const CUDNN_CONVOLUTION_FWD_NO_WORKSAPCE = 0
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 3

#cudnnConvolutionFwdAlgo_t
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_RECOMP_GEMM = 1
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
const CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4

#cudnnConvolutionFwdAlgoPerf_t
# structure type for forward performance measurements
#TODO: not sure whether need this

#cudnnConvolutionBwdFilterPreference_t
const CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
const CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FRASTEST = 1
const CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2

#cudnnConvolutionBwdFilterAlgo_t
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 4

#cudnnConvolutionBwdFilterAlgoPerf_t
# structure type for backward measurements
#TODO: not sure whether need this 


#cudnnConvolutionBwdDataPreference_t
CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2

#cudnnConvolutionBwdDataAlgo_t
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 4


#cudnnConvolutionBwdDataPerf_t
# a structure type for performance results
#TODO: not sure whether need this

#cudnnSoftmaxAlgorithm_t
const CUDNN_SOFTMAX_FAST = 0
const CUDNN_SOFTMAX_ACCURATE = 1
const CUDNN_SOFTMAX_LOG = 2

#cudnnSoftmaxMode_t
const CUDNN_SOFTMAX_MODE_INSTANCE = 0
const CUDNN_SOFTMAX_MODE_CHANNEL = 1

#cudnnPoolingMode_t
const CUDNN_POOLING_MAX = 0
const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2

#cudnnActivationMode_t
const CUDNN_ACTIVATION_SIGMOID = 0
const CUDNN_ACTIVATION_RELU = 1
const CUDNN_ACTIVATION_TANH = 2

#cudnnLRNMode_t
const CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0

#cudnnDivNormMode_t
CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0


end