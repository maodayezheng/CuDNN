# some description about NVIDIA CuDNN (CUDNN LIBRARY USer Guide v3.0)
# Features implemented in CuDNN: 
# 1. Convolution forward and backward, include cross correlation
# 2. Pooling forward and backward
# 3. Softmax forward and backward
# 4. Neuron activations forward and backward
#     i) Rectified linear
#     ii) Sigmoid
#     iii) Hyperbolic tangent
# 5. Tensor transformation function

export CuDNN
module CuDNN

using CUDA
using Compat
const libcuda = Libdl.find_library(["libcudnn"], ["/usr/lib/", "/usr/local/cuda/lib"])


immutable CuDNNError <: Exception
  code :: Int
end

macro cudnncall(fv, argtypes, args...)
  f = eval(fv)
  quote
    _curet = ccall( ($(Meta.quot(f)), $libcudnn), Cint, $argtypes, $(args...)  )
    if round(Int, _curet) != CUDNN_STATUS_SUCCESS
      throw(CuDNNError(round(Int, _curet)))
    end
  end
end


# Pointer
typealias cudnnHandle_t Ptr{Void} # hold cuDNN library context
typealias cudnnTensorDescriptor_t Ptr{Void} # hold the description of generic n-D dataset 
typealias cudnnFilterDescriptor_t Ptr{Void} # hold the description of a filter dataset 
typealias cudnnConvolutionDescriptor_t Ptr{Void} # hold the description of a convolution operation
typealias cudnnPoolingDescriptor_t Ptr{Void}


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


const cudnnStatus_t = @compat(Dict(
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

# cudnnDataType_t
const CUDNN_DATA_FLOAT = 0
const CUDDN_DATA_DOUBLE = 1
const CUDNN_DATA_HALF = 2

#cudnnTensorFormat_t

const CUDNN_TENSOR_NCHW = 0
const CUDNN_TENSOR_NHWC = 1

#cudnnAddMode_t

const CUDNN_ADD_IMAGE = 0
const CUDNN_ADD_SAME_HW = 0
const CUDNN_ADD_FEATURE_MAP = 1
const CUDNN_ADD_SAM_CHW = 1
const CUDNN_ADD_SAME_C = 2
const CUDNN_ADD_FULL_TENSOR = 3

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

#cudnnDataType_t
CUDNN_DATA_FLOAT = 0  # 32 bits 
CUDNN_DATA_DOUBLE = 1 # 64 bits
CUDNN_DATA_HALF = 2   # 16 bits



end