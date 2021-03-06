/*
* Copyright (c) 2015, Max Danielsson <work@autious.net> and Thomas Sievert
* All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "opencl_error.h"

const char* opencl_error_codename( cl_int err )
{
    switch( err )
    {
        case CL_SUCCESS: 
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: 
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: 
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: 
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: 
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: 
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: 
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: 
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: 
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: 
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: 
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: 
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: 
            return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: 
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: 
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE: 
            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE: 
            return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE: 
            return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED: 
            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: 
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE: 
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: 
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: 
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: 
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: 
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: 
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: 
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: 
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: 
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: 
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: 
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: 
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: 
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: 
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: 
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: 
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: 
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: 
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: 
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: 
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: 
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: 
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: 
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: 
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: 
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: 
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: 
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: 
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: 
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: 
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: 
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: 
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: 
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: 
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY: 
            return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR: 
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS: 
            return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS: 
            return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT: 
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        default: 
            return "UNKNOWN";
    }
}

const char* opencl_device_info_codename( cl_device_info devinfo )
{
    switch( devinfo )
    {
        case CL_DEVICE_TYPE:
            return "CL_DEVICE_TYPE";
        case CL_DEVICE_VENDOR_ID:
            return "CL_DEVICE_VENDOR_ID";
        case CL_DEVICE_MAX_COMPUTE_UNITS:
            return "CL_DEVICE_MAX_COMPUTE_UNITS";
        case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
            return "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS";
        case CL_DEVICE_MAX_WORK_GROUP_SIZE:
            return "CL_DEVICE_MAX_WORK_GROUP_SIZE";
        case CL_DEVICE_MAX_WORK_ITEM_SIZES:
            return "CL_DEVICE_MAX_WORK_ITEM_SIZES";
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
            return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR";
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
            return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT";
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
            return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT";
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
            return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG";
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
            return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT";
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
            return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE";
        case CL_DEVICE_MAX_CLOCK_FREQUENCY:
            return "CL_DEVICE_MAX_CLOCK_FREQUENCY";
        case CL_DEVICE_ADDRESS_BITS:
            return "CL_DEVICE_ADDRESS_BITS";
        case CL_DEVICE_MAX_READ_IMAGE_ARGS:
            return "CL_DEVICE_MAX_READ_IMAGE_ARGS";
        case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
            return "CL_DEVICE_MAX_WRITE_IMAGE_ARGS";
        case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
            return "CL_DEVICE_MAX_MEM_ALLOC_SIZE";
        case CL_DEVICE_IMAGE2D_MAX_WIDTH:
            return "CL_DEVICE_IMAGE2D_MAX_WIDTH";
        case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
            return "CL_DEVICE_IMAGE2D_MAX_HEIGHT";
        case CL_DEVICE_IMAGE3D_MAX_WIDTH:
            return "CL_DEVICE_IMAGE3D_MAX_WIDTH";
        case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
            return "CL_DEVICE_IMAGE3D_MAX_HEIGHT";
        case CL_DEVICE_IMAGE3D_MAX_DEPTH:
            return "CL_DEVICE_IMAGE3D_MAX_DEPTH";
        case CL_DEVICE_IMAGE_SUPPORT:
            return "CL_DEVICE_IMAGE_SUPPORT";
        case CL_DEVICE_MAX_PARAMETER_SIZE:
            return "CL_DEVICE_MAX_PARAMETER_SIZE";
        case CL_DEVICE_MAX_SAMPLERS:
            return "CL_DEVICE_MAX_SAMPLERS";
        case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
            return "CL_DEVICE_MEM_BASE_ADDR_ALIGN";
        case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
            return "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE";
        case CL_DEVICE_SINGLE_FP_CONFIG:
            return "CL_DEVICE_SINGLE_FP_CONFIG";
        case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
            return "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE";
        case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
            return "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE";
        case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
            return "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE";
        case CL_DEVICE_GLOBAL_MEM_SIZE:
            return "CL_DEVICE_GLOBAL_MEM_SIZE";
        case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
            return "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE";
        case CL_DEVICE_MAX_CONSTANT_ARGS:
            return "CL_DEVICE_MAX_CONSTANT_ARGS";
        case CL_DEVICE_LOCAL_MEM_TYPE:
            return "CL_DEVICE_LOCAL_MEM_TYPE";
        case CL_DEVICE_LOCAL_MEM_SIZE:
            return "CL_DEVICE_LOCAL_MEM_SIZE";
        case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
            return "CL_DEVICE_ERROR_CORRECTION_SUPPORT";
        case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
            return "CL_DEVICE_PROFILING_TIMER_RESOLUTION";
        case CL_DEVICE_ENDIAN_LITTLE:
            return "CL_DEVICE_ENDIAN_LITTLE";
        case CL_DEVICE_AVAILABLE:
            return "CL_DEVICE_AVAILABLE";
        case CL_DEVICE_COMPILER_AVAILABLE:
            return "CL_DEVICE_COMPILER_AVAILABLE";
        case CL_DEVICE_EXECUTION_CAPABILITIES:
            return "CL_DEVICE_EXECUTION_CAPABILITIES";
        case CL_DEVICE_QUEUE_PROPERTIES:
            return "CL_DEVICE_QUEUE_PROPERTIES";
        case CL_DEVICE_NAME:
            return "CL_DEVICE_NAME";
        case CL_DEVICE_VENDOR:
            return "CL_DEVICE_VENDOR";
        case CL_DRIVER_VERSION:
            return "CL_DRIVER_VERSION";
        case CL_DEVICE_PROFILE:
            return "CL_DEVICE_PROFILE";
        case CL_DEVICE_VERSION:
            return "CL_DEVICE_VERSION";
        case CL_DEVICE_EXTENSIONS:
            return "CL_DEVICE_EXTENSIONS";
        case CL_DEVICE_PLATFORM:
            return "CL_DEVICE_PLATFORM";
        case CL_DEVICE_DOUBLE_FP_CONFIG:
            return "CL_DEVICE_DOUBLE_FP_CONFIG";
        case CL_DEVICE_HALF_FP_CONFIG:
            return "CL_DEVICE_HALF_FP_CONFIG";
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
            return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF";
        case CL_DEVICE_HOST_UNIFIED_MEMORY:
            return "CL_DEVICE_HOST_UNIFIED_MEMORY";
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
            return "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR";
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
            return "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT";
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
            return "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT";
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
            return "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG";
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
            return "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT";
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
            return "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE";
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
            return "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF";
        case CL_DEVICE_OPENCL_C_VERSION:
            return "CL_DEVICE_OPENCL_C_VERSION";
        case CL_DEVICE_LINKER_AVAILABLE:
            return "CL_DEVICE_LINKER_AVAILABLE";
        case CL_DEVICE_BUILT_IN_KERNELS:
            return "CL_DEVICE_BUILT_IN_KERNELS";
        case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
            return "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE";
        case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
            return "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE";
        case CL_DEVICE_PARENT_DEVICE:
            return "CL_DEVICE_PARENT_DEVICE";
        case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:
            return "CL_DEVICE_PARTITION_MAX_SUB_DEVICES";
        case CL_DEVICE_PARTITION_PROPERTIES:
            return "CL_DEVICE_PARTITION_PROPERTIES";
        case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
            return "CL_DEVICE_PARTITION_AFFINITY_DOMAIN";
        case CL_DEVICE_PARTITION_TYPE:
            return "CL_DEVICE_PARTITION_TYPE";
        case CL_DEVICE_REFERENCE_COUNT:
            return "CL_DEVICE_REFERENCE_COUNT";
        case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:
            return "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC";
        case CL_DEVICE_PRINTF_BUFFER_SIZE:
            return "CL_DEVICE_PRINTF_BUFFER_SIZE";
/*
 * These seem to be new, don't exist on opencl.h version on debian.
        case CL_DEVICE_IMAGE_PITCH_ALIGNMENT:
            return "CL_DEVICE_IMAGE_PITCH_ALIGNMENT";
        case CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT:
            return "CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT";
*/
        default:
            return "UNKNOWN";
    }
}

const char* opencl_platform_info_codename( cl_platform_info platinfo )
{
    switch( platinfo )
    {
        case CL_PLATFORM_PROFILE:
            return "CL_PLATFORM_PROFILE";
        case CL_PLATFORM_VERSION:
            return "CL_PLATFORM_VERSION";
        case CL_PLATFORM_NAME:
            return "CL_PLATFORM_NAME";
        case CL_PLATFORM_VENDOR:
            return "CL_PLATFORM_VENDOR";
        case CL_PLATFORM_EXTENSIONS:
            return "CL_PLATFORM_EXTENSIONS";
        default:
            return "UNKNOWN";
    }
}

const char* opencl_device_type_codename( cl_device_type devtype )
{
    switch( devtype )
    {
        case CL_DEVICE_TYPE_DEFAULT:
            return "CL_DEVICE_TYPE_DEFAULT";
        case CL_DEVICE_TYPE_CPU:
            return "CL_DEVICE_TYPE_CPU";
        case CL_DEVICE_TYPE_GPU:
            return "CL_DEVICE_TYPE_GPU";
        case CL_DEVICE_TYPE_ACCELERATOR:
            return "CL_DEVICE_TYPE_ACCELERATOR";
        case CL_DEVICE_TYPE_CUSTOM:
            return "CL_DEVICE_TYPE_CUSTOM";
        case CL_DEVICE_TYPE_ALL:
            return "CL_DEVICE_TYPE_ALL";
        default:
            return "UNKNOWN";
    }
}
