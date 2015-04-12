#ifdef HH_USE_HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half hh_float;
#define LOAD_HHF(arr,offset) vload_half(offset,arr)
#define STORE_HHF(arr,offset,value) vstore_half(value,offset,arr)
#else
typedef float hh_float;
#define LOAD_HHF(arr,offset) arr[offset]
#define STORE_HHF(arr,offset,value) arr[offset]=value
#endif

typedef float param_float;

#define il(value,v_min,v_max) min(max(v_min,value),v_max)

#define HESSIAN_DETERMINANT_THRESHOLD 0.1f

#define SUP_HALFWIDTH 1

#define HARRIS_THRESHOLD 0.0f

#define CORNER_RESPONSE_ALPHA 0.04f
                    // I have found literally one lecture that explains that this alpha is 
                    // empirically measured to be [0.04 - 0.06] and another that just uses 
                    // 0.04 without ref
