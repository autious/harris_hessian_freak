#ifdef HH_USE_HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half hh_float;
typedef half2 hh_float2;
typedef half4 hh_float4;
typedef half8 hh_float8;
#else
typedef float hh_float;
typedef float2 hh_float2;
typedef float4 hh_float4;
typedef float8 hh_float8;
#endif

typedef float param_float;

#define il(value,v_min,v_max) min(max(v_min,value),v_max)

#define HESSIAN_DETERMINANT_THRESHOLD 0.000001f

#define SUP_HALFWIDTH 1

#define HARRIS_THRESHOLD 0.000001f

#define CORNER_RESPONSE_ALPHA 0.04f
                    // I have found literally one lecture that explains that this alpha is 
                    // empirically measured to be [0.04 - 0.06] and another that just uses 
                    // 0.04 without ref
