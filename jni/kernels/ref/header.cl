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

#define SUP_HALFWIDTH 1
