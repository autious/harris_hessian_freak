#pragma once

#include <CL/opencl.h>

const char* opencl_error_codename( cl_int err );

#define CLERR(string,code)LOGE("%s:%s", string , opencl_error_codename( code ))


