#pragma once

#include <CL/opencl.h>

const char* opencl_error_codename( cl_int err );
const char* opencl_device_info_codename( cl_device_info devinfo );
const char* opencl_platform_info_codename( cl_platform_info platinfo );
const char* opencl_device_type_codename( cl_device_type devtype );

#define CLERR(string,code)LOGE("%s:%s", string , opencl_error_codename( code ))
#define CHKBUF(buf, code)\
    if(code != CL_SUCCESS)\
    {\
        CLERR( "Unable to create cl memory buffer" #buf, code );\
    }

    


