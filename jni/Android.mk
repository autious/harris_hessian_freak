LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_C_INCLUDES := jni/include/

#THe android_io include is intended to mask some stdio functions 
LOCAL_CFLAGS += -std=c99 -Wall -DPROFILE -include "src/android_io.h"

LOCAL_SRC_FILES := src/android_opencl_test_jni.c src/opencl_error.c src/opencl_loader.c src/opencl_test.c src/lodepng.c src/gauss_kernel.c src/opencl_util.c src/opencl_program.c src/opencl_fd.c src/harris_hessian_freak.c src/util.c src/freak.c src/opencl_timer.c src/android_io.c src/stb_image.c src/compile_flag_object.c
LOCAL_MODULE := android_opencl_test_jni
LOCAL_LDLIBS := -landroid -ldl -llog -lOpenCL

include $(LOCAL_PATH)/Generate.mk

%opencl_program.c: encodekernels
	echo "Generating kernels"
%opencl_timer.c: encodeversion
	echo "Encoding version"

include $(BUILD_SHARED_LIBRARY)

