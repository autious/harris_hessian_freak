LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_C_INCLUDES := jni/include/

#THe android_io include is intended to mask some stdio functions 
LOCAL_CFLAGS += -std=c99 -Wall -DPROFILE -include "src/android_io.h"

include $(LOCAL_PATH)/Sources.mk

LOCAL_SRC_FILES := $(addprefix src/,$(LIBRARY_SOURCES))
LOCAL_SRC_FILES += src/android_opencl_test_jni.c src/android_io.c

LOCAL_MODULE := android_opencl_test_jni
LOCAL_LDLIBS := -landroid -ldl -llog -lOpenCL

include $(LOCAL_PATH)/Generate.mk

%opencl_program.c: encodekernels
	echo "Generating kernels"
%opencl_timer.c: encodeversion
	echo "Encoding version"

include $(BUILD_SHARED_LIBRARY)
