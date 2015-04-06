#!/bin/sh

/home/max/tools/android-ndk-r10d/ndk-build clean &&\
/home/max/tools/android-ndk-r10d/ndk-build &&\
ant debug &&\
adb install -r ./bin/android_opencl_test_jni-debug.apk

