#pragma once 

#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "harris_hessian_freak", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "harris_hessian_freak", __VA_ARGS__))

/* For debug builds, always enable the debug traces in this library */
#ifndef NDEBUG
#  define LOGV(...)  ((void)__android_log_print(ANDROID_LOG_VERBOSE, "opencl_app", __VA_ARGS__))
#else
#  define LOGV(...)  ((void)0)
#endif
#else

#include <stdio.h>

#define LOGI( ... )((void)fprintf( stderr, __VA_ARGS__ ));fprintf( stderr, "\n")
#define LOGE( ... )((void)fprintf( stderr, __VA_ARGS__ ));fprintf( stderr, "\n")
#define LOGV( ... )((void)fprintf( stderr, __VA_ARGS__ ));fprintf( stderr, "\n")

#endif
