#pragma once
#include <stdio.h>
#include <errno.h>

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

FILE* android_io_fopen( const char* fname, const char* mode );
void android_io_set_asset_manager( AAssetManager* mgr );

//#define fopen(name, mode) android_io_fopen(name, mode)
#endif
