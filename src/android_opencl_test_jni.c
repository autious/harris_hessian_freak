#include <stdbool.h>

#include <string.h>
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "log.h"
#include "util.h"
#include "opencl_loader.h"
#include "opencl_program.h"
#include "android_io.h"
#include "opencl_error.h"

jboolean Java_org_bth_opencltestjni_OpenCLTestJNI_initLib( JNIEnv* env, jobject x, jobject assetManager )
{
    AAssetManager *mgr = AAssetManager_fromJava( env, assetManager );
    
    android_io_set_asset_manager( mgr );
    opencl_program_add_compiler_flag( "-cl-std=CL1.1" );
    return opencl_loader_init();
}

jboolean Java_org_bth_opencltestjni_OpenCLTestJNI_closeLib( JNIEnv* env, jobject x )
{
    android_io_set_asset_manager( NULL );
    opencl_loader_close();
    return true;
}

jstring Java_org_bth_opencltestjni_OpenCLTestJNI_getLibError( JNIEnv* env, jobject x )
{
    size_t size = opencl_loader_get_error_size();
    char err[size];
    opencl_loader_get_error( err );
    return (*env)->NewStringUTF( env, err );
}

jboolean Java_org_bth_opencltestjni_OpenCLTestJNI_runTest( JNIEnv* env, jobject x )
{
        opencl_test_run();
}
