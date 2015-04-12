#include <stdbool.h>

#include <string.h>
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <CL/opencl.h>

#include "log.h"
#include "util.h"
#include "harris_hessian_freak.h"
#include "freak.h"
#include "opencl_timer.h"
#include "stb_image.h"
#include "opencl_config.h"

jboolean Java_org_bth_opencltestjni_OpenCLTestJNI_initLib( JNIEnv* env, jobject x, jobject assetManager )
{
    AAssetManager *mgr = AAssetManager_fromJava( env, assetManager );
    
    android_io_set_asset_manager( mgr );
    opencl_run_reference_mode = true;

    return true;
}

jboolean Java_org_bth_opencltestjni_OpenCLTestJNI_closeLib( JNIEnv* env, jobject x )
{
    android_io_set_asset_manager( NULL );
    return true;
}

jstring Java_org_bth_opencltestjni_OpenCLTestJNI_getLibError( JNIEnv* env, jobject x )
{
    char *err = "err";
    return (*env)->NewStringUTF( env, err );
}

jboolean Java_org_bth_opencltestjni_OpenCLTestJNI_runTest( JNIEnv* env, jobject x )
{
    //opencl_test_run();
#ifdef PROFILE
    int start_marker = PROFILE_PM( full_pass, 0 );
    PROFILE_MM( "full_hh_freak" ); 
#endif
    
    int width;
    int height;
    int n;
    uint8_t *data;

    data = stbi_load( "test.png", &width, &height, &n, 4 );

    if( data )
    {
        LOGV( "Picture dimensions: (%u,%u)", width, height );

        cl_event detection_event;
        harris_hessian_freak_init( width, height ); 
        harris_hessian_freak_detection( data, NULL, 0, NULL, &detection_event );

        cl_event generate_keypoints_list_event;
        size_t keypoints_count;
        keyPoint* keypoints_list = harris_hessian_freak_generate_keypoint_list(
            &keypoints_count,
            1,
            &detection_event,
            &generate_keypoints_list_event
        );


        size_t desc_count;
        descriptor * descriptors = harris_hessian_freak_build_descriptor( 
            keypoints_list, 
            keypoints_count, 
            &desc_count, 
            1, 
            &generate_keypoints_list_event, 
            NULL 
        );

#ifdef PROFILE 
        PROFILE_MM( "full_hh_freak" ); 
        int end_marker = PROFILE_PM( full_pass, 0 );

        LOGV("Printing timing data");
        opencl_timer_push_segment( "full_pass", start_marker, end_marker );
        opencl_timer_print_results();
#endif

        free( descriptors );

        free( keypoints_list );

        harris_hessian_freak_close( );
    }
    else
    {
        LOGE( "Unable to load image: %s", "file" );
    }

    free( data );

    return true;
}
