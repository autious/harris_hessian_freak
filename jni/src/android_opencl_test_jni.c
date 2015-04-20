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
#include "android_io.h"

static char* save_path = NULL;

jboolean Java_org_bth_HarrisHessianFreakJNI_initLib( JNIEnv* env, jobject x, jobject assetManager )
{
    AAssetManager *mgr = AAssetManager_fromJava( env, assetManager );
    
    android_io_set_asset_manager( mgr );
    opencl_run_reference_mode = false;

    return true;
}

jboolean Java_org_bth_HarrisHessianFreakJNI_setSaveFolder( JNIEnv* env, jobject x, jbyteArray path )
{
    int len = (*env)->GetArrayLength( env, path ); 

    save_path = realloc(save_path, sizeof( char ) * (len + 1));
    memset( save_path, '\0', sizeof( char ) * (len + 1) );

    (*env)->GetByteArrayRegion( env, path, 0, len, (jbyte*)save_path );

    return true;
}

jboolean Java_org_bth_HarrisHessianFreakJNI_closeLib( JNIEnv* env, jobject x )
{
    android_io_set_asset_manager( NULL );
    return true;
}

jstring Java_org_bth_HarrisHessianFreakJNI_getLibError( JNIEnv* env, jobject x )
{
    char *err = "err";
    return (*env)->NewStringUTF( env, err );
}

jboolean Java_org_bth_HarrisHessianFreakJNI_runTest( JNIEnv* env, jobject x )
{
#ifdef PROFILE
    opencl_timer_clear_events();
    int start_marker = PROFILE_PM( full_pass, 0 );
    PROFILE_MM( "full_hh_freak" ); 
#endif
    
    int width;
    int height;
    int n;
    uint8_t *data;
    
    FILE* f_handle = android_io_fopen( "test.png", "r" );
    data = stbi_load_from_file( f_handle, &width, &height, &n, 4 );

    if( data )
    {
        LOGV( "Picture dimensions: (%u,%u)", width, height );

        cl_event detection_event;
        harris_hessian_freak_init( width, height ); 
        harris_hessian_freak_detection( data, save_path, 0, NULL, &detection_event );

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
