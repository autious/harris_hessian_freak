/*
* Copyright (c) 2015, Max Danielsson <work@autious.net> and Thomas Sievert
* All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
#include "opencl_error.h"
#include "stb_image.h"
#include "opencl_config.h"
#include "android_io.h"

static char* save_path = NULL;
static int width = 0;
static int height = 0;
static int n;
static uint8_t *data = NULL;

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

    if( data )
    {
        harris_hessian_freak_close( );
        free( data );
        data = NULL;
        width = 0;
        height = 0;
    }

    return true;
}

jstring Java_org_bth_HarrisHessianFreakJNI_getLibError( JNIEnv* env, jobject x )
{
    char *err = "err";
    return (*env)->NewStringUTF( env, err );
}

jboolean Java_org_bth_HarrisHessianFreakJNI_loadImage( JNIEnv* env, jobject x )
{
    if( data == NULL )
    {
        FILE* f_handle = android_io_fopen( "test.png", "r" );

        data = stbi_load_from_file( f_handle, &width, &height, &n, 4 );

        harris_hessian_freak_init( width, height ); 
        return true;
    }
    else
    {
        return false;
    }
}

jboolean Java_org_bth_HarrisHessianFreakJNI_runTest( JNIEnv* env, jobject x, jobject callback_object )
{
    const char* callback_name = "progress";
    const char* callback_type = "(I)V";

    jclass callback_class = (*env)->GetObjectClass( env, callback_object );
    jmethodID callback_method_id = (*env)->GetMethodID( env, callback_class, callback_name, callback_type );
    int progress = 0;
    int increment = 100/6;

#ifdef PROFILE
    opencl_timer_clear_events();
    PROFILE_MM( "program_walltime" ); 
    int start_marker = PROFILE_PM( full_pass, 0 );
#endif
    
    (*env)->CallVoidMethod( env, callback_object, callback_method_id, (progress += increment) );

    if( data )
    {
        LOGV( "Picture dimensions: (%u,%u)", width, height );

        cl_event detection_event;

        (*env)->CallVoidMethod( env, callback_object, callback_method_id, (progress += increment) );

        harris_hessian_freak_detection( data, save_path, 0, NULL, &detection_event );

        (*env)->CallVoidMethod( env, callback_object, callback_method_id, (progress += increment) );

        cl_event generate_keypoints_list_event;
        size_t keypoints_count;
        keyPoint* keypoints_list = harris_hessian_freak_generate_keypoint_list(
            &keypoints_count,
            1,
            &detection_event,
            &generate_keypoints_list_event
        );

        CL_RELEASE_EVENT( detection_event );

        (*env)->CallVoidMethod( env, callback_object, callback_method_id, (progress += increment) );


        size_t desc_count;
        descriptor * descriptors = harris_hessian_freak_build_descriptor( 
            keypoints_list, 
            keypoints_count, 
            &desc_count, 
            1, 
            &generate_keypoints_list_event, 
            NULL 
        );

        CL_RELEASE_EVENT( generate_keypoints_list_event );

        (*env)->CallVoidMethod( env, callback_object, callback_method_id, (progress += increment) );

#ifdef PROFILE 
        int end_marker = PROFILE_PM( full_pass, 0 );

        LOGV("Printing timing data");
        opencl_timer_push_segment( "first_to_last_kernel", start_marker, end_marker );
        PROFILE_MM( "program_walltime" ); 
        opencl_timer_print_results();
#endif

        free( descriptors );

        free( keypoints_list );

    }
    else
    {
        LOGE( "No image is loaded: %s", "file" );
        return false;
    }

    (*env)->CallVoidMethod( env, callback_object, callback_method_id, (progress = 100) );


    return true;
}

void Java_org_bth_HarrisHessianFreakJNI_setGaussXWorkgroup( JNIEnv* env, jobject e, jint x, jint y )
{
   local_work_size_gaussx[0] = x; 
   local_work_size_gaussx[1] = y; 
}

void Java_org_bth_HarrisHessianFreakJNI_setGaussYWorkgroup( JNIEnv* env, jobject e, jint x, jint y )
{
   local_work_size_gaussy[0] = x; 
   local_work_size_gaussy[1] = y; 
}
