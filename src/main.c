#include <string.h>

#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "opencl_program.h"
#include "opencl_fd.h"
#include "harris_hessian.h"
#include "log.h"
#include "util.h"
#include "lodepng.h"
#include "freak.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef __ANDROID__


const int FREAK_NB_PAIRS = 512;
const int WORD_SIZE = 8;

static void save_keypoints_image( const char * filename, const keyPoint* keypoints, size_t keypoint_count, const uint8_t *data, int width, int height )
{
    uint8_t *dest_data = malloc( sizeof( uint8_t ) * 4 * width * height );
    memcpy( dest_data, data, sizeof( uint8_t ) * 4 * width * height );

    for( int i = 0; i < keypoint_count; i++ )
    {
        int x = keypoints[i].x;
        int y = keypoints[i].y;
        dest_data[(x+y*width)*4+0] = 255;
        dest_data[(x+y*width)*4+1] = 0;
        dest_data[(x+y*width)*4+2] = 0;
        dest_data[(x+y*width)*4+3] = 255;
    }

    size_t len = strlen( filename ) + strlen( ".png" ) + 1;
    char name[len];
    snprintf( name, len, "%s.png", filename );
    lodepng_encode_file( name, dest_data, width, height, LCT_RGBA, 8 );
    free( dest_data );
}

static void save_descriptor( const  char *filename, descriptor* descriptors, size_t desc_count )
{
    FILE* fp = fopen( filename, "wb+");
    if (fp) {
        fwrite(&desc_count, sizeof(int), 1, fp);
        for (int i = 0; i < desc_count; ++i) {
            fwrite(descriptors[i].data, sizeof(*descriptors[i].data), FREAK_NB_PAIRS / WORD_SIZE, fp);
            fwrite(&descriptors[i].x, sizeof(descriptors[i].x), 1, fp);
            fwrite(&descriptors[i].y, sizeof(descriptors[i].y), 1, fp);
        }
        fclose(fp);
    }
    else printf("Can't create descriptor file\n");
}

static void save_keypoints( const char* filename, keyPoint* keypoints, size_t count   )
{
    int filenamelen = strlen(filename)+strlen(".kpts")+1;
    char name[filenamelen];
    snprintf( name, filenamelen, "%s.kpts", filename );
    FILE* f = fopen( name, "w" );

    LOGV( "Outputting\n" );

    for( int i = 0; i < count; i++ )
    {
        int x = keypoints[i].x;
        int y = keypoints[i].y;
        float scale = keypoints[i].size;
        fprintf( f, "%d,%d,%f\n",x,y,scale);
    }

    fclose( f );
}

int main( int argc, const char ** argv )
{
    if( argc == 2 )
    {
        LOGV("Running harris hessian on %s\n", argv[1] );
        int width;
        int height;
        int n;
        uint8_t *data;

        data = stbi_load( argv[1], &width, &height, &n, 4 );

        if( data )
        {
            LOGV( "Picture dimensions: (%u,%u)", width, height );

            cl_event detection_event;
            harris_hessian_init( width, height ); 
            harris_hessian_detection( data, 0, NULL, &detection_event );

            cl_event generate_keypoints_list_event;
            size_t keypoints_count;
            keyPoint* keypoints_list = harris_hessian_generate_keypoint_list(
                &keypoints_count,
                1,
                &detection_event,
                &generate_keypoints_list_event
            );

            save_keypoints( "out", keypoints_list, keypoints_count );
            save_keypoints_image( "keypoints_image", keypoints_list, keypoints_count, data, width, height );

            size_t desc_count;
            descriptor * descriptors = harris_hessian_build_descriptor( 
                keypoints_list, 
                keypoints_count, 
                &desc_count, 
                1, 
                &generate_keypoints_list_event, 
                NULL 
            );

            save_descriptor( "out.desc", descriptors, desc_count );
            free( descriptors );

            free( keypoints_list );

            harris_hessian_close( );

        }
        else
        {
            LOGE("Unable to load image: %s", argv[1] );
        }

        free( data );
    }
    else
    {
        LOGE( "Program needs exactly one argument" );
    }

    return 0;
}

#endif
