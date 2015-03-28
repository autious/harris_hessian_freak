#define _POSIX_C_SOURCE 2

#ifndef __ANDROID__

#include <unistd.h>
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

void print_help()
{
    printf( "Usage: hh_freak_detector [-ht] [-k FILE] [-p FILE] [-d FILE] FILE\n" );
}

int main( int argc, char * const *argv )
{
    int opt;
    const char* k_name = NULL;
    const char* p_name = NULL;
    const char* d_name = NULL;
    bool output_timing = false;

    while((opt = getopt(argc,argv,"htk:p:d:")) != -1) 
    {
        switch(opt)
        {
            case 'h':
                print_help(); 
                return 2;
                break; 
            case 't':
                output_timing = true;
                break;
            case 'k':
                k_name = optarg;
                break;
            case 'p':
                p_name = optarg;
                break;
            case 'd':
                d_name = optarg;
                break;
            default:
                break;
        } 
    }

    if( optind == argc - 1 )
    {
        fprintf( stderr, "Running harris hessian on %s\n", argv[optind] );
        int width;
        int height;
        int n;
        uint8_t *data;

        data = stbi_load( argv[optind], &width, &height, &n, 4 );

        if( data )
        {
            fprintf( stderr, "Picture dimensions: (%u,%u)", width, height );

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

            if( k_name )
            {
                fprintf( stderr, "Saving keypoints to: %s\n" , k_name );
                save_keypoints( k_name, keypoints_list, keypoints_count );
            }
            
            if( p_name )
            {
                fprintf( stderr, "Saving keypoint image to: %s\n" , p_name );
                save_keypoints_image( p_name, keypoints_list, keypoints_count, data, width, height );
            }

            size_t desc_count;
            descriptor * descriptors = harris_hessian_build_descriptor( 
                keypoints_list, 
                keypoints_count, 
                &desc_count, 
                1, 
                &generate_keypoints_list_event, 
                NULL 
            );

            if( d_name )
            {
                fprintf( stderr, "Saving descriptor to: %s\n" , d_name );
                save_descriptor( d_name, descriptors, desc_count );
            }

            free( descriptors );

            free( keypoints_list );

            harris_hessian_close( );

        }
        else
        {
            fprintf( stderr, "Unable to load image: %s", argv[optind] );
        }

        free( data );
    }
    else
    {
        fprintf( stderr, "Missing input file (must be last argument)\n" );
    }

    return 0;
}

#endif
