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

#ifndef __ANDROID__

const int FREAK_NB_PAIRS = 512;
const int WORD_SIZE = 8;

int main( int argc, const char ** argv )
{
    if( argc == 2 )
    {
        LOGV("Running harris hessian on %s\n", argv[1] );
        unsigned int lode_error;
        unsigned int width;
        unsigned int height;
        uint8_t *data;

        lode_error = lodepng_decode32_file( &data, &width, &height, argv[1] );

        if( lode_error == 0 )
        {
            LOGV( "Picture dimensions: (%u,%u)", width, height );

            cl_event detection_event;
            harris_hessian_init( width, height ); 
            harris_hessian_detection( data, 0, NULL, &detection_event );
            int desc_count;

            descriptor * descriptors = harris_hessian_build_descriptor( &desc_count, 1, &detection_event, NULL );
            harris_hessian_close( );

            FILE* fp = fopen("out.desc", "wb+");
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

            free( descriptors );

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
