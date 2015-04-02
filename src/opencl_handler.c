#include "opencl_handler.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "util.h"
#include "opencl_error.h"
#include "opencl_timer.h"
#include "gauss_kernel.h"

static const char* lastCLError = "";

static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue command_queue = NULL;

static void context_callback( 
        const char* errinfo, 
        const void *private_info, 
        size_t cb, 
        void *user_data
    )
{
    LOGE("Context callback error: %s", errinfo);
}

struct DeviceList
{
    int platform_index;
    int device_index;
    cl_platform_id platform;
    cl_device_id device;
    uint32_t rank;
    struct DeviceList *next;
};

static struct DeviceList* device_list = NULL;

/*
 * Linked list construction
*/
static struct DeviceList* device_list_add( 
        cl_platform_id platform, 
        cl_device_id device, 
        uint32_t rank, 
        int platform_index, 
        int device_index
    )
{
    struct DeviceList **cur = &device_list;

    while( *cur != NULL )
    {
        cur = &((*cur)->next);
    }

    *cur = (struct DeviceList*)malloc( sizeof( struct DeviceList ) );

    (*cur)->platform = platform;
    (*cur)->device = device;
    (*cur)->next = NULL;
    (*cur)->rank = rank;
    (*cur)->platform_index = platform_index;
    (*cur)->device_index = device_index;
    return *cur;
};

/*
 * Function to quickly generate a "Ranking" for a device
 * to simplify the process of electing
 */
static uint32_t device_calculate_rank( 
        cl_device_type type, 
        cl_bool unified_memory, 
        cl_bool compiler_available )
{
    return (type == CL_DEVICE_TYPE_GPU ? 1UL << 5 : 0) |
           (unified_memory ? 1UL << 4 : 0) |
           (compiler_available ? 1UL << 3 : 0);
}

static void device_list_free( )
{
    struct DeviceList *cur = device_list;
    while( cur != NULL )
    {
        struct DeviceList *next = cur->next;

        free( cur );

        cur = next;
    } 

    device_list = NULL;
};

static struct DeviceList* device_list_best_ranked()
{
    struct DeviceList *cur = device_list;
    struct DeviceList *best = NULL;
    uint32_t rank = 0;

    while( cur != NULL )
    {
        if( cur->rank > rank )
        {
            best = cur;
            rank = cur->rank;
        }
        cur = cur->next;
    } 

    return best;
}

/* UNNUSED
static size_t device_list_count()
{
    int count = 0;
    struct DeviceList *cur = device_list;
    while( cur != NULL )
    {
        cur = cur->next;
        count++;
    }
    return count;
}
*/

static void generate_device_list()
{
    LOGV( "OpenCL platform information...");

    cl_uint num_plats = 0;
    clGetPlatformIDs( 0 , NULL, &num_plats );
    if( num_plats > 0 )
    {
        cl_platform_id plat_ids[num_plats];
        cl_int clRetErr = clGetPlatformIDs( num_plats, plat_ids, NULL );
        if( clRetErr == CL_SUCCESS )
        {
            cl_platform_info types[] = { 
                    CL_PLATFORM_PROFILE, 
                    CL_PLATFORM_NAME, 
                    CL_PLATFORM_VERSION, 
                    CL_PLATFORM_VENDOR,
                    CL_PLATFORM_EXTENSIONS 
            };

            for( cl_uint i = 0; i < num_plats; i++ )
            {
                LOGV( "Platform index %d", i );

                for( unsigned int k = 0; k < NELEMS( types ); k++ )
                {
                    size_t info_size;
                    clGetPlatformInfo( plat_ids[i], types[k], 0, NULL, &info_size );
                    char info[info_size];
                    clGetPlatformInfo( plat_ids[i], types[k], info_size, info, NULL );
                    LOGV( "%s:%s", opencl_platform_info_codename(types[k]), info );
                }

                cl_uint num_devices = 0;
                cl_int device_err = clGetDeviceIDs( 
                    plat_ids[i], 
                    CL_DEVICE_TYPE_ALL, 
                    0, 
                    NULL, 
                    &num_devices 
                );

                if( num_devices > 0 && device_err == CL_SUCCESS )
                {
                    cl_device_id device_ids[num_devices];
                    device_err = clGetDeviceIDs( 
                        plat_ids[i], 
                        CL_DEVICE_TYPE_ALL, 
                        num_devices, 
                        device_ids, 
                        NULL
                    ); 

                    if( device_err == CL_SUCCESS )
                    {
                        for( cl_uint l = 0; l < num_devices; l++ )
                        {
                            cl_device_type device_type;
                            cl_bool unified_memory;
                            cl_bool compiler_available;
                            cl_ulong max_constant_buffer_size;
                            cl_uint max_constant_args;
                            cl_bool image_support;

                            cl_device_info device_param_names[] = {
                                CL_DEVICE_VENDOR,
                                CL_DEVICE_NAME,
                                CL_DEVICE_PROFILE,
                                CL_DEVICE_TYPE,
                                CL_DEVICE_HOST_UNIFIED_MEMORY,
                                CL_DEVICE_COMPILER_AVAILABLE,
                                CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                                CL_DEVICE_MAX_CONSTANT_ARGS,
                                CL_DEVICE_IMAGE_SUPPORT
                            };

                            void* device_param_values[] = {
                                NULL,
                                NULL,
                                NULL,
                                &device_type,
                                &unified_memory,
                                &compiler_available,
                                &max_constant_buffer_size,
                                &max_constant_args,
                                &image_support
                            };

                            size_t device_param_size[] = {
                                0,
                                0,
                                0,
                                sizeof( cl_device_type ),
                                sizeof( cl_bool ),
                                sizeof( cl_bool ),
                                sizeof( cl_ulong ),
                                sizeof( cl_uint ),
                                sizeof( cl_bool )
                            };

                            bool device_param_print[] = {
                                true,
                                true,
                                true,
                                true,
                                true,
                                true,
                                true,
                                true,
                                true
                            };

                            const int N_STRING = 0;
                            const int N_CL_DEVICE_TYPE = 1;
                            const int N_BOOL = 2;
                            const int N_ULONG = 3;
                            const int N_UINT = 4;

                            int print_type[] = {
                                N_STRING,
                                N_STRING,
                                N_STRING,
                                N_CL_DEVICE_TYPE,
                                N_BOOL,
                                N_BOOL,
                                N_ULONG,
                                N_UINT,
                                N_BOOL
                            };

                            LOGV( "Device index %d", l );
                            for( unsigned int m = 0; m < NELEMS( device_param_names ); m++ )
                            {
                                if( device_param_values[m] == NULL )
                                {
                                    clGetDeviceInfo( 
                                            device_ids[l], 
                                            device_param_names[m], 
                                            0, 
                                            NULL, 
                                            &(device_param_size[m]) 
                                        );
                                }

                                char device_info_data[device_param_size[m]];
                                void* d = 
                                    device_param_values[m] == NULL 
                                    ? device_info_data 
                                    : device_param_values[m];

                                clGetDeviceInfo( 
                                        device_ids[l], 
                                        device_param_names[m], 
                                        device_param_size[m], 
                                        d, 
                                        NULL
                                );

                                if( device_param_print[m] )
                                {
                                    if( print_type[m] == N_STRING )
                                    {
                                        LOGV( 
                                            "%s:%s", 
                                            opencl_device_info_codename( device_param_names[m] ), 
                                            (char*)d
                                        );
                                    }
                                    else if( print_type[m] == N_CL_DEVICE_TYPE )
                                    {
                                        LOGV( 
                                            "%s:%s", 
                                            opencl_device_info_codename( device_param_names[m] ), 
                                            opencl_device_type_codename(*(cl_device_type*)d) 
                                        );
                                    }
                                    else if( print_type[m] == N_BOOL )
                                    {
                                    
                                        LOGV( 
                                            "%s:%s", 
                                            opencl_device_info_codename( device_param_names[m] ), 
                                            *(cl_bool*)d ? "true" : "false" 
                                        );
                                    }
                                    else if( print_type[m] == N_ULONG )
                                    {
                                        long unsigned int cast_val = *(cl_ulong*)d;
                                        LOGV( 
                                            "%s:%lu", 
                                            opencl_device_info_codename( device_param_names[m] ), 
                                            cast_val 
                                        );
                                    }
                                    else if( print_type[m] == N_UINT )
                                    {
                                        unsigned int cast_val = *(cl_uint*)d;
                                        LOGV( 
                                            "%s:%u", 
                                            opencl_device_info_codename( device_param_names[m] ), 
                                            cast_val 
                                        );
                                    }

                                }
                            }

                            device_list_add( 
                                    plat_ids[i], 
                                    device_ids[l], 
                                    device_calculate_rank( 
                                        device_type, 
                                        unified_memory, 
                                        compiler_available ),
                                    i,
                                    l
                                );
                        }
                    }
                    else
                    {
                        LOGE( "Platform failed at listing devices." );
                    }
                }
                else
                {
                    LOGV( 
                        "Platform lacks devices or is unable to retrieve, "
                        "this is odd. CL_ERR_CODE: %s %d",
                        opencl_error_codename( device_err ), 
                        device_err );
                }
            }
        } 
        else
        {
            LOGE( "Unable to list available platforms" );
        }
    }
    else
    {
        LOGE("No platforms available, listing return zero.");
    }
}

bool opencl_loader_init()
{
    LOGV( "Generating device list..." );
    generate_device_list();
    LOGV( "Done" );

    struct DeviceList* device_l = device_list_best_ranked();

    device = device_l->device;

    if( device_l != NULL )
    {
        LOGV( "Opening best device" );
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)device_l->platform,
            0, 0
        };

        cl_device_id devices[] = {
            device
        };

        LOGV( 
            "Creating context on device p:%d d:%d", 
            device_l->platform_index, 
            device_l->device_index 
        );

        cl_int errcode_ret;
        context = clCreateContext(
                props, 
                NELEMS( devices ), 
                devices, 
                context_callback, 
                NULL, 
                &errcode_ret 
        );
        
        if( errcode_ret != CL_SUCCESS )
        {
            LOGE(
                "Unable to create opencl context: %s", 
                opencl_error_codename( errcode_ret ) 
            ); 
            return false;
        }

        LOGV( "Creating command queue" );

        cl_command_queue_properties cq_props 
            = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

#ifdef PROFILE
        if( opencl_timer_enable_profile ) 
        {
            cq_props |= CL_QUEUE_PROFILING_ENABLE;
        }
#endif


        command_queue = clCreateCommandQueue( 
                context, 
                device, 
                cq_props, 
                &errcode_ret
        );

        if( errcode_ret != CL_SUCCESS )
        {
            LOGE( 
                "Unable to create command queue: %s", 
                opencl_error_codename( errcode_ret ) 
            );
            return false;
        }

        return true;
    }
    else
    {
        LOGE( "No suitable devices are available" );
        return false;
    }
}

void opencl_loader_close()
{
    LOGV( "Closing opencl" );

    LOGV( "Closing device" );

    if( context )
    {
        clReleaseContext( context );
        context = NULL;
    }

    LOGV( "Emptying device list" );

    device_list_free();
}

void opencl_loader_get_error( char* value )
{
    memcpy( value, lastCLError, strlen( lastCLError ) + 1 );
}

size_t opencl_loader_get_error_size()
{
    return strlen( lastCLError ) + 1;
}


cl_context opencl_loader_get_context()
{
    return context;
}

cl_command_queue opencl_loader_get_command_queue()
{
    return command_queue;
}

cl_device_id opencl_loader_get_device()
{
    return device;
}

/*
 * Creates kernel from given program, need to be released using clReleaseKernel
 */
cl_kernel opencl_loader_load_kernel( cl_program program, const char* kernelname )
{
    cl_kernel kernel = NULL;

    if( program )
    {
        cl_int errcode_ret;
        kernel = clCreateKernel( program, kernelname, &errcode_ret );

        if( errcode_ret != CL_SUCCESS )
        {
            CLERR( ":Unable to create kernel from program", errcode_ret );
        }
    }

    return kernel;
}
