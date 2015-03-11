#pragma once 
#include "opencl_handler.h"

struct FD
{
    uint8_t *rgba_host; //Memory of image_rgba_char
    cl_mem image_rgba_char; // "Image source buffer"
    //Following buffers are use when filters are applied to the greyscale image.
    cl_uint width;
    cl_uint height;
};

bool opencl_fd_load_png_file( const char *input_filename, struct FD* state );

void opencl_fd_save_buffer_to_image( 
    const char * name, 
    struct FD* state,
    cl_mem in,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list );

void opencl_fd_create_image_buffers( struct FD* state, cl_mem* buffers, size_t count );
void opencl_fd_release_image_buffers( struct FD* state, cl_mem* buffers, size_t count );

bool opencl_fd_run_gaussxy( 
    float sigma, 
    struct FD* state,
    cl_mem in, //Can be same as out
    cl_mem middle,
    cl_mem out, //Can be same as in
    cl_uint num_events_in_wait_list, 
    const cl_event* event_wait_list, 
    cl_event* event 
);

bool opencl_fd_desaturate_image( 
    struct FD *state,
    cl_mem out,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);

bool opencl_fd_derivate_image( struct FD* state,
    cl_mem in,
    cl_mem ddxout,
    cl_mem ddyout,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);

void opencl_fd_free( struct FD* state, 
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list
);

/*
bool opencl_fd_run_gauss2d(
    struct FD* state,
    float sigma, 
    cl_uint num_events_in_wait_list, 
    const cl_event* event_wait_list, 
    cl_event* event 
);
*/
