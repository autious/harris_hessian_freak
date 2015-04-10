#pragma once 
#include "opencl_loader.h"
#include "opencl_timer.h"

struct FD
{
    cl_mem image_rgba_char; // "Image source buffer"
    //Following buffers are use when filters are applied to the greyscale image.
    cl_uint source_width;
    cl_uint source_height;

    cl_uint width; //Target size, working size
    cl_uint height; //Target size, working size
};

void opencl_fd_save_buffer_to_image( 
    const char * name, 
    struct FD* state,
    cl_mem in,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list );

bool opencl_fd_load_rgba( 
        uint8_t* data, 
        struct FD* state 
);

cl_mem opencl_fd_create_image_buffer( struct FD* state );
void opencl_fd_create_image_buffers( struct FD* state, cl_mem* buffers, size_t count );
void opencl_fd_release_image_buffer( struct FD* state, cl_mem buffer );
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

bool opencl_fd_second_moment_matrix_elements( struct FD* state,
    cl_mem ddx,
    cl_mem ddy,
    cl_mem xx,
    cl_mem xy,
    cl_mem yy,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);

bool opencl_fd_run_harris_corner_response( struct FD* state,
    cl_mem xx,
    cl_mem xy,
    cl_mem yy,
    cl_mem output,
    cl_float sigmaD,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);

bool opencl_fd_run_harris_corner_suppression( struct FD* state,
    cl_mem in,
    cl_mem out,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);

bool opencl_fd_run_hessian( struct FD* state,
    cl_mem xx,
    cl_mem xy,
    cl_mem yy,
    cl_mem out,
    cl_float sigmaD,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);

bool opencl_fd_harris_corner_count( struct FD* state,
    cl_mem corners_in,
    cl_mem strong_responses,
    cl_mem corner_count,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);

bool opencl_fd_find_keypoints( 
        struct FD* state,
        cl_mem source_det, 
        cl_mem corner_counts, 
        cl_mem keypoints_data, 
        cl_mem hessian_determinant_indices,
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
