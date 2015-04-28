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

#pragma once 
#include "opencl_loader.h"
#include "opencl_timer.h"
#include "opencl_config.h"

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
    cl_int sample_width,
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
    param_float sigmaD,
    param_float corner_response_alpha,
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
    param_float sigmaD,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);

bool opencl_fd_harris_corner_count( struct FD* state,
    cl_mem corners_in,
    cl_mem strong_responses,
    cl_mem corner_count,
    param_float harris_threshold,
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
        param_float hessian_determinant_threshold,
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
