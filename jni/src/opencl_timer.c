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

#ifdef PROFILE

#define _POSIX_C_SOURCE 199309L

#include "opencl_timer.h"

#include <string.h>
#include <time.h>

#include "opencl_error.h"
#include "_commit_data.h"

#define TIMER_EVENT_NAME_LENGTH 32
#define TIMER_EVENT_INFO_LENGTH 32

static const int TIMER_STACK_SIZE_INCREASE = 256;

enum TimerEventType
{
    EVENT,
    MARKER,
    SEGMENT,
    MONOTONIC
};

struct TimerEvent
{
    char name[TIMER_EVENT_NAME_LENGTH];
    char info[TIMER_EVENT_INFO_LENGTH];

    enum TimerEventType type;

    union 
    {
        cl_event event;
        int reoccurance;
        int segment[2];
        struct timespec monotonic;
    } val;

};

static struct TimerEvent * timer_stack = NULL;

static size_t timer_stack_count = 0;
static size_t timer_stack_size = 0;

static void resize()
{
    if( timer_stack_count + 1 > timer_stack_size )
    {
        timer_stack_size += TIMER_STACK_SIZE_INCREASE;
        timer_stack = (struct TimerEvent*)realloc( timer_stack, sizeof(struct TimerEvent) * timer_stack_size );
    }       
}

static int get_first_following_event( int position )
{
    while( position < timer_stack_count && timer_stack[position].type != EVENT )
    {
        position++;
    }

    if( position >= timer_stack_count)
        position = -1;

    return position;
}

static int get_first_prior_event( int position )
{
    while( position >= 0 && timer_stack[position].type != EVENT )
    {
        position--;
    }

    return position;
}

static int get_matching_prev_monotonic( const int startposition )
{
    int position = startposition-1;
    while( position >= 0 
            && !(timer_stack[position].type == MONOTONIC 
                && strcmp(timer_stack[position].name, timer_stack[startposition].name ) == 0 ) )
    {
        position--; 
    }

    return position;
}

static double nano_to_milli( cl_ulong nano )
{
    return ((double)nano / 1000.0 / 1000.0);
}

void opencl_timer_push_event( const char* name, const char* info, cl_event event )
{
    clRetainEvent( event ); 
    resize();

    snprintf(timer_stack[timer_stack_count].name, TIMER_EVENT_NAME_LENGTH, "%s", name );
    snprintf(timer_stack[timer_stack_count].info, TIMER_EVENT_INFO_LENGTH, "%s", info );

    timer_stack[timer_stack_count].val.event = event;
    timer_stack[timer_stack_count].type = EVENT;
    timer_stack_count++;
}

int opencl_timer_push_marker( const char* name, int reoccurance )
{
    resize();
    
    snprintf( timer_stack[timer_stack_count].name, TIMER_EVENT_NAME_LENGTH, "%s", name );
    timer_stack[timer_stack_count].val.reoccurance = reoccurance;
    timer_stack[timer_stack_count].type = MARKER;
    return timer_stack_count++;
}

int opencl_timer_push_monotonic( const char* name )
{
    resize();
    snprintf( timer_stack[timer_stack_count].name, TIMER_EVENT_NAME_LENGTH, "%s", name );
    clock_gettime( CLOCK_MONOTONIC, &timer_stack[timer_stack_count].val.monotonic );
    timer_stack[timer_stack_count].type = MONOTONIC;
    return timer_stack_count++;
}

#define MAP_STRING_DOUBLE_NAME_SIZE 32

struct MapStringDouble
{
    char name[MAP_STRING_DOUBLE_NAME_SIZE];
    double value;
    struct MapStringDouble *next;  
};

static void mapstringdouble_add( struct MapStringDouble** start, const char* name, double value )
{
    struct MapStringDouble **cur = start;

    while( *cur != NULL && strncmp( (**cur).name, name, MAP_STRING_DOUBLE_NAME_SIZE - 1 ) != 0 )
    {
        cur = &(**cur).next;
    } 

    if( *cur == NULL )
    {
        *cur = (struct MapStringDouble*)malloc( sizeof( struct MapStringDouble ) );

        (**cur).next = NULL; 
        snprintf( (**cur).name, MAP_STRING_DOUBLE_NAME_SIZE, "%s", name );
    }

    (**cur).value += value;
}

static void mapstringdouble_clear( struct MapStringDouble* start )
{
    while( start != NULL )
    {
        void *tmp = start; 
        start = start->next;
        free( tmp );
    }
}

void opencl_timer_push_segment( const char* name, int start, int end )
{
    resize();
    
    strncpy( timer_stack[timer_stack_count].name, name, TIMER_EVENT_NAME_LENGTH );
    timer_stack[timer_stack_count].val.segment[0] = start;
    timer_stack[timer_stack_count].val.segment[1] = end;
    timer_stack[timer_stack_count].type = SEGMENT;
    timer_stack_count++;
}

void opencl_timer_print_results( )
{
    const char ET_FORMAT[] = "__TIMER__ %-10s %32s: %fms \"%s\"";
    const char T_FORMAT[] = "__TIMER__ %-10s %32s: %fms";
    const char M_FORMAT[] = "__TIMER__ %-10s %32s: %d";
    const char E_FORMAT[] = "__TIMER__ %-10s %32s:";
    cl_ulong time_start, time_end;
    cl_int errcode_ret;

    int segment_start, segment_end;
    time_t d_secs;
    long d_nsec;

    double tmp;

    struct MapStringDouble *event_sum = NULL;

    LOGV( "__TIMER__ BEGIN" );
    time_t t = time( NULL );
    char * t_s = ctime( &t );
    LOGV( "__TIMER__ %s", t_s ? t_s : "" );
    LOGV( "__TIMER__ %s", git_commit_info );

    for( int i = 0; i < timer_stack_count; i++ )
    {
        switch( timer_stack[i].type )
        {
            case EVENT:
                errcode_ret = clGetEventProfilingInfo( timer_stack[i].val.event,
                    CL_PROFILING_COMMAND_START,
                    sizeof( cl_ulong ),
                    &time_start,
                    NULL 
                );
                ASSERT_PROF( time_start, errcode_ret );

                errcode_ret = clGetEventProfilingInfo( timer_stack[i].val.event,
                    CL_PROFILING_COMMAND_END,
                    sizeof( cl_ulong ),
                    &time_end,
                    NULL
                );
                ASSERT_PROF( time_end, errcode_ret );
                tmp = nano_to_milli(time_end-time_start);
                LOGV( ET_FORMAT,
                    "EVENT",
                    timer_stack[i].name,
                    tmp,
                    timer_stack[i].info
                );
                mapstringdouble_add( &event_sum, timer_stack[i].name, tmp );
                break;

            case MARKER:
                LOGV(
                    M_FORMAT,
                    "MARKER",
                    timer_stack[i].name,
                    timer_stack[i].val.reoccurance
                );
                break;
            case MONOTONIC:
                segment_start = get_matching_prev_monotonic( i );
                segment_end = i;

                if( segment_start >= 0 )
                {
                    d_secs 
                        = timer_stack[segment_end].val.monotonic.tv_sec
                        - timer_stack[segment_start].val.monotonic.tv_sec;
                    d_nsec 
                        = timer_stack[segment_end].val.monotonic.tv_nsec
                        - timer_stack[segment_start].val.monotonic.tv_nsec;
                    LOGV( 
                        T_FORMAT,
                        "MT_END",
                        timer_stack[i].name,
                        nano_to_milli(d_secs*1.0E9+d_nsec)
                    );
                }
                else
                {
                    LOGV( 
                        E_FORMAT,
                        "MT_START",
                        timer_stack[i].name
                    );
                }
                break;

            case SEGMENT:
                segment_start = get_first_following_event( timer_stack[i].val.segment[0] );
                segment_end =   get_first_prior_event( timer_stack[i].val.segment[1] );

                if( segment_start != -1 && segment_end != -1 )
                {
                    errcode_ret = clGetEventProfilingInfo( timer_stack[segment_start].val.event,
                        CL_PROFILING_COMMAND_START,
                        sizeof( cl_ulong ),
                        &time_start,
                        NULL 
                    );
                    ASSERT_PROF( time_start, errcode_ret );

                    errcode_ret = clGetEventProfilingInfo( timer_stack[segment_end].val.event,
                        CL_PROFILING_COMMAND_END,
                        sizeof( cl_ulong ),
                        &time_end,
                        NULL
                    );
                    ASSERT_PROF( time_end, errcode_ret );

                    LOGV( 
                        T_FORMAT,
                        "SEGMENT",
                        timer_stack[i].name,
                        nano_to_milli(time_end-time_start)
                    );
                }
                else
                {
                    LOGE( "Invalid segment:%s", timer_stack[i].name );
                }
                break;
        }
    }

    struct MapStringDouble *cur = event_sum;;
    double event_sum_total2 = 0;

    while( cur != NULL )
    {
        LOGV( T_FORMAT, "SUMEVENT", cur->name, cur->value );
        event_sum_total2 += cur->value;
        cur = cur->next;
    }

    LOGV( T_FORMAT, "SUMEVENT", "total_sum_in_kernels", event_sum_total2 );
    
    mapstringdouble_clear( event_sum );
    event_sum = NULL;

    LOGV( "__TIMER__ END" );
}

void opencl_timer_clear_events( )
{
    for( int i = 0; i < timer_stack_count; i++ )
    {
        if( timer_stack[i].type == EVENT )
        {
            CL_RELEASE_EVENT(timer_stack[i].val.event);
        } 
    }

    timer_stack_count = 0;
}

#endif
