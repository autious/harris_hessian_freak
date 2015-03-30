#ifdef PROFILE

#include "opencl_timer.h"

#include <string.h>

#include "opencl_error.h"



#define TIMER_EVENT_NAME_LENGTH 24
static const int TIMER_STACK_SIZE_INCREASE = 256;

bool opencl_timer_enable_profile = false;

enum TimerEventType
{
    EVENT,
    MARKER,
    SEGMENT,
};

struct TimerEvent
{
    char name[TIMER_EVENT_NAME_LENGTH];
    enum TimerEventType type;

    union 
    {
        cl_event event;
        int reoccurance;
        int segment[2];
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

static double nano_to_milli( cl_ulong nano )
{
    return ((double)nano / 1000.0 / 1000.0);
}

void opencl_timer_push_event( const char* name, cl_event event )
{
    resize();

    strncpy( timer_stack[timer_stack_count].name, name, TIMER_EVENT_NAME_LENGTH );
    timer_stack[timer_stack_count].val.event = event;
    timer_stack[timer_stack_count].type = EVENT;
    timer_stack_count++;
}

int opencl_timer_push_marker( const char* name, int reoccurance )
{
    resize();
    
    strncpy( timer_stack[timer_stack_count].name, name, TIMER_EVENT_NAME_LENGTH );
    timer_stack[timer_stack_count].val.reoccurance = reoccurance;
    timer_stack[timer_stack_count].type = MARKER;
    return timer_stack_count++;
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

void opencl_timer_print_results( FILE* f )
{
    const char T_FORMAT[] = "%-10s %24s: %fms\n";
    const char M_FORMAT[] = "%-10s %24s: %d\n";
    cl_ulong time_start, time_end;
    cl_int errcode_ret;

    int segment_start, segment_end;

    double event_sum_total;
    double tmp;

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
                fprintf( 
                    f, 
                    T_FORMAT,
                    "EVENT",
                    timer_stack[i].name,
                    tmp
                );
                event_sum_total += tmp;
                break;

            case MARKER:
                fprintf(
                    f,
                    M_FORMAT,
                    "MARKER",
                    timer_stack[i].name,
                    timer_stack[i].val.reoccurance
                );
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

                    errcode_ret = clGetEventProfilingInfo( timer_stack[i].val.event,
                        CL_PROFILING_COMMAND_END,
                        sizeof( cl_ulong ),
                        &time_end,
                        NULL
                    );
                    ASSERT_PROF( time_end, errcode_ret );

                    fprintf( 
                        f, 
                        T_FORMAT,
                        "SEGMENT",
                        timer_stack[i].name,
                        nano_to_milli(time_end-time_start)
                    );
                }
                else
                {
                }
                break;
        }
        
    }
    fprintf(f, T_FORMAT, "SUMEVENT", "", event_sum_total );
    

     
}

void opencl_timer_clear_events( )
{
    timer_stack_count = 0;
}

#endif
