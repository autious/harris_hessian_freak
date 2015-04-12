__kernel void harris_corner_response( 
        __global hh_float* xx, 
        __global hh_float* xy, 
        __global hh_float* yy, 
        __global hh_float* output, 
        param_float sigmaD )
{
    int i = get_global_id(0);
    float A = LOAD_HHF(xx,i) * pow(sigmaD, 2);
    float B = LOAD_HHF(yy,i) * pow(sigmaD, 2);
    float C = LOAD_HHF(xy,i) * pow(sigmaD, 2);

    STORE_HHF(output, i, A*B-C*C-CORNER_RESPONSE_ALPHA*pow(A+B,2));
}


__kernel void harris_corner_suppression(
        __global hh_float* in,
        __global hh_float* out,
        int width,
        int height )
{
    int2 c = (int2)(get_global_id(0),get_global_id(1));

    float m = -HUGE_VALF;

    for( int y = -SUP_HALFWIDTH; y <= SUP_HALFWIDTH; y++ )
    {
        for( int x = -SUP_HALFWIDTH; x <= SUP_HALFWIDTH; x++ )
        {
            int2 lc = (int2)(il(x+c.x,0,width),il(y+c.y,0,height));
            m = fmax(m,LOAD_HHF(in,lc.x+lc.y*width));
        }
    }

    float in_value = LOAD_HHF(in,c.x+c.y*width);
    if( m > in_value )
        STORE_HHF(out, c.x+c.y*width, 0.0f);
    else
        STORE_HHF(out, c.x+c.y*width, in_value);
}

__kernel void harris_count( __global hh_float* in, volatile __global uint* strong, volatile __global int* count ) 
{
    int i = get_global_id(0);

    float value = LOAD_HHF(in, i);

    if( value > 0.0f )
    {
        atomic_inc( count );
        if( value > HARRIS_THRESHOLD )
        {
           strong[i] += 1;
        }
    }
}
