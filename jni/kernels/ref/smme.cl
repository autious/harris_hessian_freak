__kernel void second_moment_matrix_elements( 
    __global hh_float* ddx, 
    __global hh_float* ddy, 
    __global hh_float* xx,
    __global hh_float* xy,
    __global hh_float* yy 
)
{
    int i = get_global_id(0);

    hh_float dx = ddx[i];
    hh_float dy = ddy[i];

    xx[i] = dx*dx;
    xy[i] = dx*dy;
    yy[i] = dy*dy;
}

