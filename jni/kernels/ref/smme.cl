__kernel void second_moment_matrix_elements( 
    __global hh_float* ddx, 
    __global hh_float* ddy, 
    __global hh_float* xx,
    __global hh_float* xy,
    __global hh_float* yy 
)
{
    int i = get_global_id(0);

    float dx = LOAD_HHF(ddx,i);
    float dy = LOAD_HHF(ddy,i);

    STORE_HHF(xx, i, dx*dx);
    STORE_HHF(xy, i, dx*dy);
    STORE_HHF(yy, i, dy*dy);
}

