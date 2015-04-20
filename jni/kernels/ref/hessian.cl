__kernel void hessian( __global hh_float *xx, __global hh_float* xy, __global hh_float* yy, __global hh_float* out, param_float sigmaD )
{
    int i = get_global_id(0); 
    out[i] = fabs(((xx[i] * yy[i] - pow(xy[i],2)) * pow(sigmaD,2)));
}

__kernel void find_keypoints( 
        __global hh_float *source_determinants, 
        __global int* corner_counts,
        __global ushort* keypoints, 
        __global int* hessian_determinant_indices,
        int width, 
        int height,
        param_float hessian_determinant_threshold  )
{
    int i = get_global_id(0);

    keypoints[i] = 0;

    if( corner_counts[i] )
    {
        for( int j = 0; j < SCALE_COUNT; j++ )
        {
            hh_float before = 0;
            hh_float after = 0;
            hh_float current = source_determinants[hessian_determinant_indices[j] * width * height + i];

            if( j > 0 )
            {
                before = source_determinants[hessian_determinant_indices[j-1] * width * height + i];
            }

            if( j < SCALE_COUNT - 1 )
            {
                after = source_determinants[hessian_determinant_indices[j+1] * width * height + i];
            }

            if( current > before && current > after && current > hessian_determinant_threshold )
            {
                keypoints[i] |= (1UL << j);
            } 
        }
    }
}
