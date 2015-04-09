__kernel void hessian( __global float *xx, __global float* xy, __global float* yy, __global float* out, float sigmaD )
{
    int i = get_global_id(0); 
    out[i] = fabs(((xx[i] * yy[i] - pow(xy[i],2)) / sigmaD ));
}

__kernel void find_keypoints( 
        __global float *source_determinants, 
        __global int* corner_counts,
        __global ushort* keypoints, 
        __global int* hessian_determinant_indices,
        int width, 
        int height )
{
    int i = get_global_id(0);

    keypoints[i] = 0;

    if( corner_counts[i] )
    {
        for( int j = 0; j < SCALE_COUNT; j++ )
        {
            float before = 0;
            float after = 0;
            float current = source_determinants[hessian_determinant_indices[j] * width * height + i];

            if( j > 0 )
            {
                before = source_determinants[hessian_determinant_indices[j-1] * width * height + i];
            }

            if( j < SCALE_COUNT - 1 )
            {
                after = source_determinants[hessian_determinant_indices[j+1] * width * height + i];
            }

            if( current > before && current > after && current > 10.0f )
            {
                keypoints[i] |= (1UL << j);
            } 
        }
    }
}
