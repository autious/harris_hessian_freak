__kernel void hessian( __global hh_float *xx, __global hh_float* xy, __global hh_float* yy, __global hh_float* out, param_float sigmaD )
{
    int i = get_global_id(0); 

    STORE_HHF(out, i, fabs(((LOAD_HHF(xx, i) * LOAD_HHF(yy, i) - pow( LOAD_HHF(xy, i),2.0f)) / sigmaD)));
}


__kernel void find_keypoints( 
        __global hh_float *source_determinants, 
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
            float current = LOAD_HHF(source_determinants, hessian_determinant_indices[j]*width*height+i);

            if( j > 0 )
            {
                before = LOAD_HHF(source_determinants,hessian_determinant_indices[j-1]*width*height+i);
            }

            if( j < SCALE_COUNT - 1 )
            {
                after = LOAD_HHF(source_determinants,hessian_determinant_indices[j+1]*width*height+i);
            }

            if( current > before && current > after && current > HESSIAN_DETERMINANT_THRESHOLD )
            {
                keypoints[i] |= (1UL << j);
            } 
        }
    }
}
