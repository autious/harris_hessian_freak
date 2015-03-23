struct KeyPoint
{
    int x;
    int y;
    int scale_id;
};

__kernel void hessian( __global float *xx, __global float* xy, __global float* yy, __global float* out, float sigmaD )
{
    int i = get_global_id(0); 
    out[i] = ( xx[i] * yy[i] - pow(xy[i],2) / sigmaD );
}

__kernel void find_keypoints( 
        __global float *source_determinants, 
        __global int* corner_counts,
        __global struct KeyPoint* keypoints, 
        __global int* keypoint_count,
        int keypoint_limit,
        int width, 
        int height )
{
    int i = get_global_id(0);
    int keypoint_id = 0;

    if( corner_counts[i] )
    {
        for( int j = 0; j < SCALE_COUNT; j++ )
        {
            float before = 0;
            float after = 0;
            float current = source_determinants[(j) * width * height * sizeof( float ) + i];

            if( j > 0 )
            {
                before  = source_determinants[(j-1) * width * height * sizeof( float ) + i];
            }

            if( j < SCALE_COUNT - 1 )
            {
                after   = source_determinants[(j+1) * width * height * sizeof( float ) + i];
            }

            if( current > before && current > after && current > 10.0f )
            {
                keypoint_id = atomic_inc( keypoint_count ); 

                if( keypoint_id < keypoint_limit )
                {
                    struct KeyPoint new_keypoint;
                    
                    new_keypoint.x = i % width;
                    new_keypoint.y = i / width;
                    new_keypoint.scale_id = j;

                    keypoints[keypoint_id] = new_keypoint;
                }
            } 
        }
    }
}
