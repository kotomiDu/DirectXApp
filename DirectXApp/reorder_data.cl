__kernel void convertRGBAToRGBfloat(__read_only image2d_t inRGBA, __global uint* dstptr)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    unsigned int gid = get_global_id(0);

    const int2 srcCoord = (int2)(x, y);

    const uint4 c = read_imageui(inRGBA, srcCoord);

    vstore4(c, gid, dstptr);
}

