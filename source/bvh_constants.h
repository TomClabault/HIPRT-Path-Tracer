#ifndef BVH_CONSTANTS_H
#define BVH_CONSTANTS_H

struct BVHConstants
{
    static constexpr int FLATTENED_BVH_MAX_STACK_SIZE = 100000;

    static constexpr int PLANES_COUNT = 7;
    static constexpr int MAX_TRIANGLES_PER_LEAF = 8;
};

#endif
