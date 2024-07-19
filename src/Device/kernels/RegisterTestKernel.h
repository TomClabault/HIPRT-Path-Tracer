/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

/**
 * This kernel is a playground for understanding what kind of optimizations the GPU compiler is able to do on variable usage ---> register pressure.
 * 
 * This allows the validation of simple intuitions such as: "The compiler optimizes away unused variables". 
 * 
 * But what if the variable is passed in a function that itself doesn't use it? Try it out: the compiler optimizes it away too.
 * Fun fact: the variable is also optimized if used but not initialized.
 * 
 * You get the idea.
 */

/**
 * Here's a rundown of all that I tested already:
 * 
 * - Unused variable: optimized away, no register cost.
 * - Variable passed to a function that doesn't use it: optimized away, no register cost.
 * - Precomputing a result in a temporary variable to avoid recomputing many times: no register cost. 
 *      This must be because using a temporary variable or not, the result of the calculation must be in
 *      a register anyway so it's only 1 register in both cases
 * - Two different variables equal to the same value: only using one register
 * - Variable declared in a structure. That structure is passed to a function that doesn't use the variable ---> variable optimized away.
 */

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Xorshift.h"

//struct DataStruct
//{
//    unsigned char a, b, c, d;
//};
//
//HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int wang_hash(unsigned int seed)
//{
//    seed = (seed ^ 61) ^ (seed >> 16);
//    seed *= 9;
//    seed = seed ^ (seed >> 4);
//    seed *= 0x27d4eb2d;
//    seed = seed ^ (seed >> 15);
//    return seed;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE int sumFunction(DataStruct& data)
//{
//    return (int)data.a % (int)data.b + (int)data.c + (int)data.d;
//}
//
//#ifdef __KERNELCC__
//GLOBAL_KERNEL_SIGNATURE(void) TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
//#else
//GLOBAL_KERNEL_SIGNATURE(void) inline TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
//#endif
//{
//#ifdef __KERNELCC__
//    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
//    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
//#endif
//    const uint32_t threadId = (x + y * res.x);
//
//    if (threadId >= res.x * res.y)
//        return;
//
//    Xorshift32Generator randomGenerator(wang_hash(threadId + 1));
//
//    DataStruct data;
//
//    unsigned char  rand_a = (unsigned char)randomGenerator.xorshift32();
//    unsigned char  rand_b = (unsigned char)randomGenerator.xorshift32();
//    unsigned char  rand_c = (unsigned char)randomGenerator.xorshift32();
//    unsigned char  rand_d = (unsigned char)randomGenerator.xorshift32();
//
//    data.a = rand_a;
//    data.b = rand_b;
//    data.c = rand_c;
//    data.d = rand_d;
//
//    int result = sumFunction(data);
//
//    render_data.buffers.pixels[threadId] = ColorRGB32F(result);
//}
// 9 registers








//struct DataStruct
//{
//    int packed;
//};
//
//HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int wang_hash(unsigned int seed)
//{
//    seed = (seed ^ 61) ^ (seed >> 16);
//    seed *= 9;
//    seed = seed ^ (seed >> 4);
//    seed *= 0x27d4eb2d;
//    seed = seed ^ (seed >> 15);
//    return seed;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE int sumFunction(DataStruct& data)
//{
//    return  ((data.packed >> 0) | 0b11111111) % // a 
//            ((data.packed >> 8) | 0b11111111) + // b
//            ((data.packed >> 16) | 0b11111111) + // c
//            ((data.packed >> 24) | 0b11111111); // d
//}
//
//#ifdef __KERNELCC__
//GLOBAL_KERNEL_SIGNATURE(void) TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
//#else
//GLOBAL_KERNEL_SIGNATURE(void) inline TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
//#endif
//{
//#ifdef __KERNELCC__
//    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
//    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
//#endif
//    const uint32_t threadId = (x + y * res.x);
//
//    if (threadId >= res.x * res.y)
//        return;
//
//    Xorshift32Generator randomGenerator(wang_hash(threadId + 1));
//
//    DataStruct data;
//
//    data.packed = (int)randomGenerator.xorshift32();
//    int result = sumFunction(data);
//
//    render_data.buffers.pixels[threadId] = ColorRGB32F(result);
//}
// 7 registers









//struct DataStruct
//{
//    short int a;
//    short int b;
//    short int c;
//    short int d;
//};
//
//HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int wang_hash(unsigned int seed)
//{
//    seed = (seed ^ 61) ^ (seed >> 16);
//    seed *= 9;
//    seed = seed ^ (seed >> 4);
//    seed *= 0x27d4eb2d;
//    seed = seed ^ (seed >> 15);
//    return seed;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE int sumFunction(DataStruct& data)
//{
//    return data.a % data.b + data.c + data.d;
//}
//
//#ifdef __KERNELCC__
//GLOBAL_KERNEL_SIGNATURE(void) TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
//#else
//GLOBAL_KERNEL_SIGNATURE(void) inline TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
//#endif
//{
//#ifdef __KERNELCC__
//    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
//    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
//#endif
//    const uint32_t threadId = (x + y * res.x);
//
//    if (threadId >= res.x * res.y)
//        return;
//
//    Xorshift32Generator randomGenerator(wang_hash(threadId + 1));
//
//    DataStruct data;
//    data.a = (short int)randomGenerator.xorshift32();
//    data.b = (short int)randomGenerator.xorshift32();
//    data.c = (short int)randomGenerator.xorshift32();
//    data.d = (short int)randomGenerator.xorshift32();
//    int result = sumFunction(data);
//
//    render_data.buffers.pixels[threadId] = ColorRGB32F(result);
//}
//// 10 registers














//struct DataStruct
//{
//    int packed1;
//    int packed2;
//};
//
//HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int wang_hash(unsigned int seed)
//{
//    seed = (seed ^ 61) ^ (seed >> 16);
//    seed *= 9;
//    seed = seed ^ (seed >> 4);
//    seed *= 0x27d4eb2d;
//    seed = seed ^ (seed >> 15);
//    return seed;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE int sumFunction(DataStruct& data)
//{
//    return  ((data.packed1 >> 0) | 0xFFFF) % ((data.packed1 >> 16) | 0xFFFF) + ((data.packed2 >> 0) | 0xFFFF) + ((data.packed2 >> 16) | 0xFFFF);
//}
//
//#ifdef __KERNELCC__
//GLOBAL_KERNEL_SIGNATURE(void) TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
//#else
//GLOBAL_KERNEL_SIGNATURE(void) inline TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
//#endif
//{
//#ifdef __KERNELCC__
//    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
//    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
//#endif
//    const uint32_t threadId = (x + y * res.x);
//
//    if (threadId >= res.x * res.y)
//        return;
//
//    Xorshift32Generator randomGenerator(wang_hash(threadId + 1));
//
//    DataStruct data;
//    data.packed1 = randomGenerator.xorshift32();
//    data.packed2 = randomGenerator.xorshift32();
//    int result = sumFunction(data);
//
//    render_data.buffers.pixels[threadId] = ColorRGB32F(result);
//}
// 7 registers









//struct DataStruct
//{
//    int a;
//    int b;
//    int c;
//    int d;
//    int e;
//    int f;
//};
//
//HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int wang_hash(unsigned int seed)
//{
//    seed = (seed ^ 61) ^ (seed >> 16);
//    seed *= 9;
//    seed = seed ^ (seed >> 4);
//    seed *= 0x27d4eb2d;
//    seed = seed ^ (seed >> 15);
//    return seed;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE int sumFunction(DataStruct& data)
//{
//    return data.a % data.b + data.c + data.d + data.e + data.f;
//}
//
//#ifdef __KERNELCC__
//GLOBAL_KERNEL_SIGNATURE(void) TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
//#else
//GLOBAL_KERNEL_SIGNATURE(void) inline TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
//#endif
//{
//#ifdef __KERNELCC__
//    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
//    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
//#endif
//    const uint32_t threadId = (x + y * res.x);
//
//    if (threadId >= res.x * res.y)
//        return;
//
//    Xorshift32Generator randomGenerator(wang_hash(threadId + 1));
//
//    DataStruct data;
//    data.a = randomGenerator.xorshift32();
//    data.b = randomGenerator.xorshift32();
//    data.c = randomGenerator.xorshift32();
//    data.d = randomGenerator.xorshift32();
//    data.e = randomGenerator.xorshift32();
//    data.f = randomGenerator.xorshift32();
//    int result = sumFunction(data);
//
//    render_data.buffers.pixels[threadId] = ColorRGB32F(result);
//}
// 10 registers






struct DataStruct
{
    int ab;
    int cd;
    int ef;
};

HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int wang_hash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int sumFunction(DataStruct& data)
{
    int a = data.ab & (0xFFFF << 0);
    int b = data.ab & (0xFFFF << 16);
    int c = data.cd & (0xFFFF << 0);
    int d = data.cd & (0xFFFF << 16);
    int e = data.ef & (0xFFFF << 0);
    int f = data.ef & (0xFFFF << 16);
    return a % b + c + d + e + f;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline TestFunction(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    const uint32_t threadId = (x + y * res.x);

    if (threadId >= res.x * res.y)
        return;

    Xorshift32Generator randomGenerator(wang_hash(threadId + 1));

    DataStruct data;
    data.ab = randomGenerator.xorshift32();
    data.cd = randomGenerator.xorshift32();
    data.ef = randomGenerator.xorshift32();
    int result = sumFunction(data);

    render_data.buffers.pixels[threadId] = ColorRGB32F(result);
}