#ifndef HIPRT_FIX_VS_H
#define HIPRT_FIX_VS_H

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define __KERNELCC__
#endif

// The HIPRT_KERNEL_SIGNATURE is only useful to help Visual Studio's Intellisense
// Without this macro, all kernel functions would be declared as:
// extern "C" void __global__ my_function(......)
// but Visual Studio doesn't like the 'extern "C" void __global__' part and it
// breaks code coloration and autocompletion. It is however required for the shader
// compiler
// To circumvent this problem, we're only declaring the functions 'void' when in the text editor
// (when __KERNELCC__ is not defined) and we're correctly declaring the function with the full
// attributes when it's the shader compiler processing the function (when __KERNELCC__ is defined)
// We're also defining blockDim, blockIdx and threadIdx because they are udefined otherwise...
#ifdef __KERNELCC__
#define GLOBAL_KERNEL_SIGNATURE(returnType) extern "C" returnType __global__
#define DEVICE_KERNEL_SIGNATURE(returnType) extern "C" returnType __device__
#else
struct dummyVec3
{
    int x, y, z;
};

static dummyVec3 blockDim, blockIdx, threadIdx;

#define GLOBAL_KERNEL_SIGNATURE(returnType) returnType
#define DEVICE_KERNEL_SIGNATURE(returnType) returnType
#endif

#endif