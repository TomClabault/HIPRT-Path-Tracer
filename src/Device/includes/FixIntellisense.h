/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_FIX_INTELLISENSE_H
#define DEVICE_FIX_INTELLISENSE_H

/*
 * All that is in this file is meant to make Visual Studio's intellisense happy
 * in the kernel code so that we have autocompletion and no 
 * red-underlined-stinky-disgusting intellisense error telling us that no, M_PI is not
 * defined (even though it is at compile time for the GPU) etc... blah blah blah
 */

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
#define __shared__

// For using printf in Kernels
#include <stdio.h>
#endif

#endif // FIX_INTELISSENSE_H
