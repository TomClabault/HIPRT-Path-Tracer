/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef FIX_INTELLISENSE_H
#define FIX_INTELLISENSE_H

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

// This error will only be thrown if this file is compiled by the CPU compiler
// i.e. if this file was included on the CPU side.
// This file should not be included on the CPU because it defines some arbitrary things
// such as M_PI value below
#error "You shall not include hiprt_fix_vs.h on the CPU side you monster! This file should only be included in GPU kernels to fix Visual Studio's intellisense screaming."

// We don't care about the actual value, we just want intellisense to be happy.
// This 3.14f value will never be used because the GPU compiler will not get here
// (we're in the #else of #ifdef __KERNELCC__) and the CPU cannot include this file
#define M_PI 3.14f

struct dummyVec3
{
    int x, y, z;
};

static dummyVec3 blockDim, blockIdx, threadIdx;

#define GLOBAL_KERNEL_SIGNATURE(returnType) returnType
#define DEVICE_KERNEL_SIGNATURE(returnType) returnType
#endif

#endif