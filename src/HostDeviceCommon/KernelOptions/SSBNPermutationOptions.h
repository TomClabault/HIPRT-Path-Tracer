/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_SSBN_PERMUTATION_OPTIONS_H
#define HOST_DEVICE_SSBN_PERMUTATION_OPTIONS_H

#ifndef __KERNELCC__

/**
 * Whether or not the implementation of [Distributing Monte Carlo Errors as a Blue Noise in Screen Space by Permuting Pixel Seeds Between Frames, Heitz 2019] is
 * enabled in the render
 */
#define SSBNPermutationEnabled KERNEL_OPTION_FALSE

/**
 * Block size for the sorting pass of the implementation of [Distributing Monte Carlo Errors as a Blue Noise in Screen Space by Permuting Pixel Seeds Between
 * Frames, Heitz 2019]
 */
#define SSBNPermutationBlockSize 32

/**
 * If true, the hash grid will be displayed instead of rendering radiance
 */
#define SSBNPermutationDebugHashGrid KERNEL_OPTION_FALSE

/**
 * If true, the luminance of the random seeds will be displayed to the screen instead of the path tracer radiance
 */
#define SSBNPermutationDebugSeeds KERNEL_OPTION_FALSE

#endif

#endif
