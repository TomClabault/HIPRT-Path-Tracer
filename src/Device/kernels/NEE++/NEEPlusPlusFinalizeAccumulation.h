/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NEE_PLUS_PLUS_FINALIZE_ACCUMULATION_H
#define KERNELS_NEE_PLUS_PLUS_FINALIZE_ACCUMULATION_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/NEE++/NEE++.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) NEEPlusPlusFinalizeAccumulation(NEEPlusPlusDevice nee_plus_plus_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline NEEPlusPlusFinalizeAccumulation(NEEPlusPlusDevice nee_plus_plus_data, int x)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    uint32_t pixel_index = x;
    if (x >= nee_plus_plus_data.m_total_number_of_cells)
        return;

    // nee_plus_plus_data.copy_accumulation_buffers(pixel_index);
}

#endif
