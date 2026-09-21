/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NEE_DEFERRED_MIS_CONTEXT_SIZE_H
#define KERNELS_NEE_DEFERRED_MIS_CONTEXT_SIZE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/NEEDeferredMISContext.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) NEEDeferredMISContextSize(size_t* out_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline NEEDeferredMISContextSize(size_t* out_buffer)
#endif
{
	out_buffer[0] = sizeof(NEEDeferredMISContext);
}

#endif // #ifndef KERNELS_NEE_DEFERRED_MIS_CONTEXT_SIZE_H
