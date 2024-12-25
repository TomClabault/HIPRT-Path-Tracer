/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RAY_VOLUME_STATE_SIZE_H
#define KERNELS_RAY_VOLUME_STATE_SIZE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/RayVolumeState.h"
#include "HostDeviceCommon/Packing.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) RayVolumeStateSize(size_t* out_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline RayVolumeStateSize(size_t* out_buffer)
#endif
{
	out_buffer[0] = sizeof(RayVolumeState);
}

#endif
