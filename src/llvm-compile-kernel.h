/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef LLVM_COMPILE_KERNEL_H
#define LLVM_COMPILE_KERNEL_H

#include <hip/hip_vector_types.h>
#include <hip/hip_runtime.h>
#include "hiprt/impl/Math.h"
#include "hiprt/hiprt_device.h"
#include "hiprt/impl/hiprt_device_impl.h"

#include "Device/kernels/ReSTIR/ReGIR/GridFillTemporalReuse.h"
#include "Device/kernels/ReSTIR/ReGIR/SpatialReuse.h"

__device__ bool filter_function(const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit);

HIPRT_DEVICE bool intersectFunc(uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, const hiprtRay& ray, void* payload, hiprtHit& hit)
{
    const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
    [[maybe_unused]] const void* data = tableHeader.funcDataSets[index].intersectFuncData;
    switch (index)
    {
    default: { return false; }
    }
}

HIPRT_DEVICE bool filterFunc(uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, const hiprtRay& ray, void* payload, const hiprtHit& hit)
{
    const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
    [[maybe_unused]] const void* data = tableHeader.funcDataSets[index].filterFuncData;
    switch (index)
    {
    case 0: { return filter_function(ray, data, payload, hit); }
    default: { return false; }
    }
}

int main()
{
    HIPRTRenderData dummy;

    int number_of_blocks;
    int threads_per_block;

    ReGIR_Grid_Fill_Temporal_Reuse<<<dim3(number_of_blocks), dim3(threads_per_block), 0, hipStreamDefault>>>(dummy);
}

#endif
