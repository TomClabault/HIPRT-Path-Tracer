/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_LIGHT_PRESAMPLING_H
#define DEVICE_KERNELS_REGIR_LIGHT_PRESAMPLING_H

#include "Device/includes/LightSampling/LightUtils.h"

#include "HostDeviceCommon/RenderData.h"

 /**
  * This kernel inserts the keys of the input hash table into the output hash table
  *
  * This is used when the hash table has been resized and we need to re-insert the keys
  * of the old (smaller) hash table into the new (larger) hash table
  */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Light_Presampling(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Light_Presampling(HIPRTRenderData render_data, int thread_index)
#endif
{
    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

#ifdef __KERNELCC__
    const uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

    if (thread_index >= render_data.render_settings.regir_settings.presampled_lights.get_presampled_light_count())
        return;

    Xorshift32Generator rng(wang_hash(thread_index ^ render_data.random_number));

    LightSampleInformation light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, rng);

    ReGIRPresampledLight presampled_light;
    presampled_light.emissive_triangle_index = light_sample.emissive_triangle_index;
    presampled_light.point_on_light = light_sample.point_on_light;
    presampled_light.normal.pack(light_sample.light_source_normal);
    presampled_light.triangle_area = light_sample.light_area;

    render_data.render_settings.regir_settings.presampled_lights.store_one_presampled_light(presampled_light, thread_index);
}

#endif
