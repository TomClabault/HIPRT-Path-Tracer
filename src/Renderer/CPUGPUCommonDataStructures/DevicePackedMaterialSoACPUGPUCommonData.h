/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_DEVICE_PACKED_MATERIAL_SOA_CPU_GPU_COMMON_DATA_H
#define RENDERER_DEVICE_PACKED_MATERIAL_SOA_CPU_GPU_COMMON_DATA_H

struct DevicePackedMaterialSoACPUGPUCommonData
{
    /**
     * Takes a pointer to some 'DevicePackedTexturedMaterial' in the 'gpu_packed_materials' array (which could be std::vector().data() for example)
     * and returns a vector of type T that contains 'element_count' elements at offset 'offset' of the 'DevicePackedTexturedMaterial' structure
     *
     * For example:
     * expand_from_gpu_packed_materials<Uint2xPacked>(3, gpu_packed_materials, offsetof(DevicePackedTexturedMaterial, normal_map_emission_index), 2)
     *
     * return an std::vector that contains the 'normal_map_emission_index' of gpu_packed_materials[3] and gpu_packed_materials[4]
     */
    template <typename T>
    std::vector<T> expand_from_gpu_packed_materials(unsigned int start_index, const DevicePackedTexturedMaterial* gpu_packed_materials, size_t offset_in_struct, size_t element_count)
    {
        std::vector<T> out(element_count);

        for (int i = 0; i < element_count; i++)
            out[i] = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(&gpu_packed_materials[start_index + i]) + offset_in_struct);

        return out;
    }
};

#endif