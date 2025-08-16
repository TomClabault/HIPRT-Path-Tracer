/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_EMISSIVE_MESH_HOST_H
#define RENDERER_EMISSIVE_MESH_HOST_H

#include "HIPRT-Orochi/OrochiBuffer.h"

template <template <typename> typename DataContainer>
struct EmissiveMeshHost
{
    // Alias table built on the power of all the emissive triangles of the mesh
    DataContainer<float> alias_probas;
    DataContainer<int> alias_aliases;

    // Average of all the vertices of the emissive mesh
    float3 average_mesh_point = make_float3(0.0f, 0.0f, 0.0f);

    float total_mesh_emissive_power = 0.0f;
    unsigned int emissive_triangle_count = 0;
};

namespace EmissiveMeshHostHelpers
{
    static EmissiveMeshHost<OrochiBuffer> to_GPU_emissive_mesh_host(const EmissiveMeshHost<std::vector>& CPU_emissive_mesh_host)
    {
        EmissiveMeshHost<OrochiBuffer> out;

        out.alias_probas = OrochiBuffer<float>(CPU_emissive_mesh_host.alias_probas);
        out.alias_aliases = OrochiBuffer<int>(CPU_emissive_mesh_host.alias_aliases);
        out.average_mesh_point = CPU_emissive_mesh_host.average_mesh_point;
        out.total_mesh_emissive_power = CPU_emissive_mesh_host.total_mesh_emissive_power;

        return out;
    }
}

#endif
