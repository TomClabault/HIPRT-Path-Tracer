/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SPATIAL_REUSE_H
#define DEVICE_KERNELS_REGIR_SPATIAL_REUSE_H
 
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"
#include "Device/includes/ReSTIR/ReGIR/VisibilityReuse.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE ReGIRReservoir read_full_reservoir(int reservoir_index, 
    Float3xLengthUint10bPacked* emission, int* emissive_triangle_index, float* light_area,
	float3* point_on_light, Octahedral24BitNormal* light_source_normal,
	float* UCW, unsigned char* M)
{
	ReGIRReservoir reservoir;

	reservoir.UCW = UCW[reservoir_index];
	reservoir.M = M[reservoir_index];

	reservoir.sample.emission = emission[reservoir_index];
	reservoir.sample.emissive_triangle_index = emissive_triangle_index[reservoir_index];
	reservoir.sample.light_area = light_area[reservoir_index];
	reservoir.sample.point_on_light = point_on_light[reservoir_index];
	reservoir.sample.light_source_normal = light_source_normal[reservoir_index];

    return reservoir;
}

template <bool includeVisibility, bool withCosineTerm>
HIPRT_HOST_DEVICE float ReGIR_non_shading_evaluate_target_function_local(const HIPRTRenderData& render_data, int linear_cell_index, ColorRGB32F sample_emission, float3 sample_position,
    int representative_prim_index, float3 rep_point, float3 rep_normal, Xorshift32Generator& rng)
{
    float3 to_light_direction = sample_position - rep_point;
    float distance_to_light = hippt::length(to_light_direction);
    to_light_direction /= distance_to_light;

    float target_function = sample_emission.luminance() / hippt::square(distance_to_light);
    if (representative_prim_index != -1 && withCosineTerm)
        // We do have a representative normal, taking the cosine term into account
        target_function *= hippt::max(0.0f, hippt::dot(rep_normal, to_light_direction));

    if constexpr (includeVisibility)
        target_function *= ReGIR_grid_cell_visibility_test(render_data, rep_point, representative_prim_index, sample_position, rng);

    return target_function;
}

HIPRT_HOST_DEVICE void store_reservoir_local(ReGIRReservoir& reservoir, int reservoir_index,
    Float3xLengthUint10bPacked* emission, int* emissive_triangle_index, float* light_area,
    float3* point_on_light, Octahedral24BitNormal* light_source_normal,
    float* UCW, unsigned char* M)
{
	UCW[reservoir_index] = reservoir.UCW;
	M[reservoir_index] = reservoir.M;

	emission[reservoir_index] = reservoir.sample.emission;
	emissive_triangle_index[reservoir_index] = reservoir.sample.emissive_triangle_index;
	light_area[reservoir_index] = reservoir.sample.light_area;
	point_on_light[reservoir_index] = reservoir.sample.point_on_light;
	light_source_normal[reservoir_index] = reservoir.sample.light_source_normal;
}

 /** 
  * This kernel is in charge of the spatial reuse on the ReGIR grid.
  * 
  * Each cell reuses from random cells adjacent to it
  */
 #ifdef __KERNELCC__
 GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Spatial_Reuse(HIPRTRenderData render_data,
     Float3xLengthUint10bPacked* emission, int* emissive_triangle_index,
     float* light_area, float3* point_on_light, Octahedral24BitNormal* light_source_normal,
     float* UCW, unsigned char* M,

     Float3xLengthUint10bPacked* spatial_emission, int* spatial_emissive_triangle_index,
     float* spatial_light_area, float3* spatial_point_on_light, Octahedral24BitNormal* spatial_light_source_normal,
     float* spatial_UCW, unsigned char* spatial_M,

     int* representative_primitive, unsigned int* representative_points, Octahedral24BitNormal* representative_normals)
 #else
 GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Spatial_Reuse(HIPRTRenderData render_data, 
     Float3xLengthUint10bPacked* emission, int* emissive_triangle_index,
     float* light_area, float3* point_on_light, Octahedral24BitNormal* light_source_normal,
     float* UCW, unsigned char* M, 

     Float3xLengthUint10bPacked* spatial_emission, int* spatial_emissive_triangle_index,
     float* spatial_light_area, float3* spatial_point_on_light, Octahedral24BitNormal* spatial_light_source_normal,
     float* spatial_UCW, unsigned char* spatial_M,
     
     int* representative_primitive, unsigned int* representative_points, Octahedral24BitNormal* representative_normals, int reservoir_index)
 #endif
 {
    if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // No initial candidates to sample since no lights
        return;

    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

#ifdef __KERNELCC__
    const uint32_t reservoir_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (reservoir_index >= regir_settings.get_number_of_reservoirs_per_grid())
        return;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(reservoir_index + 1);
    else
        seed = wang_hash((reservoir_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

    Xorshift32Generator random_number_generator(seed);

    // Everyone is going to use the same RNG such that memory accesses on the spatial neighbors are coalesced
    // This is ~2x performance on a 7900XTX
    unsigned int spatial_neighbor_rng_seed = wang_hash((render_data.render_settings.sample_number + 1) * render_data.random_number);
    Xorshift32Generator spatial_neighbor_rng(spatial_neighbor_rng_seed);

    ReGIRReservoir output_reservoir;

    int reservoir_index_in_cell = reservoir_index % regir_settings.grid_fill.get_total_reservoir_count_per_cell();
    int linear_center_cell_index = reservoir_index / regir_settings.grid_fill.get_total_reservoir_count_per_cell();
    int3 xyz_center_cell_index = regir_settings.get_xyz_cell_index_from_linear(linear_center_cell_index);

    int selected = 0;
    for (int neighbor_index = 0; neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_reuse_count + 1; neighbor_index++)
    {
        int3 offset;
        if (neighbor_index == regir_settings.spatial_reuse.spatial_neighbor_reuse_count)
            // The last neighbor reused is the center cell
            offset = make_int3(0, 0, 0);
        else
        {
            float3 offset_float_radius_1 = make_float3(spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f);
            float3 offset_float_radius = offset_float_radius_1 * regir_settings.spatial_reuse.spatial_reuse_radius;

            offset = make_int3(roundf(offset_float_radius.x), roundf(offset_float_radius.y), roundf(offset_float_radius.z));
        }

        int3 neighbor_xyz_cell_index = xyz_center_cell_index + offset;
        int neighbor_linear_cell_index_in_grid = regir_settings.get_linear_cell_index_from_xyz(neighbor_xyz_cell_index);
        if (neighbor_linear_cell_index_in_grid == -1)
            // Neighbor is outside of the grid
            continue;

        // Picking the same reservoir cell-index in the a neighbor cell
        int neighbor_reservoir_linear_index_in_grid = neighbor_linear_cell_index_in_grid * regir_settings.grid_fill.get_total_reservoir_count_per_cell() + reservoir_index_in_cell;

        ReGIRReservoir neighbor_reservoir;
        //if (regir_settings.temporal_reuse.do_temporal_reuse)
        //    // Reading from the output of the temporal reuse
        //    neighbor_reservoir = regir_settings.get_temporal_reservoir(neighbor_reservoir_linear_index_in_grid);
        //else
            // No temporal reuse, reading from the output of the grid fill buffer
            // regir_settings.get_grid_fill_output_reservoir(neighbor_reservoir_linear_index_in_grid);
        neighbor_reservoir = read_full_reservoir(neighbor_reservoir_linear_index_in_grid, emission, emissive_triangle_index, light_area, point_on_light, light_source_normal, UCW, M);

        if (neighbor_reservoir.UCW <= 0.0f)
            continue;

        float mis_weight = 1.0f;
        float target_function_at_center;

        //if (reservoir_index_in_cell < regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell())
        //    target_function_at_center = ReGIR_non_shading_evaluate_target_function<false, true>(render_data, linear_center_cell_index, neighbor_reservoir.sample.emission.unpack(), neighbor_reservoir.sample.point_on_light, random_number_generator);
        //else
        //    // Never using the template visibility/consine terms arguments for canonical reservoirs
        //    target_function_at_center = ReGIR_non_shading_evaluate_target_function<false, false>(render_data, linear_center_cell_index, neighbor_reservoir.sample.emission.unpack(), neighbor_reservoir.sample.point_on_light, random_number_generator);

        {
            int representative_primitive_index = representative_primitive[linear_center_cell_index];

            float3 representative_point;
            unsigned int rep_point_packed = representative_points[linear_center_cell_index];
            if (rep_point_packed == ReGIRRepresentative::UNDEFINED_POINT)
                representative_point = render_data.render_settings.regir_settings.get_cell_center_from_linear_cell_index(linear_center_cell_index);
            else
                representative_point = ReGIR_unpack_representative_point(render_data.render_settings.regir_settings, rep_point_packed, linear_center_cell_index);

            float3 representative_normal = representative_normals[linear_center_cell_index].unpack();

            if (reservoir_index_in_cell < regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell())
                target_function_at_center = ReGIR_non_shading_evaluate_target_function_local<false, true>(render_data, linear_center_cell_index, neighbor_reservoir.sample.emission.unpack(), neighbor_reservoir.sample.point_on_light, representative_primitive_index, representative_normal, representative_point, random_number_generator);
            else
                // Never using the template visibility/consine terms arguments for canonical reservoirs
                target_function_at_center = ReGIR_non_shading_evaluate_target_function_local<false, false>(render_data, linear_center_cell_index, neighbor_reservoir.sample.emission.unpack(), neighbor_reservoir.sample.point_on_light, representative_primitive_index, representative_normal, representative_point, random_number_generator);
        }

        output_reservoir.stream_reservoir(mis_weight, target_function_at_center, neighbor_reservoir, random_number_generator);
    }

    spatial_neighbor_rng.m_state.seed = spatial_neighbor_rng_seed;

    // Now counting the number of neighbors that could have produced this sample for the MIS weight
    // This is 1/Z MIS weights
    float valid_neighbor_count = 0.0f;
    if (output_reservoir.weight_sum > 0.0f)
    {
        for (int neighbor_index = 0; neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_reuse_count + 1; neighbor_index++)
        {
            int3 offset;
            if (neighbor_index == regir_settings.spatial_reuse.spatial_neighbor_reuse_count)
                // The last neighbor reused is the center cell
                offset = make_int3(0, 0, 0);
            else
            {
                float3 offset_float_radius_1 = make_float3(spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f);
                float3 offset_float_radius = offset_float_radius_1 * regir_settings.spatial_reuse.spatial_reuse_radius;

                offset = make_int3(roundf(offset_float_radius.x), roundf(offset_float_radius.y), roundf(offset_float_radius.z));
            }

            int3 neighbor_xyz_cell_index = xyz_center_cell_index + offset;
            int neighbor_linear_cell_index_in_grid = regir_settings.get_linear_cell_index_from_xyz(neighbor_xyz_cell_index);
            if (neighbor_linear_cell_index_in_grid == -1)
                // Neighbor is outside of the grid
                continue;

            int neighbor_reservoir_linear_index_in_grid = neighbor_linear_cell_index_in_grid * regir_settings.grid_fill.get_total_reservoir_count_per_cell() + reservoir_index_in_cell;

            if (reservoir_index_in_cell < regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell())
            {
                int representative_primitive_index = representative_primitive[neighbor_linear_cell_index_in_grid];

                float3 representative_point;
                unsigned int rep_point_packed = representative_points[neighbor_linear_cell_index_in_grid];
                if (rep_point_packed == ReGIRRepresentative::UNDEFINED_POINT)
                    representative_point = render_data.render_settings.regir_settings.get_cell_center_from_linear_cell_index(neighbor_linear_cell_index_in_grid);
                else
                    representative_point = ReGIR_unpack_representative_point(render_data.render_settings.regir_settings, rep_point_packed, neighbor_linear_cell_index_in_grid);

                float3 representative_normal = representative_normals[neighbor_linear_cell_index_in_grid].unpack();

                if (ReGIR_non_shading_evaluate_target_function_local<ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility, ReGIR_GridFillTargetFunctionCosineTerm>(render_data, neighbor_linear_cell_index_in_grid, output_reservoir.sample.emission.unpack(), output_reservoir.sample.point_on_light, representative_primitive_index, representative_point, representative_normal, random_number_generator) > 0.0f)
                    valid_neighbor_count += 1.0f;

                /*if (ReGIR_shading_can_sample_be_produced_by(render_data, output_reservoir.sample, neighbor_linear_cell_index_in_grid, random_number_generator))
                    valid_neighbor_count += 1.0f;*/
            }
            else
                // A canonical reservoir can always be produced by anyone
                valid_neighbor_count += 1.0f;
        }
    }

    // Normalizing the reservoirs to 1
    output_reservoir.M = 1;
    output_reservoir.finalize_resampling(valid_neighbor_count);

    if (reservoir_index_in_cell < regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell())
        // Only visibility-checking non-canonical reservoirs because canonical reservoirs are never visibility-reused so that they stay canonical
        //
        // This visibility check of the reservoirs is needed such that the shading at path tracing time
        // can properly assess whether a given cell could have produced a given sample or not
        output_reservoir = visibility_reuse(render_data, output_reservoir, linear_center_cell_index, random_number_generator);

    store_reservoir_local(output_reservoir, reservoir_index, spatial_emission, spatial_emissive_triangle_index, spatial_light_area, spatial_point_on_light, spatial_light_source_normal, spatial_UCW, spatial_M);
    // regir_settings.spatial_reuse.store_reservoir(output_reservoir, reservoir_index);
}

#endif