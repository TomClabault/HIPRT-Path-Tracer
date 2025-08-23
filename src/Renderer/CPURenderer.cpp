/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/kernels/CameraRays.h"
#include "Device/kernels/Megakernel.h"
#include "Device/kernels/GMoN/GMoNComputeMedianOfMeans.h"
#include "Device/kernels/NEE++/NEEPlusPlusFinalizeAccumulation.h"

#include "Device/kernels/ReSTIR/ReGIR/ComputeCellsAliasTables.h"
#include "Device/kernels/ReSTIR/ReGIR/GridFillTemporalReuse.h"
#include "Device/kernels/ReSTIR/ReGIR/GridPrepopulate.h"
#include "Device/kernels/ReSTIR/ReGIR/LightPresampling.h"
#include "Device/kernels/ReSTIR/ReGIR/PreIntegration.h"
#include "Device/kernels/ReSTIR/ReGIR/Rehash.h"
#include "Device/kernels/ReSTIR/ReGIR/SpatialReuse.h"
#include "Device/kernels/ReSTIR/ReGIR/SupersamplingCopy.h"

#include "Device/kernels/ReSTIR/DirectionalReuseCompute.h"

#include "Device/kernels/ReSTIR/DI/LightsPresampling.h"
#include "Device/kernels/ReSTIR/DI/InitialCandidates.h"
#include "Device/kernels/ReSTIR/DI/TemporalReuse.h"
#include "Device/kernels/ReSTIR/DI/SpatialReuse.h"
#include "Device/kernels/ReSTIR/DI/FusedSpatiotemporalReuse.h"

#include "Device/kernels/ReSTIR/GI/InitialCandidates.h"
#include "Device/kernels/ReSTIR/GI/SpatialReuse.h"
#include "Device/kernels/ReSTIR/GI/TemporalReuse.h"
#include "Device/kernels/ReSTIR/GI/Shading.h"

#include "Renderer/Baker/GPUBaker.h"
#include "Renderer/Baker/GPUBakerConstants.h"
#include "Renderer/CPURenderer.h"
#include "Threads/ThreadManager.h"
#include "UI/ApplicationSettings.h"

#include <atomic>
#include <chrono>
#include <numeric>
#include <omp.h>

 // If 1, only the pixel at DEBUG_PIXEL_X and DEBUG_PIXEL_Y will be rendered,
 // allowing for fast step into that pixel with the debugger to see what's happening.
 // Otherwise if 0, all pixels of the image are rendered
#define DEBUG_PIXEL 0

// If 0, the pixel with coordinates (x, y) = (0, 0) is top left corner.
// If 1, it's bottom left corner.
// Useful if you're using an image viewer to get the the coordinates of 
// the interesting pixel. If that image viewer has its (0, 0) in the top
// left corner, you'll need to set that DEBUG_FLIP_Y to 0. Set 1 to if
// you're measuring the coordinates of the pixel with (0, 0) in the bottom left corner
#define DEBUG_FLIP_Y 0

// Coordinates of the pixel whose neighborhood needs to rendered (useful for algorithms
// where pixels are not completely independent from each other such as ReSTIR Spatial Reuse).
// 
// The neighborhood around pixel will be rendered if DEBUG_RENDER_NEIGHBORHOOD is 1.
#define DEBUG_PIXEL_X 861
#define DEBUG_PIXEL_Y 545

// Same as DEBUG_FLIP_Y but for the "other debug pixel"
#define DEBUG_OTHER_FLIP_Y 0

// Allows to render the neighborhood around the DEBUG_PIXEL_X/Y but to debug at the location
// of DEBUG_OTHER_PIXEL_X/Y given below.
// 
// -1 to disable. If disabled, the pixel at (DEBUG_PIXEL_X, DEBUG_PIXEL_Y) will be debugged
#define DEBUG_OTHER_PIXEL_X -1
#define DEBUG_OTHER_PIXEL_Y -1

// If 1, a square of DEBUG_NEIGHBORHOOD_SIZE x DEBUG_NEIGHBORHOOD_SIZE pixels
// will be rendered around the pixel to debug (given by DEBUG_PIXEL_X and
// DEBUG_PIXEL_Y). The pixel of interest is going to be rendered first so you
// can just set a breakpoint in the pass of interest and it will break when rendering the
// pixel that you want to debug.
// This can be useful when debugging spatial passes such as ReSTIR spatial reusing.
// If you were only rendering the precise pixel at the given debug coordinates, you
// wouldn't be able to debug correctly since all the neighborhood wouldn't have been
// rendered which means no reservoir which means improper rendering
#define DEBUG_RENDER_NEIGHBORHOOD 1
// How many pixels to render around the debugged pixel given by the DEBUG_PIXEL_X and
// DEBUG_PIXEL_Y coordinates
#define DEBUG_NEIGHBORHOOD_SIZE 50

CPURenderer::CPURenderer(int width, int height) : m_resolution(make_int2(width, height))
{
    m_framebuffer = Image32Bit(width, height, 3);

    m_render_data.render_settings.render_resolution = m_resolution;

    // Resizing buffers + initial value
    m_pixel_active_buffer.resize(width * height, 0);
    m_denoiser_albedo.resize(width * height, ColorRGB32F(0.0f));
    m_denoiser_normals.resize(width * height, float3{ 0.0f, 0.0f, 0.0f });
    m_pixel_sample_count.resize(width * height, 0);
    m_pixel_converged_sample_count.resize(width * height, 0);
    m_pixel_squared_luminance.resize(width * height, 0.0f);


    unsigned int new_cell_count_primary_hits = ReGIRHashGridStorage::DEFAULT_GRID_CELL_COUNT_PRIMARY_HITS;
    unsigned int new_cell_count_secondary_hits = ReGIRHashGridStorage::DEFAULT_GRID_CELL_COUNT_SECONDARY_HITS;

    m_regir_state.presampled_lights.resize(m_render_data.render_settings.regir_settings.presampled_lights.get_presampled_light_count());

    m_regir_state.grid_buffer_primary_hit.resize(new_cell_count_primary_hits, m_render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(true));
    m_regir_state.spatial_grid_buffer_primary_hit.resize(new_cell_count_primary_hits, m_render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(true));
    m_regir_state.hash_cell_data_primary_hit.resize(new_cell_count_primary_hits);

    m_regir_state.grid_buffer_secondary_hit.resize(new_cell_count_secondary_hits, m_render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(false));
    m_regir_state.spatial_grid_buffer_secondary_hit.resize(new_cell_count_secondary_hits, m_render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(false));
    m_regir_state.hash_cell_data_secondary_hit.resize(new_cell_count_secondary_hits);

    m_regir_state.non_canonical_pre_integration_factors_primary_hit = std::vector<AtomicType<float>>(new_cell_count_primary_hits); std::fill(m_regir_state.non_canonical_pre_integration_factors_primary_hit.begin(), m_regir_state.non_canonical_pre_integration_factors_primary_hit.end(), 0.0f);
    m_regir_state.canonical_pre_integration_factors_primary_hit = std::vector<AtomicType<float>>(new_cell_count_primary_hits); std::fill(m_regir_state.canonical_pre_integration_factors_primary_hit.begin(), m_regir_state.canonical_pre_integration_factors_primary_hit.end(), 0.0f);

#if ReGIR_GridFillUsePerCellDistributions == KERNEL_OPTION_TRUE
    m_regir_state.cells_light_distributions_primary_hit.resize(new_cell_count_primary_hits, m_render_data.render_settings.regir_settings.cells_distributions_primary_hits.alias_table_size);
    m_regir_state.cells_light_distributions_secondary_hit.resize(new_cell_count_secondary_hits, m_render_data.render_settings.regir_settings.cells_distributions_secondary_hits.alias_table_size);
#endif

    m_regir_state.non_canonical_pre_integration_factors_secondary_hit = std::vector<AtomicType<float>>(new_cell_count_primary_hits); std::fill(m_regir_state.non_canonical_pre_integration_factors_secondary_hit.begin(), m_regir_state.non_canonical_pre_integration_factors_secondary_hit.end(), 0.0f);
    m_regir_state.canonical_pre_integration_factors_secondary_hit = std::vector<AtomicType<float>>(new_cell_count_primary_hits); std::fill(m_regir_state.canonical_pre_integration_factors_secondary_hit.begin(), m_regir_state.canonical_pre_integration_factors_secondary_hit.end(), 0.0f);

    if (m_render_data.render_settings.regir_settings.supersampling.do_correlation_reduction)
        m_regir_state.correlation_reduction_grid.resize(new_cell_count_primary_hits, m_render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(true) * m_render_data.render_settings.regir_settings.supersampling.correlation_reduction_factor);



    m_restir_di_state.initial_candidates_reservoirs.resize(width * height);
    m_restir_di_state.spatial_output_reservoirs_1.resize(width * height);
    m_restir_di_state.spatial_output_reservoirs_2.resize(width * height);
    m_restir_di_state.presampled_lights_buffer.resize(width * height);
    m_restir_di_state.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
#if ReSTIR_DI_SpatialDirectionalReuseBitCount > 32
    m_restir_di_state.per_pixel_spatial_reuse_directions_mask_ull.resize(width * height);
#else
    m_restir_di_state.per_pixel_spatial_reuse_directions_mask_u.resize(width * height);
#endif
    m_restir_di_state.per_pixel_spatial_reuse_radius.resize(width * height);

    m_restir_gi_state.initial_candidates_reservoirs.resize(width * height);
    m_restir_gi_state.temporal_reservoirs.resize(width * height);
    m_restir_gi_state.spatial_reservoirs.resize(width * height);
#if ReSTIR_GI_SpatialDirectionalReuseBitCount > 32
    m_restir_gi_state.per_pixel_spatial_reuse_directions_mask_ull.resize(width * height);
#else
    m_restir_gi_state.per_pixel_spatial_reuse_directions_mask_u.resize(width * height);
#endif
    m_restir_gi_state.per_pixel_spatial_reuse_radius.resize(width * height);

    m_g_buffer.resize(width * height);
    m_g_buffer_prev_frame.resize(width * height);

    setup_brdfs_data();
    setup_nee_plus_plus();
    setup_gmon();
}

void CPURenderer::setup_brdfs_data()
{
    m_sheen_ltc_params = Image32Bit(reinterpret_cast<float*>(ltc_parameters_table_approximation.data()), 32, 32, 3);
    m_GGX_conductor_directional_albedo = Image32Bit::read_image_hdr("../data/BRDFsData/GGX/" + GPUBakerConstants::get_GGX_conductor_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing), 1, true);

    std::vector<Image32Bit> images(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_glossy_dielectric_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
        std::string filepath = "../data/BRDFsData/GlossyDielectrics/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_glossy_dielectrics_directional_albedo = Image32Bit3D(images);

    images.resize(GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
        std::string filepath = "../data/BRDFsData/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_glass_directional_albedo = Image32Bit3D(images);

    for (int i = 0; i < GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_directional_albedo_inv_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
        std::string filepath = "../data/BRDFsData/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_glass_inverse_directional_albedo = Image32Bit3D(images);

    images.resize(GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_thin_glass_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
        std::string filepath = "../data/BRDFsData/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_thin_glass_directional_albedo = Image32Bit3D(images);
}

void CPURenderer::setup_nee_plus_plus()
{
#if DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE
    // Only doing if using NEE++ 

    m_nee_plus_plus.total_num_rays = std::vector<AtomicType<unsigned int>>(1000000);
    m_nee_plus_plus.total_unoccluded_rays = std::vector<AtomicType<unsigned int>>(1000000);
    m_nee_plus_plus.num_rays_staging = std::vector<AtomicType<unsigned int>>(1000000);
    m_nee_plus_plus.unoccluded_rays_staging = std::vector<AtomicType<unsigned int>>(1000000);
    m_nee_plus_plus.checksum_buffer = std::vector<AtomicType<unsigned int>>(1000000);
    for (AtomicType<unsigned int>& checksum : m_nee_plus_plus.checksum_buffer)
        checksum.store(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);

    m_render_data.nee_plus_plus.m_entries_buffer.total_num_rays = m_nee_plus_plus.total_num_rays.data();
    m_render_data.nee_plus_plus.m_entries_buffer.total_unoccluded_rays = m_nee_plus_plus.total_unoccluded_rays.data();
    //m_render_data.nee_plus_plus.m_entries_buffer.num_rays_staging = m_nee_plus_plus.num_rays_staging.data();
    //m_render_data.nee_plus_plus.m_entries_buffer.unoccluded_rays_staging = m_nee_plus_plus.unoccluded_rays_staging.data();
    m_render_data.nee_plus_plus.m_entries_buffer.checksum_buffer = m_nee_plus_plus.checksum_buffer.data();

    m_render_data.nee_plus_plus.m_total_number_of_cells = 1000000;

    m_render_data.nee_plus_plus.m_total_shadow_ray_queries = &m_nee_plus_plus.total_shadow_ray_queries;
    m_render_data.nee_plus_plus.m_shadow_rays_actually_traced = &m_nee_plus_plus.shadow_rays_actually_traced;
    m_render_data.nee_plus_plus.m_total_cells_alive_count = &m_nee_plus_plus.total_cell_alive_count;
#endif
}

void CPURenderer::setup_gmon()
{
    if (m_render_data.render_settings.samples_per_frame < m_gmon.number_of_sets)
        m_gmon.using_gmon = false;

    if (m_gmon.using_gmon)
    {
        m_gmon.resize(m_resolution.x, m_resolution.y);
        m_render_data.buffers.gmon_estimator.sets = m_gmon.sets.data();
        m_render_data.buffers.gmon_estimator.result_framebuffer = m_gmon.result_framebuffer.get_data_as_ColorRGB32F();
    }
}

void CPURenderer::nee_plus_plus_memcpy_accumulation(int frame_number)
{
#if DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE
    bool enough_frames_passed = frame_number % m_nee_plus_plus.frame_timer_before_visibility_map_update == 0;
    bool not_updating_vis_map_anymore = !m_render_data.nee_plus_plus.m_update_visibility_map;
    if (!enough_frames_passed || not_updating_vis_map_anymore)
        return;

    // Only doing if using NEE++
    for (int x = 0; x < m_render_data.nee_plus_plus.m_total_number_of_cells; x++)
        NEEPlusPlusFinalizeAccumulation(m_render_data.nee_plus_plus, x);
#else
    // Otherwise, it's a no-op
#endif
}

void CPURenderer::gmon_check_for_sets_accumulation()
{
    if (m_gmon.using_gmon)
    {
        m_render_data.buffers.gmon_estimator.next_set_to_accumulate++;

        if (m_render_data.buffers.gmon_estimator.next_set_to_accumulate % m_gmon.number_of_sets == 0)
        {
            // We've added 1 sample to each sets of GMoN so we can compute the median of means
            gmon_compute_median_of_means();

            m_render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;
        }
    }
}

void CPURenderer::ReGIR_post_render_update()
{
#if DirectLightSamplingBaseStrategy != LSS_BASE_REGIR
    return;
#endif

    if (m_render_data.render_settings.regir_settings.supersampling.do_correlation_reduction)
    {
        ReGIRHashGridSoADevice to_copy;
        if (m_render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
            to_copy = m_render_data.render_settings.regir_settings.get_actual_spatial_output_reservoirs_grid(true);
        else
            to_copy = m_render_data.render_settings.regir_settings.get_initial_reservoirs_grid(true);

#pragma omp parallel for
        for (int x = 0; x < *m_render_data.render_settings.regir_settings.get_hash_cell_data_soa(true).grid_cells_alive_count * m_render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(true); x++)
        {
            ReGIR_Supersampling_Copy(m_render_data, to_copy, x);
        }

        m_render_data.render_settings.regir_settings.supersampling.correl_reduction_current_grid++;
        m_render_data.render_settings.regir_settings.supersampling.correl_reduction_current_grid %= m_render_data.render_settings.regir_settings.supersampling.correlation_reduction_factor;

        m_render_data.render_settings.regir_settings.supersampling.correl_frames_available++;
        m_render_data.render_settings.regir_settings.supersampling.correl_frames_available = hippt::min(m_render_data.render_settings.regir_settings.supersampling.correl_frames_available, m_render_data.render_settings.regir_settings.supersampling.correlation_reduction_factor);
    }
}

void CPURenderer::set_scene(Scene& parsed_scene)
{
    m_render_data.GPU_BVH = nullptr;

    std::vector<DevicePackedTexturedMaterial> gpu_packed_materials;
    gpu_packed_materials.resize(parsed_scene.materials.size());
    for (int i = 0; i < parsed_scene.materials.size(); i++)
        gpu_packed_materials[i] = parsed_scene.materials[i].pack_to_GPU();

    m_gpu_packed_materials.upload_data(gpu_packed_materials);
    m_render_data.buffers.materials_buffer = m_gpu_packed_materials.get_device_SoA_struct();
    m_render_data.buffers.material_indices = parsed_scene.material_indices.data();

    // Computing the opaqueness of materials i.e. whether or not they are FULLY opaque
    m_material_opaque.resize(parsed_scene.materials.size());
    for (int i = 0; i < parsed_scene.materials.size(); i++)
        m_material_opaque[i] = parsed_scene.material_has_opaque_base_color_texture[i] && parsed_scene.materials[i].alpha_opacity == 1.0f;

    m_render_data.buffers.material_opaque = m_material_opaque.data();
    m_render_data.buffers.has_vertex_normals = parsed_scene.has_vertex_normals.data();
    m_render_data.buffers.accumulated_ray_colors = m_framebuffer.get_data_as_ColorRGB32F();
    m_render_data.buffers.triangles_indices = parsed_scene.triangles_vertex_indices.data();
    m_render_data.buffers.vertices_positions = parsed_scene.vertices_positions.data();
    m_render_data.buffers.vertex_normals = parsed_scene.vertex_normals.data();
    m_render_data.buffers.texcoords = parsed_scene.texcoords.data();
    m_render_data.buffers.triangles_areas = parsed_scene.triangle_areas.data();

    m_render_data.bsdfs_data.sheen_ltc_parameters_texture = &m_sheen_ltc_params;
    m_render_data.bsdfs_data.GGX_conductor_directional_albedo = &m_GGX_conductor_directional_albedo;
    m_render_data.bsdfs_data.glossy_dielectric_directional_albedo = &m_glossy_dielectrics_directional_albedo;
    m_render_data.bsdfs_data.GGX_glass_directional_albedo = &m_GGX_glass_directional_albedo;
    m_render_data.bsdfs_data.GGX_glass_directional_albedo_inverse = &m_GGX_glass_inverse_directional_albedo;
    m_render_data.bsdfs_data.GGX_thin_glass_directional_albedo = &m_GGX_thin_glass_directional_albedo;

    ThreadManager::join_threads(ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY);
    m_render_data.buffers.material_textures = parsed_scene.textures.data();

    m_render_data.aux_buffers.pixel_active = m_pixel_active_buffer.data();
    m_render_data.aux_buffers.denoiser_albedo = m_denoiser_albedo.data();
    m_render_data.aux_buffers.denoiser_normals = m_denoiser_normals.data();
    m_render_data.aux_buffers.pixel_sample_count = m_pixel_sample_count.data();
    m_render_data.aux_buffers.pixel_converged_sample_count = m_pixel_converged_sample_count.data();
    m_render_data.aux_buffers.pixel_squared_luminance = m_pixel_squared_luminance.data();
    m_render_data.aux_buffers.still_one_ray_active = &m_still_one_ray_active;
    m_render_data.aux_buffers.pixel_count_converged_so_far = &m_stop_noise_threshold_count;

    m_render_data.g_buffer.materials = m_g_buffer.materials.data();
    m_render_data.g_buffer.geometric_normals = m_g_buffer.geometric_normals.data();
    m_render_data.g_buffer.shading_normals = m_g_buffer.shading_normals.data();
    m_render_data.g_buffer.primary_hit_position = m_g_buffer.primary_hit_position.data();
    m_render_data.g_buffer.first_hit_prim_index = m_g_buffer.first_hit_prim_index.data();

    m_render_data.g_buffer_prev_frame.materials = m_g_buffer_prev_frame.materials.data();
    m_render_data.g_buffer_prev_frame.geometric_normals = m_g_buffer_prev_frame.geometric_normals.data();
    m_render_data.g_buffer_prev_frame.shading_normals = m_g_buffer_prev_frame.shading_normals.data();
    m_render_data.g_buffer_prev_frame.primary_hit_position = m_g_buffer_prev_frame.primary_hit_position.data();
    m_render_data.g_buffer_prev_frame.first_hit_prim_index = m_g_buffer_prev_frame.first_hit_prim_index.data();





    m_regir_state.presampled_lights.to_device(m_render_data.render_settings.regir_settings.presampled_lights.presampled_lights_soa);

    m_regir_state.grid_buffer_primary_hit.to_device(m_render_data.render_settings.regir_settings.initial_reservoirs_primary_hits_grid);
    m_regir_state.spatial_grid_buffer_primary_hit.to_device(m_render_data.render_settings.regir_settings.spatial_output_primary_hits_grid);
    m_render_data.render_settings.regir_settings.hash_cell_data_primary_hits = m_regir_state.hash_cell_data_primary_hit.to_device();

    m_regir_state.grid_buffer_secondary_hit.to_device(m_render_data.render_settings.regir_settings.initial_reservoirs_secondary_hits_grid);
    m_regir_state.spatial_grid_buffer_secondary_hit.to_device(m_render_data.render_settings.regir_settings.spatial_output_secondary_hits_grid);
    m_render_data.render_settings.regir_settings.hash_cell_data_secondary_hits = m_regir_state.hash_cell_data_secondary_hit.to_device();

    m_regir_state.correlation_reduction_grid.to_device(m_render_data.render_settings.regir_settings.supersampling.correlation_reduction_grid);

    m_render_data.render_settings.regir_settings.non_canonical_pre_integration_factors_primary_hits = m_regir_state.non_canonical_pre_integration_factors_primary_hit.data();
    m_render_data.render_settings.regir_settings.canonical_pre_integration_factors_primary_hits = m_regir_state.canonical_pre_integration_factors_primary_hit.data();

    m_render_data.render_settings.regir_settings.non_canonical_pre_integration_factors_secondary_hits = m_regir_state.non_canonical_pre_integration_factors_primary_hit.data();
    m_render_data.render_settings.regir_settings.canonical_pre_integration_factors_secondary_hits = m_regir_state.canonical_pre_integration_factors_primary_hit.data();

    m_render_data.render_settings.restir_di_settings.light_presampling.light_samples = m_restir_di_state.presampled_lights_buffer.data();
    m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs = m_restir_di_state.initial_candidates_reservoirs.data();
    m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
    m_render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u = m_restir_di_state.per_pixel_spatial_reuse_directions_mask_u.data();
    m_render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = m_restir_di_state.per_pixel_spatial_reuse_directions_mask_ull.data();
    m_render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_radius = m_restir_di_state.per_pixel_spatial_reuse_radius.data();
    m_render_data.render_settings.restir_di_settings.common_spatial_pass.spatial_reuse_hit_rate_total = &m_restir_di_state.spatial_reuse_hit_rate_total;
    m_render_data.render_settings.restir_di_settings.common_spatial_pass.spatial_reuse_hit_rate_hits = &m_restir_di_state.spatial_reuse_hit_rate_hits;

    m_render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer = m_restir_gi_state.initial_candidates_reservoirs.data();
    m_render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs = m_restir_gi_state.initial_candidates_reservoirs.data();
    m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs = m_restir_gi_state.temporal_reservoirs.data();
    m_render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs = m_restir_gi_state.temporal_reservoirs.data();
    m_render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs = m_restir_gi_state.spatial_reservoirs.data();
    m_render_data.aux_buffers.restir_gi_reservoir_buffer_1 = m_restir_gi_state.initial_candidates_reservoirs.data();
    m_render_data.aux_buffers.restir_gi_reservoir_buffer_2 = m_restir_gi_state.spatial_reservoirs.data();
    m_render_data.aux_buffers.restir_gi_reservoir_buffer_3 = m_restir_gi_state.temporal_reservoirs.data();
    m_render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u = m_restir_gi_state.per_pixel_spatial_reuse_directions_mask_u.data();
    m_render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = m_restir_gi_state.per_pixel_spatial_reuse_directions_mask_ull.data();
    m_render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius = m_restir_gi_state.per_pixel_spatial_reuse_radius.data();
    m_render_data.render_settings.restir_gi_settings.common_spatial_pass.spatial_reuse_hit_rate_total = &m_restir_gi_state.spatial_reuse_hit_rate_total;
    m_render_data.render_settings.restir_gi_settings.common_spatial_pass.spatial_reuse_hit_rate_hits = &m_restir_gi_state.spatial_reuse_hit_rate_hits;

    ThreadManager::join_threads(ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
    m_render_data.buffers.emissive_triangles_count = parsed_scene.emissive_triangles_primitive_indices.size();
    m_render_data.buffers.emissive_triangles_primitive_indices = parsed_scene.emissive_triangles_primitive_indices.data();
    m_render_data.buffers.emissive_triangles_primitive_indices_and_emissive_textures = parsed_scene.emissive_triangles_primitive_indices_and_emissive_textures.data();

    m_emissive_meshes_alias_tables.load_from_emissive_meshes(parsed_scene);
    m_render_data.buffers.emissive_meshes_alias_tables = m_emissive_meshes_alias_tables.to_device();
#if ReGIR_GridFillUsePerCellDistributions == KERNEL_OPTION_TRUE
    m_render_data.render_settings.regir_settings.cells_distributions_primary_hits = m_regir_state.cells_light_distributions_primary_hit.to_device(m_render_data);
    m_render_data.render_settings.regir_settings.cells_distributions_secondary_hits = m_regir_state.cells_light_distributions_secondary_hit.to_device(m_render_data);
#endif

    std::cout << "Building scene's BVH..." << std::endl;
    m_triangle_buffer = parsed_scene.get_triangles(parsed_scene.triangles_vertex_indices);
    m_emissive_triangles_buffer = parsed_scene.get_triangles(parsed_scene.emissive_triangle_vertex_indices);

    m_bvh = std::make_shared<BVH>(&m_triangle_buffer);
    m_light_bvh = std::make_shared<BVH>(&m_emissive_triangles_buffer);


    m_render_data.cpu_only.bvh = m_bvh.get();
    m_render_data.cpu_only.light_bvh = m_light_bvh.get();

#if DirectLightSamplingBaseStrategy == LSS_BASE_POWER || (DirectLightSamplingBaseStrategy == LSS_BASE_REGIR && ReGIR_GridFillLightSamplingBaseStrategy == LSS_BASE_POWER)
    std::cout << "Building scene's power alias table" << std::endl;
    compute_emissives_power_alias_table(parsed_scene);
#endif
}

void CPURenderer::compute_emissives_power_alias_table(const Scene& scene)
{
    ThreadManager::add_dependency(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
    ThreadManager::start_thread(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE, [this, &scene]()
        {
            auto start = std::chrono::high_resolution_clock::now();

            std::vector<float> power_list(scene.emissive_triangles_primitive_indices.size());
            float power_sum = 0.0f;

            for (int i = 0; i < scene.emissive_triangles_primitive_indices.size(); i++)
            {
                int emissive_triangle_index = scene.emissive_triangles_primitive_indices[i];

                // Computing the area of the triangle
                float3 vertex_A = scene.vertices_positions[scene.triangles_vertex_indices[emissive_triangle_index * 3 + 0]];
                float3 vertex_B = scene.vertices_positions[scene.triangles_vertex_indices[emissive_triangle_index * 3 + 1]];
                float3 vertex_C = scene.vertices_positions[scene.triangles_vertex_indices[emissive_triangle_index * 3 + 2]];

                float3 AB = vertex_B - vertex_A;
                float3 AC = vertex_C - vertex_A;

                float3 normal = hippt::cross(AB, AC);
                float length_normal = hippt::length(normal);
                float triangle_area = 0.5f * length_normal;

                int mat_index = scene.material_indices[emissive_triangle_index];
                float emission_luminance = scene.materials[mat_index].emission.luminance() * scene.materials[mat_index].emission_strength * scene.materials[mat_index].global_emissive_factor;

                float area_power = emission_luminance * triangle_area;

                power_list[i] = area_power;
                power_sum += area_power;
            }

            Utils::compute_alias_table(power_list, power_sum, m_power_alias_table_probas, m_power_alias_table_alias);

            m_render_data.buffers.emissive_triangles_power_alias_table.alias_table_alias = m_power_alias_table_alias.data();
            m_render_data.buffers.emissive_triangles_power_alias_table.alias_table_probas = m_power_alias_table_probas.data();
            m_render_data.buffers.emissive_triangles_power_alias_table.sum_elements = power_sum;
            m_render_data.buffers.emissive_triangles_power_alias_table.size = scene.emissive_triangles_primitive_indices.size();

            auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "Power alias table construction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
        });
}

void CPURenderer::set_envmap(Image32Bit& envmap_image)
{
    ThreadManager::join_threads(ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD);

    if (envmap_image.width == 0 || envmap_image.height == 0)
    {
        m_render_data.world_settings.ambient_light_type = AmbientLightType::UNIFORM;

        std::cout << "Empty envmap set on the CPURenderer... Defaulting to uniform ambient light type" << std::endl;

        return;
    }

    if (EnvmapSamplingStrategy == ESS_BINARY_SEARCH)
    {
        m_envmap_cdf = envmap_image.compute_cdf();
        m_render_data.world_settings.envmap_total_sum = m_envmap_cdf.back();
    }
    else if (EnvmapSamplingStrategy == ESS_ALIAS_TABLE)
    {
        float total_sum;

        envmap_image.compute_alias_table(m_envmap_alias_table_probas, m_envmap_alias_table_alias, &total_sum);
        m_render_data.world_settings.envmap_total_sum = total_sum;
    }

    m_packed_envmap.pack_from(envmap_image);
    m_render_data.world_settings.envmap = m_packed_envmap.get_data_pointer();
    m_render_data.world_settings.envmap_width = envmap_image.width;
    m_render_data.world_settings.envmap_height = envmap_image.height;
    m_render_data.world_settings.ambient_light_type = AmbientLightType::ENVMAP;

    if (EnvmapSamplingStrategy == ESS_BINARY_SEARCH)
        m_render_data.world_settings.envmap_cdf = m_envmap_cdf.data();
    else if (EnvmapSamplingStrategy == ESS_ALIAS_TABLE)
    {
        m_render_data.world_settings.envmap_alias_table.alias_table_probas = m_envmap_alias_table_probas.data();
        m_render_data.world_settings.envmap_alias_table.alias_table_alias = m_envmap_alias_table_alias.data();
        m_render_data.world_settings.envmap_alias_table.sum_elements = m_render_data.world_settings.envmap_total_sum;
        m_render_data.world_settings.envmap_alias_table.size = envmap_image.width * envmap_image.height;
    }
}

void CPURenderer::set_camera(Camera& camera)
{
    m_camera = camera;
    m_render_data.current_camera = camera.to_hiprt(m_resolution.x, m_resolution.y);
}

HIPRTRenderData& CPURenderer::get_render_data()
{
    return m_render_data;
}

HIPRTRenderSettings& CPURenderer::get_render_settings()
{
    return m_render_data.render_settings;
}

Image32Bit& CPURenderer::get_framebuffer()
{
    if (m_gmon.using_gmon)
        return m_gmon.result_framebuffer;
    else
        return m_framebuffer;
}

void CPURenderer::render()
{
    std::cout << "CPU rendering..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Using 'samples_per_frame' as the number of samples to render on the CPU
    for (int frame_number = 1; frame_number <= m_render_data.render_settings.samples_per_frame; frame_number++)
    {
        m_render_data.render_settings.do_update_status_buffers = true;

        pre_render_update(frame_number);
        update_render_data(frame_number);

        camera_rays_pass();

#if DirectLightSamplingBaseStrategy == LSS_BASE_REGIR
        ReGIR_pass();
#endif

#if DirectLightSamplingStrategy == LSS_RESTIR_DI
        // Only doing ReSTIR DI is ReSTIR DI is enabled 
        ReSTIR_DI_pass();
#endif

#if PathSamplingStrategy == PSS_BSDF
        tracing_pass();
#elif PathSamplingStrategy == PSS_RESTIR_GI
        ReSTIR_GI_pass();
#endif

        post_sample_update(frame_number);

        std::cout << "Frame " << frame_number << ": " << frame_number / static_cast<float>(m_render_data.render_settings.samples_per_frame) * 100.0f << "%" << std::endl;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void CPURenderer::pre_render_update(int frame_number)
{
    // Resetting the status buffers
    // Uploading false to reset the flag
    *m_render_data.aux_buffers.still_one_ray_active = false;
    // Resetting the counter of pixels converged to 0
    m_render_data.aux_buffers.pixel_count_converged_so_far->store(0);

    if (frame_number > m_render_data.nee_plus_plus.m_stop_update_samples)
        m_render_data.nee_plus_plus.m_update_visibility_map = false;
}

void CPURenderer::post_sample_update(int frame_number)
{
    if (m_render_data.render_settings.accumulate)
        m_render_data.render_settings.sample_number++;
    m_render_data.random_number = m_rng.xorshift32();
    m_render_data.render_settings.need_to_reset = false;
    // We want the G Buffer of the frame that we just rendered to go in the "g_buffer_prev_frame"
    // and then we can re-use the old buffers of to be filled by the current frame render

    nee_plus_plus_memcpy_accumulation(frame_number);
    gmon_check_for_sets_accumulation();
    ReGIR_post_render_update();
}

void CPURenderer::update_render_data(int sample)
{
    m_render_data.prev_camera = m_render_data.current_camera;
    m_render_data.current_camera = m_camera.to_hiprt(m_resolution.x, m_resolution.y);
}

void CPURenderer::reset()
{
    m_render_data.render_settings.need_to_reset = true;
    m_render_data.render_settings.sample_number = 0;
}

void CPURenderer::debug_render_pass(std::function<void(int, int)> render_pass_function)
{
    // Center pixel when rendering a neighborhood
    int center_x = 0;
    int center_y = 0;

    // If we want to debug a pixel that is not the center pixel,
    // the coordinates will be stored there
    int debug_x = -1;
    int debug_y = -1;

#if DEBUG_PIXEL


#if DEBUG_FLIP_Y
    center_x = DEBUG_PIXEL_X;
    center_y = DEBUG_PIXEL_Y;

    debug_x = center_x;
    debug_y = center_y;
#else // DEBUG_FLIP_Y
    center_x = DEBUG_PIXEL_X;
    center_y = m_resolution.y - DEBUG_PIXEL_Y - 1;

    debug_x = center_x;
    debug_y = center_y;
#endif // DEBUG_FLIP_Y


#if DEBUG_OTHER_PIXEL_X != -1 && DEBUG_OTHER_PIXEL_Y != -1
#if DEBUG_OTHER_FLIP_Y
    debug_x = DEBUG_OTHER_PIXEL_X;
    debug_y = DEBUG_OTHER_PIXEL_Y;
#else // DEBUG_OTHER_FLIP_Y
    debug_x = DEBUG_OTHER_PIXEL_X;
    debug_y = m_resolution.y - DEBUG_OTHER_PIXEL_Y - 1;
#endif // DEBUG_OTHER_FLIP_Y
#endif // DEBUG_OTHER_PIXEL_X != -1 && DEBUG_OTHER_PIXEL_Y != -1

    // Debugging the chosen pixel first
    render_pass_function(debug_x, debug_y);

#if DEBUG_RENDER_NEIGHBORHOOD
    // Rendering the neighborhood

#pragma omp parallel for schedule(dynamic)
    for (int render_y = std::max(0, center_y - DEBUG_NEIGHBORHOOD_SIZE); render_y <= std::min(m_resolution.y - 1, center_y + DEBUG_NEIGHBORHOOD_SIZE); render_y++)
    {
        for (int render_x = std::max(0, center_x - DEBUG_NEIGHBORHOOD_SIZE); render_x <= std::min(m_resolution.x - 1, center_x + DEBUG_NEIGHBORHOOD_SIZE); render_x++)
        {
            if (render_x == debug_x && render_y == debug_y)
                // Skipping the pixel that we debugged to avoid rendering it twice
                continue;

            render_pass_function(render_x, render_y);
        }
    }
#endif // DEBUG_RENDER_NEIGHBORHOOD

#else // DEBUG_PIXEL

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
    {
        for (int x = 0; x < m_resolution.x; x++)
        {
            if (x == debug_x && y == debug_y)
                // Skipping the pixel that we debugged to avoid rendering it twice
                continue;

            render_pass_function(x, y);
        }
    }

#endif // DEBUG_PIXEL
}

void CPURenderer::nee_plus_plus_cache_visibility_pass()
{
    //debug_render_pass([this](int x, int y) {
    //    NEEPlusPlusCachingPrepass(m_render_data, /* caching sample count */ 8, x, y);
    //});

    //nee_plus_plus_memcpy_accumulation(/* frame_number */ 0);
}

void CPURenderer::camera_rays_pass()
{
    m_render_data.random_number = m_rng.xorshift32();

    debug_render_pass([this](int x, int y) {
        CameraRays(m_render_data, x, y);
        });
}

void CPURenderer::ReGIR_pass()
{
    if (m_render_data.render_settings.sample_number == 0)
    {
        ReGIR_pre_population();
        ReGIR_compute_cells_light_distributions();

        ReGIR_pre_integration();
    }
    else
        ReGIR_compute_cells_light_distributions();

    ReGIR_presample_lights();

    ReGIR_grid_fill_pass<false>(true);
    ReGIR_grid_fill_pass<false>(false);

    m_render_data.render_settings.regir_settings.actual_spatial_output_buffers_primary_hits = ReGIR_spatial_reuse_pass<false>(true);
    m_render_data.render_settings.regir_settings.actual_spatial_output_buffers_secondary_hits = ReGIR_spatial_reuse_pass<false>(false);
}

void CPURenderer::ReGIR_presample_lights()
{
    for (int index = 0; index < m_render_data.render_settings.regir_settings.presampled_lights.get_presampled_light_count(); index++)
    {
        ReGIR_Light_Presampling(m_render_data, index);
    }
}

template <bool accumulatePreIntegration>
void CPURenderer::ReGIR_grid_fill_pass(bool primary_hit)
{
    m_render_data.random_number = m_rng.xorshift32();

#pragma omp parallel for
    for (int index = 0; index < *m_render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_count * m_render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(primary_hit); index++)
    {
        ReGIR_Grid_Fill_Temporal_Reuse<accumulatePreIntegration>(m_render_data, m_render_data.render_settings.regir_settings.get_initial_reservoirs_grid(primary_hit), *m_render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_count, primary_hit, index);
    }
}

template <bool accumulatePreIntegration>
ReGIRHashGridSoADevice CPURenderer::ReGIR_spatial_reuse_pass(bool primary_hit)
{
    if (!m_render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
        return ReGIRHashGridSoADevice();

    ReGIRHashGridSoADevice input_reservoirs = m_render_data.render_settings.regir_settings.get_initial_reservoirs_grid(primary_hit);
    ReGIRHashGridSoADevice output_reservoirs = m_render_data.render_settings.regir_settings.get_raw_spatial_output_reservoirs_grid(primary_hit);

    for (int i = 0; i < m_render_data.render_settings.regir_settings.spatial_reuse.spatial_reuse_pass_count; i++)
    {
        m_render_data.render_settings.regir_settings.spatial_reuse.spatial_reuse_pass_index = i;

#pragma omp parallel for
        for (int index = 0; index < *m_render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_count * m_render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(primary_hit); index++)
        {
            ReGIR_Spatial_Reuse<accumulatePreIntegration>(m_render_data,
                input_reservoirs,
                output_reservoirs,
                m_render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit),
                *m_render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_count, primary_hit, index);
        }

        std::swap(input_reservoirs, output_reservoirs);
    }

    // Returning the reservoirs into which the spatial reuse pass last output the result
    //
    // This is the 'input' buffer and not 'output' because of the std::swap that happens on the last iteration
    return input_reservoirs;
}

void CPURenderer::ReGIR_pre_population()
{
    debug_render_pass([this](int x, int y) 
    {
        ReGIR_Grid_Prepopulate(m_render_data, x, y);
    });
}

void CPURenderer::ReGIR_pre_integration()
{
    // 2 iterations: 1 for the primary hits, 1 for the secondary hits
    for (int i = 0; i < 2; i++)
    {
        bool primary_hit = (i == 0);

        unsigned int seed_backup = m_render_data.random_number;
        unsigned int nb_cells_alive = *m_render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_count;
        unsigned int nb_threads = nb_cells_alive;

        for (int i = 0; i < m_render_data.render_settings.DEBUG_REGIR_PRE_INTEGRATION_ITERATIONS; i++)
        {
            m_render_data.random_number = m_rng.xorshift32();

            ReGIR_presample_lights();
            ReGIR_grid_fill_pass<true>(primary_hit);
            ReGIR_spatial_reuse_pass<true>(primary_hit);
        }

        m_render_data.random_number = seed_backup;
    }
}

void CPURenderer::ReGIR_compute_cells_light_distributions()
{
#if ReGIR_GridFillUsePerCellDistributions == KERNEL_OPTION_FALSE
    return;
#endif

    ReGIR_compute_cells_light_distributions_internal(true);
    ReGIR_compute_cells_light_distributions_internal(false);
}

void CPURenderer::ReGIR_compute_cells_light_distributions_internal(bool primary_hit)
{
    if (m_render_data.buffers.emissive_meshes_alias_tables.alias_table_count > ReGIR_ComputeCellsLightDistributionsScratchBufferMaxContributionsCount)
    {
        // There are more emissive meshes than the space in our scratch buffer so we're not
        // even going to be able to compute one single alias table, aborting

        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Too many emissive meshes in the scene. ReGIR can't compute per-cell alias tables.");

        return;
    }

    unsigned int nb_cells_alive = primary_hit ? m_regir_state.hash_cell_data_primary_hit.m_grid_cells_alive_count.at(0) : m_regir_state.hash_cell_data_secondary_hit.m_grid_cells_alive_count.at(0);
    if (nb_cells_alive == 0)
        return;

    unsigned int& last_nb_computed_cells_alias_tables = primary_hit ? m_regir_state.m_last_cells_alias_tables_compute_count_primary_hits : m_regir_state.m_last_cells_alias_tables_compute_count_secondary_hits;

    unsigned int total_number_of_cells_to_compute = nb_cells_alive - last_nb_computed_cells_alias_tables;
    if (total_number_of_cells_to_compute == 0)
        return;
    unsigned int emissive_mesh_count = m_render_data.buffers.emissive_meshes_alias_tables.alias_table_count;
    unsigned int max_number_of_cells_computed_per_iteration = std::floor(ReGIR_ComputeCellsLightDistributionsScratchBufferMaxContributionsCount / emissive_mesh_count);

    auto start = std::chrono::high_resolution_clock::now();

    // Allocating the scratch buffer with a maximum size of SCRATCH_BUFFER_MAX_SIZE_BYTES.
    // If we don't need that much size, then we're just allocating what we need (that's the outer min() part)
    //
    // The inner min() part on ReGIR_ComputeCellsLightDistributionsScratchBufferMaxContributionsCount is to round down the buffer on an integer number of
    // cells computed per each iteration. We're not going to compute 2.5 alias table per iteration for example, only 2
    std::vector<float> contribution_scratch_buffer(hippt::min(max_number_of_cells_computed_per_iteration * emissive_mesh_count, total_number_of_cells_to_compute * emissive_mesh_count));

    std::vector<unsigned int> grid_cell_alive_list = primary_hit
        ? m_regir_state.hash_cell_data_primary_hit.m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>()
        : m_regir_state.hash_cell_data_secondary_hit.m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>();

    unsigned int cell_offset = last_nb_computed_cells_alias_tables;
    const unsigned int iteration_needed = std::ceil(total_number_of_cells_to_compute / (float)max_number_of_cells_computed_per_iteration);
    const unsigned int actual_number_of_cells_computed_per_iteration = hippt::min(max_number_of_cells_computed_per_iteration, total_number_of_cells_to_compute);
    for (int iter = 0; iter < iteration_needed; iter++)
    {
        // Computing the contributions of emissive meshes
        size_t contributions_left_to_compute = (total_number_of_cells_to_compute - cell_offset) * emissive_mesh_count;
        unsigned int dispatch_size = hippt::min(contributions_left_to_compute, contribution_scratch_buffer.size());

        auto compute = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int thread_index = 0; thread_index < dispatch_size; thread_index++)
        {
            ReGIR_Compute_Cells_Alias_Tables(m_render_data, contribution_scratch_buffer.data(), cell_offset, primary_hit, thread_index);
        }
        auto stop_compute = std::chrono::high_resolution_clock::now();
        std::cout << "Compute time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_compute - compute).count() << "ms. " << std::endl;




        // Sorting the contributions because we're only going to build the alias table on the best
        // emissives meshes
        //
        // We're actually not going to sort the contributions directly but rather sort the
        // indices that point to the contributions because we're going to need the sorted indices later
        std::vector<unsigned int> sorted_indices(contribution_scratch_buffer.size());

        for (int i = 0; i < actual_number_of_cells_computed_per_iteration; i++)
            std::iota(sorted_indices.begin() + emissive_mesh_count * i, sorted_indices.begin() + emissive_mesh_count * (i + 1), 0); // 0,1,2,...

#pragma omp parallel for
        for (int i = 0; i < actual_number_of_cells_computed_per_iteration; i++)
        {
            auto first = sorted_indices.begin() + emissive_mesh_count * i;
            auto last = sorted_indices.begin() + emissive_mesh_count * (i + 1);

            std::sort(first, last, [&](unsigned int a, unsigned int b)
            {
                // Sorting in descendant order
                return contribution_scratch_buffer[i * emissive_mesh_count + a] > contribution_scratch_buffer[i * emissive_mesh_count + b];
            });
        }

        unsigned int alias_table_size = m_render_data.render_settings.regir_settings.cells_distributions_primary_hits.alias_table_size;
        unsigned int cells_yet_to_compute_count = contributions_left_to_compute / emissive_mesh_count;

        auto upload = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int cell_index_in_iteration = 0; cell_index_in_iteration < hippt::min(actual_number_of_cells_computed_per_iteration, cells_yet_to_compute_count); cell_index_in_iteration++)
        {
            unsigned int hash_grid_cell_index = grid_cell_alive_list[cell_index_in_iteration + cell_offset];
            // Either the alias table size or the number of emissive meshes
            // (number of contributions per cell), whichever is the smallest
            unsigned contribution_count_min = hippt::min(alias_table_size, emissive_mesh_count);

            // We're only going to keep the best 'alias_table_size' contributing meshes
            // in case there are more than that, i.e. the alias table is going to be built only on
            // the 'alias_table_size' meshes that contribute the most to the cell
            float sum_best_contributions = 0.0f;
            std::vector<float> best_contributions(contribution_count_min);
            for (int contribution_index = 0; contribution_index < contribution_count_min; contribution_index++)
            {
                float contribution = contribution_scratch_buffer.at(sorted_indices.at(contribution_index + cell_index_in_iteration * emissive_mesh_count) + cell_index_in_iteration * emissive_mesh_count);

                best_contributions[contribution_index] = contribution;
                sum_best_contributions += contribution;
            }

            ReGIRCellsAliasTablesSoAHost<std::vector>& soa_host = primary_hit ? m_regir_state.cells_light_distributions_primary_hit : m_regir_state.cells_light_distributions_secondary_hit;
            assert(hash_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);

            // Computing the PDFs
            std::vector<float> PDFs(alias_table_size, 0.0f);

            // And computing the alias tables from the contributions
            std::vector<float> probas(alias_table_size, 0.0f);
            std::vector<int> aliases(alias_table_size, 0);
            if (sum_best_contributions > 0.0f)
            {
                for (int pdf_index = 0; pdf_index < contribution_count_min; pdf_index++)
                    PDFs[pdf_index] = best_contributions[pdf_index] / sum_best_contributions;

                Utils::compute_alias_table(best_contributions, sum_best_contributions, probas, aliases);

                soa_host.soa.template upload_to_buffer_partial<ReGIRCellsAliasTablesSoAHostBuffers::REGIR_CELLS_EMISSIVE_MESHES_INDICES>(hash_grid_cell_index * alias_table_size, sorted_indices.begin() + cell_index_in_iteration * emissive_mesh_count, contribution_count_min);
            }

            soa_host.soa.template upload_to_buffer_partial<ReGIRCellsAliasTablesSoAHostBuffers::REGIR_CELLS_ALIAS_TABLES_PROBAS>(hash_grid_cell_index * alias_table_size, probas, contribution_count_min);
            soa_host.soa.template upload_to_buffer_partial<ReGIRCellsAliasTablesSoAHostBuffers::REGIR_CELLS_ALIAS_TABLES_ALIASES>(hash_grid_cell_index * alias_table_size, aliases, contribution_count_min);
            soa_host.soa.template upload_to_buffer_partial<ReGIRCellsAliasTablesSoAHostBuffers::REGIR_CELLS_ALIAS_PDFS>(hash_grid_cell_index * alias_table_size, PDFs, contribution_count_min);
        }
        
        auto stop_upload = std::chrono::high_resolution_clock::now();
        std::cout << "Best contrib / alias table / upload time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_upload - upload).count() << "ms. " << std::endl;

        cell_offset += max_number_of_cells_computed_per_iteration;
    }

    last_nb_computed_cells_alias_tables = nb_cells_alive;

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Distribution compute time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms. ";
    printf("Total contributions [cells x meshes] = [%u * %u = %u]\n", total_number_of_cells_to_compute, emissive_mesh_count, total_number_of_cells_to_compute * emissive_mesh_count);
}

void CPURenderer::ReSTIR_DI_pass()
{
    launch_ReSTIR_DI_presampling_lights_pass();
    launch_ReSTIR_DI_initial_candidates_pass();

    if (m_render_data.render_settings.restir_di_settings.do_fused_spatiotemporal)
        // If fused-spatiotemporal
        // Also not doing it on the very first frame as we would get no samples through
        launch_ReSTIR_DI_spatiotemporal_reuse_pass();
    else
    {
        if (m_render_data.render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
            launch_ReSTIR_DI_temporal_reuse_pass();

        if (m_render_data.render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
            for (int spatial_reuse_pass = 0; spatial_reuse_pass < m_render_data.render_settings.restir_di_settings.common_spatial_pass.number_of_passes; spatial_reuse_pass++)
                launch_ReSTIR_DI_spatial_reuse_pass(spatial_reuse_pass);
    }

    configure_ReSTIR_DI_output_buffer();
    m_restir_di_state.odd_frame = !m_restir_di_state.odd_frame;
}

void CPURenderer::ReSTIR_GI_pass()
{
    compute_ReSTIR_GI_optimal_spatial_reuse_radii();

    configure_ReSTIR_GI_initial_candidates_pass();
    launch_ReSTIR_GI_initial_candidates_pass();

    configure_ReSTIR_GI_temporal_reuse_pass();
    launch_ReSTIR_GI_temporal_reuse_pass();

    for (int i = 0; i < m_render_data.render_settings.restir_gi_settings.common_spatial_pass.number_of_passes; i++)
    {
        configure_ReSTIR_GI_spatial_reuse_pass(i);
        launch_ReSTIR_GI_spatial_reuse_pass();
    }

    configure_ReSTIR_GI_shading_pass();
    launch_ReSTIR_GI_shading_pass();
}

LightPresamplingParameters CPURenderer::configure_ReSTIR_DI_light_presampling_pass()
{
    LightPresamplingParameters parameters;

    /**
     * Parameters specific to the kernel
     */

     // From all the lights of the scene, how many subsets to presample
    parameters.number_of_subsets = m_render_data.render_settings.restir_di_settings.light_presampling.number_of_subsets;
    // How many lights to presample in each subset
    parameters.subset_size = m_render_data.render_settings.restir_di_settings.light_presampling.subset_size;
    // Buffer that holds the presampled lights
    parameters.out_light_samples = m_restir_di_state.presampled_lights_buffer.data();

    // For each presampled light, the probability that this is going to be an envmap sample
    parameters.envmap_sampling_probability = m_render_data.render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability;

    m_render_data.random_number = m_rng.xorshift32();

    return parameters;
}

void CPURenderer::compute_ReSTIR_DI_optimal_spatial_reuse_radii()
{
    m_render_data.random_number = m_rng.xorshift32();

    debug_render_pass([this](int x, int y) {
        ReSTIR_Directional_Reuse_Compute<false>(m_render_data, x, y,
            m_render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u,
            m_render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull,
            m_render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_radius);
        });
}

void CPURenderer::launch_ReSTIR_DI_presampling_lights_pass()
{
    if (ReSTIR_DI_DoLightPresampling == KERNEL_OPTION_TRUE)
    {
        LightPresamplingParameters launch_parameters = configure_ReSTIR_DI_light_presampling_pass();

        for (int index = 0; index < launch_parameters.number_of_subsets * launch_parameters.subset_size; index++)
            ReSTIR_DI_LightsPresampling(launch_parameters, m_render_data, index);
    }
}

void CPURenderer::configure_ReSTIR_DI_initial_pass()
{
    m_render_data.random_number = m_rng.xorshift32();
    m_render_data.render_settings.restir_di_settings.light_presampling.light_samples = m_restir_di_state.presampled_lights_buffer.data();
    m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs = m_restir_di_state.initial_candidates_reservoirs.data();
}

void CPURenderer::launch_ReSTIR_DI_initial_candidates_pass()
{
    configure_ReSTIR_DI_initial_pass();

    debug_render_pass([this](int x, int y) {
        ReSTIR_DI_InitialCandidates(m_render_data, x, y);
        });
}

void CPURenderer::configure_ReSTIR_DI_temporal_pass()
{
    m_render_data.random_number = m_rng.xorshift32();
    m_render_data.render_settings.restir_di_settings.common_temporal_pass.permutation_sampling_random_bits = m_rng.xorshift32();

    // The input of the temporal pass is the output of last frame's
    // ReSTIR (and also the initial candidates but this is implicit
    // and "hardcoded in the shader"
    m_render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs = m_render_data.render_settings.restir_di_settings.restir_output_reservoirs;

    if (m_render_data.render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
        // If we're going to do spatial reuse, reuse the initial
        // candidate reservoirs to store the output of the temporal pass.
        // The spatial reuse pass will read form that buffer.
        // 
        // Reusing the initial candidates buffer (which is an input
        // to the temporal pass) as the output is legal and does not
        // cause a race condition because a given pixel only read and
        // writes to its own pixel in the initial candidates buffer.
        // We're not risking another pixel reading in someone else's
        // pixel in the initial candidates buffer while we write into
        // it (that would be a race condition)
        m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_restir_di_state.initial_candidates_reservoirs.data();
    else
    {
        // Else, no spatial reuse, the output of the temporal pass is going to be in its own buffer.
        // Alternatively using spatial_output_reservoirs_1 and spatial_output_reservoirs_2 to avoid race conditions
        if (m_restir_di_state.odd_frame)
            m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
        else
            m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_2.data();
    }
}

void CPURenderer::configure_ReSTIR_DI_temporal_pass_for_fused_spatiotemporal()
{
    m_render_data.random_number = m_rng.xorshift32();
    m_render_data.render_settings.restir_di_settings.common_temporal_pass.permutation_sampling_random_bits = m_rng.xorshift32();

    // The input of the temporal pass is the output of last frame's
    // ReSTIR (and also the initial candidates but this is implicit
    // and hardcoded in the shader)
    if (m_restir_di_state.odd_frame)
        m_render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
    else
        m_render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs = m_restir_di_state.spatial_output_reservoirs_2.data();

    // Not needed. In the fused spatiotemporal pass, everything is output by the spatial pass
    m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs = nullptr;
}

void CPURenderer::configure_ReSTIR_DI_spatial_pass(int spatial_pass_index)
{
    m_render_data.random_number = m_rng.xorshift32();

    if (spatial_pass_index == 0)
    {
        if (m_render_data.render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
            // For the first spatial reuse pass, we hardcode reading from the output of the temporal pass and storing into 'spatial_output_reservoirs_1'
            m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs;
        else
            // If there is no temporal reuse pass, using the initial candidates as the input to the spatial reuse pass
            m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;

        m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
    }
    else
    {
        // And then, starting at the second spatial reuse pass, we read from the output of the previous spatial pass and store
        // in either spatial_output_reservoirs_1 or spatial_output_reservoirs_2, depending on which one isn't the input (we don't
        // want to store in the same buffers that is used for output because that's a race condition so
        // we're ping-ponging between the two outputs of the spatial reuse pass)

        if ((spatial_pass_index & 1) == 0)
        {
            m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_restir_di_state.spatial_output_reservoirs_2.data();
            m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
        }
        else
        {
            m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
            m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_2.data();

        }
    }
}

void CPURenderer::configure_ReSTIR_DI_spatial_pass_for_fused_spatiotemporal(int spatial_pass_index)
{
    if (spatial_pass_index == 0)
    {
        // The input of the spatial resampling in the fused spatiotemporal pass is the
        // temporal buffer of the last frame i.e. the input to the temporal pass
        //
        // Note, this line of code below assumes that the temporal pass was configured
        // prior to calling this function such that
        // 'm_render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs'
        // is the proper pointer
        m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs;

        if (m_restir_di_state.odd_frame)
            m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_2.data();
        else
            m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
    }
}

void CPURenderer::configure_ReSTIR_DI_spatiotemporal_pass()
{
    // The buffers of the temporal pass are going to be configured in the same way
    configure_ReSTIR_DI_temporal_pass_for_fused_spatiotemporal();

    // But the spatial pass is going to read from the input of the temporal pass i.e. the temporal buffer of the last frame, it's not going to read from the output of the temporal pass
    configure_ReSTIR_DI_spatial_pass_for_fused_spatiotemporal(0);
}

void CPURenderer::configure_ReSTIR_DI_output_buffer()
{
    // Keeping in mind which was the buffer used last for the output of the spatial reuse pass as this is the buffer that
        // we're going to use as the input to the temporal reuse pass of the next frame
    if (m_render_data.render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
        // If there was spatial reuse, using the output of the spatial reuse pass as the input of the temporal
        // pass of next frame
        m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs;
    else if (m_render_data.render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
        // If there was a temporal reuse pass, using that output as the input of the next temporal reuse pass
        m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs;
    else
        // No spatial or temporal, the output of ReSTIR is just the output of the initial candidates pass
        m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;
}

void CPURenderer::launch_ReSTIR_DI_temporal_reuse_pass()
{
    configure_ReSTIR_DI_temporal_pass();

    debug_render_pass([this](int x, int y) {
        ReSTIR_DI_TemporalReuse(m_render_data, x, y);
        });
}

void CPURenderer::launch_ReSTIR_DI_spatial_reuse_pass(int spatial_reuse_pass_index)
{
    configure_ReSTIR_DI_spatial_pass(spatial_reuse_pass_index);

    debug_render_pass([this](int x, int y) {
        ReSTIR_DI_SpatialReuse(m_render_data, x, y);
        });
}

void CPURenderer::launch_ReSTIR_DI_spatiotemporal_reuse_pass()
{
    configure_ReSTIR_DI_spatiotemporal_pass();

    debug_render_pass([this](int x, int y) {
        ReSTIR_DI_SpatiotemporalReuse(m_render_data, x, y);
        });
}

void CPURenderer::tracing_pass()
{
    m_render_data.random_number = m_rng.xorshift32();

    debug_render_pass([this](int x, int y) {
        MegaKernel(m_render_data, x, y);
        });
}

void CPURenderer::compute_ReSTIR_GI_optimal_spatial_reuse_radii()
{
    m_render_data.random_number = m_rng.xorshift32();

    debug_render_pass([this](int x, int y) {
        ReSTIR_Directional_Reuse_Compute<true>(m_render_data, x, y,
            m_render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u,
            m_render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull,
            m_render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius);
        });
}

void CPURenderer::configure_ReSTIR_GI_initial_candidates_pass()
{
    m_render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer = m_restir_gi_state.initial_candidates_reservoirs.data();

    m_render_data.random_number = m_rng.xorshift32();
}

static unsigned int seed;

void CPURenderer::launch_ReSTIR_GI_initial_candidates_pass()
{
    seed = m_render_data.random_number;

    if (m_render_data.render_settings.nb_bounces > 0)
    {
        debug_render_pass([this](int x, int y) {
            ReSTIR_GI_InitialCandidates(m_render_data, x, y);
            });
    }
}

void CPURenderer::configure_ReSTIR_GI_temporal_reuse_pass()
{
    if (m_render_data.render_settings.sample_number == 0)
        // First frame, using the initial candidates as the input
        m_render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs = m_render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer;
    else
        // Not the first frame, the input to the temporal pass is the output of the last frame ReSTIR
        m_render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs = m_render_data.render_settings.restir_gi_settings.restir_output_reservoirs;

    // For the output, using whatever buffer isn't the one we're reading from (the input buffer)
    if (m_render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs == m_restir_gi_state.temporal_reservoirs.data())
        m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs = m_restir_gi_state.spatial_reservoirs.data();
    else
        m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs = m_restir_gi_state.temporal_reservoirs.data();

    m_render_data.random_number = m_rng.xorshift32();
}

void CPURenderer::launch_ReSTIR_GI_temporal_reuse_pass()
{
    if (m_render_data.render_settings.nb_bounces > 0 && m_render_data.render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
    {
        debug_render_pass([this](int x, int y) {
            ReSTIR_GI_TemporalReuse(m_render_data, x, y);
            });
    }
}

void CPURenderer::configure_ReSTIR_GI_spatial_reuse_pass(int spatial_pass_index)
{
    m_render_data.render_settings.restir_gi_settings.common_spatial_pass.spatial_pass_index = spatial_pass_index;

    // The spatial reuse pass spatially reuse on the output of the temporal pass in the 'temporal buffer' and
    // stores in the 'spatial buffer'

    ReSTIRGIReservoir* input_reservoirs;
    ReSTIRGIReservoir* output_reservoirs;

    if (spatial_pass_index > 0)
        // If this is the second spatial reuse pass or more, reading from the output of the previous pass
        input_reservoirs = m_render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs;
    else
    {
        // This is the first spatial reuse pass, reading from the output of the temporal pass
        // or the initial candidates depending on whether or not we have a temporal reuse pass at all

        if (m_render_data.render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
            // and we have a temporal reuse pass so we're going to read from the temporal reservoirs
            input_reservoirs = m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs;
        else
            // and we do not have a temporal reuse pass so we're just going to read from the initial candidates
            input_reservoirs = m_restir_gi_state.initial_candidates_reservoirs.data();
    }

    // Outputting to whichever reservoir we're not reading from to avoid race conditions
    if (input_reservoirs == m_restir_gi_state.temporal_reservoirs.data())
        output_reservoirs = m_restir_gi_state.spatial_reservoirs.data();
    else
        output_reservoirs = m_restir_gi_state.temporal_reservoirs.data();

    m_render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs = input_reservoirs;
    m_render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs = output_reservoirs;

    m_render_data.random_number = m_rng.xorshift32();
}

void CPURenderer::launch_ReSTIR_GI_spatial_reuse_pass()
{
    debug_render_pass([this](int x, int y) {
        ReSTIR_GI_SpatialReuse(m_render_data, x, y);
        });
}

void CPURenderer::configure_ReSTIR_GI_shading_pass()
{
    if (m_render_data.render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass)
        m_render_data.render_settings.restir_gi_settings.restir_output_reservoirs = m_render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs;
    else if (m_render_data.render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
        m_render_data.render_settings.restir_gi_settings.restir_output_reservoirs = m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs;
    else
        m_render_data.render_settings.restir_gi_settings.restir_output_reservoirs = m_render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer;

    m_render_data.random_number = seed;
}

void CPURenderer::launch_ReSTIR_GI_shading_pass()
{
    debug_render_pass([this](int x, int y) {
        ReSTIR_GI_Shading(m_render_data, x, y);
        });
}

void CPURenderer::gmon_compute_median_of_means()
{
    debug_render_pass([this](int x, int y) {
        GMoNComputeMedianOfMeans(m_render_data, x, y);
        });
}

void CPURenderer::tonemap(float gamma, float exposure)
{
    ColorRGB32F* framebuffer_data = get_framebuffer().get_data_as_ColorRGB32F();

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
    {
        for (int x = 0; x < m_resolution.x; x++)
        {
            int index = x + y * m_resolution.x;

            ColorRGB32F hdr_color = framebuffer_data[index];

            if (m_render_data.render_settings.accumulate)
                // Scaling by sample count
                hdr_color = hdr_color / float(m_render_data.render_settings.sample_number);

            ColorRGB32F tone_mapped = ColorRGB32F(1.0f) - exp(-hdr_color * exposure);
            tone_mapped = pow(tone_mapped, 1.0f / gamma);

            framebuffer_data[index] = tone_mapped;
        }
    }
}
