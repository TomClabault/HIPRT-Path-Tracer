/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/kernels/CameraRays.h"
#include "Device/kernels/FullPathTracer.h"
#include "Device/kernels/ReSTIR/DI/LightsPresampling.h"
#include "Device/kernels/ReSTIR/DI/InitialCandidates.h"
#include "Device/kernels/ReSTIR/DI/TemporalReuse.h"
#include "Device/kernels/ReSTIR/DI/SpatialReuse.h"
#include "Device/kernels/ReSTIR/DI/FusedSpatiotemporalReuse.h"

#include "Renderer/Baker/GPUBaker.h"
#include "Renderer/Baker/GPUBakerConstants.h"
#include "Renderer/CPURenderer.h"
#include "Threads/ThreadManager.h"
#include "UI/ApplicationSettings.h"

#include <atomic>
#include <chrono>
#include <omp.h>

 // If 1, only the pixel at DEBUG_PIXEL_X and DEBUG_PIXEL_Y will be rendered,
 // allowing for fast step into that pixel with the debugger to see what's happening.
 // Otherwise if 0, all pixels of the image are rendered
#define DEBUG_PIXEL 1

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
#define DEBUG_PIXEL_X 634
#define DEBUG_PIXEL_Y 441

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
#define DEBUG_NEIGHBORHOOD_SIZE 30

CPURenderer::CPURenderer(int width, int height) : m_resolution(make_int2(width, height))
{
    m_framebuffer = Image32Bit(width, height, 3);

    // Resizing buffers + initial value
    m_pixel_active_buffer.resize(width * height, 0);
    m_denoiser_albedo.resize(width * height, ColorRGB32F(0.0f));
    m_denoiser_normals.resize(width * height, float3{ 0.0f, 0.0f, 0.0f });
    m_pixel_sample_count.resize(width * height, 0);
    m_pixel_converged_sample_count.resize(width * height, 0);
    m_pixel_squared_luminance.resize(width * height, 0.0f);
    m_restir_di_state.initial_candidates_reservoirs.resize(width * height);
    m_restir_di_state.spatial_output_reservoirs_1.resize(width * height);
    m_restir_di_state.spatial_output_reservoirs_2.resize(width * height);
    m_restir_di_state.presampled_lights_buffer.resize(width * height);
    m_restir_di_state.output_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();

    m_g_buffer.resize(width * height);
    m_g_buffer_prev_frame.resize(width * height);

    setup_brdfs_data();

    m_rng = Xorshift32Generator(42);
}

void CPURenderer::setup_brdfs_data()
{
    m_sheen_ltc_params = Image32Bit(reinterpret_cast<float*>(ltc_parameters_table_approximation.data()), 32, 32, 3);
    m_GGX_conductor_Ess = Image32Bit::read_image_hdr("../data/BRDFsData/GGX/" + GPUBakerConstants::get_GGX_conductor_Ess_filename(), 1, true);

    std::vector<Image32Bit> images(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_glossy_dielectric_Ess_filename();
        std::string filepath = "../data/BRDFsData/GlossyDielectrics/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_glossy_dielectrics_Ess = Image32Bit3D(images);

    images.resize(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_Ess_filename();
        std::string filepath = "../data/BRDFsData/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_Ess_glass = Image32Bit3D(images);

    for (int i = 0; i < GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_inv_Ess_filename();
        std::string filepath = "../data/BRDFsData/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_Ess_glass_inverse = Image32Bit3D(images);

    images.resize(GPUBakerConstants::GGX_THIN_GLASS_ESS_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GGX_THIN_GLASS_ESS_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_thin_glass_Ess_filename();
        std::string filepath = "../data/BRDFsData/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_Ess_thin_glass = Image32Bit3D(images);
}

void CPURenderer::set_scene(Scene& parsed_scene)
{
    m_render_data.geom = nullptr;

    m_gpu_packed_materials.resize(parsed_scene.materials.size());
    for (int i = 0; i < parsed_scene.materials.size(); i++)
        m_gpu_packed_materials[i] = parsed_scene.materials[i].pack_to_GPU();

    m_render_data.buffers.materials_buffer = m_gpu_packed_materials.data();
    m_render_data.buffers.material_indices = parsed_scene.material_indices.data();

    // Computing the opaqueness of materials i.e. whether or not they are FULLY opaque
    m_material_opaque.resize(parsed_scene.materials.size());
    for (int i = 0; i < parsed_scene.materials.size(); i++)
        m_material_opaque[i] = parsed_scene.material_has_opaque_base_color_texture[i] && parsed_scene.materials[i].alpha_opacity == 1.0f;

    m_render_data.buffers.material_opaque = m_material_opaque.data();
    m_render_data.buffers.has_vertex_normals = parsed_scene.has_vertex_normals.data();
    m_render_data.buffers.pixels = m_framebuffer.get_data_as_ColorRGB32F();
    m_render_data.buffers.triangles_indices = parsed_scene.triangle_indices.data();
    m_render_data.buffers.vertices_positions = parsed_scene.vertices_positions.data();
    m_render_data.buffers.vertex_normals = parsed_scene.vertex_normals.data();
    m_render_data.buffers.texcoords = parsed_scene.texcoords.data();

    m_render_data.bsdfs_data.sheen_ltc_parameters_texture = &m_sheen_ltc_params;
    m_render_data.bsdfs_data.GGX_conductor_Ess = &m_GGX_conductor_Ess;
    m_render_data.bsdfs_data.glossy_dielectric_Ess = &m_glossy_dielectrics_Ess;
    m_render_data.bsdfs_data.GGX_Ess_glass = &m_GGX_Ess_glass;
    m_render_data.bsdfs_data.GGX_Ess_glass_inverse = &m_GGX_Ess_glass_inverse;
    m_render_data.bsdfs_data.GGX_Ess_thin_glass = &m_GGX_Ess_thin_glass;

    ThreadManager::join_threads(ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY);
    m_render_data.buffers.material_textures = parsed_scene.textures.data();

    m_render_data.aux_buffers.pixel_active = m_pixel_active_buffer.data();
    m_render_data.aux_buffers.denoiser_albedo = m_denoiser_albedo.data();
    m_render_data.aux_buffers.denoiser_normals = m_denoiser_normals.data();
    m_render_data.aux_buffers.pixel_sample_count = m_pixel_sample_count.data();
    m_render_data.aux_buffers.pixel_converged_sample_count = m_pixel_converged_sample_count.data();
    m_render_data.aux_buffers.pixel_squared_luminance = m_pixel_squared_luminance.data();
    m_render_data.aux_buffers.still_one_ray_active = &m_still_one_ray_active;
    m_render_data.aux_buffers.stop_noise_threshold_converged_count = &m_stop_noise_threshold_count;

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

    m_render_data.render_settings.restir_di_settings.light_presampling.light_samples = m_restir_di_state.presampled_lights_buffer.data();
    m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs = m_restir_di_state.initial_candidates_reservoirs.data();
    m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_restir_di_state.spatial_output_reservoirs_1.data();
    m_render_data.aux_buffers.restir_reservoir_buffer_1 = m_restir_di_state.initial_candidates_reservoirs.data();
    m_render_data.aux_buffers.restir_reservoir_buffer_2 = m_restir_di_state.spatial_output_reservoirs_1.data();
    m_render_data.aux_buffers.restir_reservoir_buffer_3 = m_restir_di_state.spatial_output_reservoirs_2.data();
    
    ThreadManager::join_threads(ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
    m_render_data.buffers.emissive_triangles_count = parsed_scene.emissive_triangle_indices.size();
    m_render_data.buffers.emissive_triangles_indices = parsed_scene.emissive_triangle_indices.data();

    std::cout << "Building scene BVH..." << std::endl;
    m_triangle_buffer = parsed_scene.get_triangles();
    m_bvh = std::make_shared<BVH>(&m_triangle_buffer);
    m_render_data.cpu_only.bvh = m_bvh.get();
}

void CPURenderer::set_envmap(Image32Bit& envmap_image)
{
    ThreadManager::join_threads(ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD);

    if (envmap_image.width == 0 || envmap_image.height == 0)
    {
        m_render_data.world_settings.ambient_light_type = AmbientLightType::UNIFORM;
        m_render_data.world_settings.uniform_light_color = ColorRGB32F(0.5f, 0.5f, 0.5f);

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

        envmap_image.compute_alias_table(m_alias_table_probas, m_alias_table_alias, &total_sum);
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
        m_render_data.world_settings.alias_table_probas = m_alias_table_probas.data();
        m_render_data.world_settings.alias_table_alias = m_alias_table_alias.data();
    }
}

void CPURenderer::set_camera(Camera& camera)
{
    m_camera = camera;
    m_render_data.current_camera = camera.to_hiprt();
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

        update(frame_number);
        update_render_data(frame_number);

        camera_rays_pass();
#if DirectLightSamplingStrategy == LSS_RESTIR_DI
        ReSTIR_DI();
#endif
        tracing_pass();

        if (m_render_data.render_settings.accumulate)
            m_render_data.render_settings.sample_number++;
        m_render_data.random_seed = m_rng.xorshift32();
        m_render_data.render_settings.need_to_reset = false;
        // We want the G Buffer of the frame that we just rendered to go in the "g_buffer_prev_frame"
        // and then we can re-use the old buffers of to be filled by the current frame render

        std::cout << "Frame " << frame_number << ": " << frame_number/ static_cast<float>(m_render_data.render_settings.samples_per_frame) * 100.0f << "%" << std::endl;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void CPURenderer::update(int frame_number)
{
    // Resetting the status buffers
    // Uploading false to reset the flag
    *m_render_data.aux_buffers.still_one_ray_active = false;
    // Resetting the counter of pixels converged to 0
    m_render_data.aux_buffers.stop_noise_threshold_converged_count->store(0);

    // Update the camera
    /*if (frame_number == 2)
        m_camera.translate(glm::vec3(0.05, 0.0, 0.0));*/
}

void CPURenderer::update_render_data(int sample)
{
    m_render_data.prev_camera = m_render_data.current_camera;
    m_render_data.current_camera = m_camera.to_hiprt();
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

    // Debugging the chosen pixel
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

void CPURenderer::camera_rays_pass()
{
    debug_render_pass([this](int x, int y) {
        CameraRays(m_render_data, m_resolution, x, y);
    });
}

void CPURenderer::ReSTIR_DI()
{
    launch_ReSTIR_DI_presampling_lights_pass();
    launch_ReSTIR_DI_initial_candidates_pass();

    if (m_render_data.render_settings.restir_di_settings.do_fused_spatiotemporal)
    {
        // If fused-spatiotemporal
        // Also not doing it on the very first frame as we would get no samples through
        configure_ReSTIR_DI_spatiotemporal_pass();
        ReSTIR_DI_spatiotemporal_reuse_pass();
    }
    else
    {
        if (m_render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
        {
            configure_ReSTIR_DI_temporal_pass();
            ReSTIR_DI_temporal_reuse_pass();
        }


        if (m_render_data.render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
        {
            for (int spatial_reuse_pass = 0; spatial_reuse_pass < m_render_data.render_settings.restir_di_settings.spatial_pass.number_of_passes; spatial_reuse_pass++)
            {
                configure_ReSTIR_DI_spatial_pass(spatial_reuse_pass);
                ReSTIR_DI_spatial_reuse_pass();
            }
        }
    }

    configure_ReSTIR_DI_output_buffer();
    m_restir_di_state.odd_frame = !m_restir_di_state.odd_frame;
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



    /**
     * Generic parameters needed by the kernel
     */
    parameters.emissive_triangles_count = m_render_data.buffers.emissive_triangles_count;
    parameters.emissive_triangles_indices = m_render_data.buffers.emissive_triangles_indices;
    parameters.triangles_indices = m_render_data.buffers.triangles_indices;
    parameters.vertices_positions = m_render_data.buffers.vertices_positions;
    parameters.material_indices = m_render_data.buffers.material_indices;
    parameters.materials = m_render_data.buffers.materials_buffer;

    // World settings for sampling the envmap
    parameters.world_settings = m_render_data.world_settings;

    parameters.freeze_random = m_render_data.render_settings.freeze_random;
    parameters.sample_number = m_render_data.render_settings.sample_number;
    parameters.random_seed = m_rng.xorshift32();

    // For each presampled light, the probability that this is going to be an envmap sample
    parameters.envmap_sampling_probability = m_render_data.render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability;

    return parameters;
}

void CPURenderer::launch_ReSTIR_DI_presampling_lights_pass()
{
    if (ReSTIR_DI_DoLightsPresampling == KERNEL_OPTION_TRUE)
    {
        LightPresamplingParameters launch_parameters = configure_ReSTIR_DI_light_presampling_pass();

        for (int index = 0; index < launch_parameters.number_of_subsets * launch_parameters.subset_size; index++)
            ReSTIR_DI_LightsPresampling(launch_parameters, index);
    }
}

void CPURenderer::configure_ReSTIR_DI_initial_pass()
{
    m_render_data.random_seed = m_rng.xorshift32();
    m_render_data.render_settings.restir_di_settings.light_presampling.light_samples = m_restir_di_state.presampled_lights_buffer.data();
    m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs = m_restir_di_state.initial_candidates_reservoirs.data();
}

void CPURenderer::launch_ReSTIR_DI_initial_candidates_pass()
{
    configure_ReSTIR_DI_initial_pass();

    debug_render_pass([this](int x, int y) {
        ReSTIR_DI_InitialCandidates(m_render_data, m_resolution, x, y);
    });
}

void CPURenderer::configure_ReSTIR_DI_temporal_pass()
{
    m_render_data.random_seed = m_rng.xorshift32();
    m_render_data.render_settings.restir_di_settings.temporal_pass.permutation_sampling_random_bits = m_rng.xorshift32();

    // The input of the temporal pass is the output of last frame's
    // ReSTIR (and also the initial candidates but this is implicit
    // and "hardcoded in the shader"
    m_render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs = m_render_data.render_settings.restir_di_settings.restir_output_reservoirs;

    if (m_render_data.render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
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
    m_render_data.random_seed = m_rng.xorshift32();
    m_render_data.render_settings.restir_di_settings.temporal_pass.permutation_sampling_random_bits = m_rng.xorshift32();

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
    m_render_data.random_seed = m_rng.xorshift32();

    if (spatial_pass_index == 0)
    {
        if (m_render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
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
    if (m_render_data.render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
        // If there was spatial reuse, using the output of the spatial reuse pass as the input of the temporal
        // pass of next frame
        m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs;
    else if (m_render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
        // If there was a temporal reuse pass, using that output as the input of the next temporal reuse pass
        m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs;
    else
        // No spatial or temporal, the output of ReSTIR is just the output of the initial candidates pass
        m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;
}

void CPURenderer::ReSTIR_DI_temporal_reuse_pass()
{
    debug_render_pass([this](int x, int y) {
        ReSTIR_DI_TemporalReuse(m_render_data, m_resolution, x, y);
    });
}

void CPURenderer::ReSTIR_DI_spatial_reuse_pass()
{
    debug_render_pass([this](int x, int y) {
        ReSTIR_DI_SpatialReuse(m_render_data, m_resolution, x, y);
    });
}

void CPURenderer::ReSTIR_DI_spatiotemporal_reuse_pass()
{
    debug_render_pass([this](int x, int y) {
        ReSTIR_DI_SpatiotemporalReuse(m_render_data, m_resolution, x, y);
    });
}

void CPURenderer::tracing_pass()
{
    debug_render_pass([this](int x, int y) {
        FullPathTracer(m_render_data, m_resolution, x, y);
    });
}

void CPURenderer::tonemap(float gamma, float exposure)
{
#if DEBUG_PIXEL == 0
#pragma omp parallel for schedule(dynamic)
#endif
    for (int y = 0; y < m_resolution.y; y++)
    {
        for (int x = 0; x < m_resolution.x; x++)
        {
            int index = x + y * m_resolution.x;

            ColorRGB32F hdr_color = m_render_data.buffers.pixels[index];

            if (m_render_data.render_settings.accumulate)
                // Scaling by sample count
                hdr_color = hdr_color / float(m_render_data.render_settings.sample_number);

            ColorRGB32F tone_mapped = ColorRGB32F(1.0f) - exp(-hdr_color * exposure);
            tone_mapped = pow(tone_mapped, 1.0f / gamma);

            m_render_data.buffers.pixels[index] = tone_mapped;
        }
    }
}
