/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/kernels/CameraRays.h"
#include "Device/kernels/FullPathTracer.h"
#include "Device/kernels/ReSTIR/ReSTIR_DI_InitialCandidates.h"
#include "Device/kernels/ReSTIR/ReSTIR_DI_TemporalReuse.h"
#include "Device/kernels/ReSTIR/ReSTIR_DI_SpatialReuse.h"

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
// Coordinates of the pixel to render
#define DEBUG_PIXEL_X 881
#define DEBUG_PIXEL_Y 233
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

    // Resizing buffers + initial value
    m_pixel_active_buffer.resize(width * height, 0);
    m_denoiser_albedo.resize(width * height, ColorRGB32F(0.0f));
    m_denoiser_normals.resize(width * height, float3{ 0.0f, 0.0f, 0.0f });
    m_pixel_sample_count.resize(width * height, 0);
    m_pixel_squared_luminance.resize(width * height, 0.0f);
    m_restir_initial_candidates_reservoirs.resize(width * height);
    m_restir_spatial_output_reservoirs_1.resize(width * height);
    m_restir_spatial_output_reservoirs_2.resize(width * height);
    m_last_spatial_output_reservoirs = m_restir_spatial_output_reservoirs_1.data();

    m_g_buffer.materials.resize(width * height);
    m_g_buffer.geometric_normals.resize(width * height);
    m_g_buffer.shading_normals.resize(width * height);
    m_g_buffer.view_directions.resize(width * height);
    m_g_buffer.first_hits.resize(width * height);
    m_g_buffer.cameray_ray_hit.resize(width * height);
    m_g_buffer.ray_volume_states.resize(width * height);

    m_rng = Xorshift32Generator(42);
}

void CPURenderer::set_scene(Scene& parsed_scene)
{
    m_render_data.geom = nullptr;

    m_render_data.buffers.emissive_triangles_count = parsed_scene.emissive_triangle_indices.size();
    m_render_data.buffers.emissive_triangles_indices = parsed_scene.emissive_triangle_indices.data();
    m_render_data.buffers.materials_buffer = parsed_scene.materials.data();
    m_render_data.buffers.material_indices = parsed_scene.material_indices.data();
    m_render_data.buffers.has_vertex_normals = parsed_scene.has_vertex_normals.data();
    m_render_data.buffers.pixels = m_framebuffer.get_data_as_ColorRGB32F();
    m_render_data.buffers.triangles_indices = parsed_scene.triangle_indices.data();
    m_render_data.buffers.vertices_positions = parsed_scene.vertices_positions.data();
    m_render_data.buffers.vertex_normals = parsed_scene.vertex_normals.data();
    m_render_data.buffers.texcoords = parsed_scene.texcoords.data();

    m_render_data.buffers.material_textures = parsed_scene.textures.data();
    m_render_data.buffers.textures_dims = parsed_scene.textures_dims.data();

    m_render_data.aux_buffers.pixel_active = m_pixel_active_buffer.data();
    m_render_data.aux_buffers.denoiser_albedo = m_denoiser_albedo.data();
    m_render_data.aux_buffers.denoiser_normals = m_denoiser_normals.data();
    m_render_data.aux_buffers.pixel_sample_count = m_pixel_sample_count.data();
    m_render_data.aux_buffers.pixel_squared_luminance = m_pixel_squared_luminance.data();
    m_render_data.aux_buffers.still_one_ray_active = &m_still_one_ray_active;
    m_render_data.aux_buffers.stop_noise_threshold_converged_count = &m_stop_noise_threshold_count;

    m_render_data.g_buffer.materials= m_g_buffer.materials.data();
    m_render_data.g_buffer.geometric_normals = m_g_buffer.geometric_normals.data();
    m_render_data.g_buffer.shading_normals = m_g_buffer.shading_normals.data();
    m_render_data.g_buffer.view_directions = m_g_buffer.view_directions.data();
    m_render_data.g_buffer.first_hits = m_g_buffer.first_hits.data();
    m_render_data.g_buffer.camera_ray_hit = m_g_buffer.cameray_ray_hit.data();
    m_render_data.g_buffer.ray_volume_states = m_g_buffer.ray_volume_states.data();

    m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs = m_restir_initial_candidates_reservoirs.data();
    m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_restir_spatial_output_reservoirs_1.data();
    m_render_data.aux_buffers.restir_reservoir_buffer_1 = m_restir_initial_candidates_reservoirs.data();
    m_render_data.aux_buffers.restir_reservoir_buffer_2 = m_restir_spatial_output_reservoirs_1.data();
    m_render_data.aux_buffers.restir_reservoir_buffer_3 = m_restir_spatial_output_reservoirs_2.data();

    std::cout << "Building scene BVH..." << std::endl;
    m_triangle_buffer = parsed_scene.get_triangles();
    m_bvh = std::make_shared<BVH>(&m_triangle_buffer);
    m_render_data.cpu_only.bvh = m_bvh.get();
}

void CPURenderer::set_envmap(Image32Bit& envmap_image)
{
    ThreadManager::join_threads(ThreadManager::ENVMAP_LOAD_THREAD_KEY);

    if (envmap_image.width == 0 || envmap_image.height == 0)
    {
        m_render_data.world_settings.ambient_light_type = AmbientLightType::UNIFORM;
        m_render_data.world_settings.uniform_light_color = ColorRGB32F(1.0f, 1.0f, 1.0f);

        return;
    }

    m_render_data.world_settings.envmap = &envmap_image;
    m_render_data.world_settings.envmap_width = envmap_image.width;
    m_render_data.world_settings.envmap_height = envmap_image.height;
    m_render_data.world_settings.envmap_cdf = envmap_image.get_cdf().data();
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
    m_render_data.render_settings.sample_number = 1;
    for (int frame_number = 0; frame_number < m_render_data.render_settings.samples_per_frame; frame_number++)
    {
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

        if (m_render_data.render_settings.samples_per_frame > 1)
            std::cout << "Frame " << frame_number + 1 << ": " << (frame_number + 1) / static_cast<float>(m_render_data.render_settings.samples_per_frame) * 100.0f << "%" << std::endl;

        
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void CPURenderer::update(int frame_number)
{
    if (frame_number == 8)
        m_camera.translate(glm::vec3(0, +0.1, 0));
}

void CPURenderer::update_render_data(int sample)
{
    m_render_data.prev_camera = m_render_data.current_camera;
    m_render_data.current_camera = m_camera.to_hiprt();
}

void CPURenderer::camera_rays_pass()
{
#if DEBUG_PIXEL
    int x, y;
#if DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = DEBUG_PIXEL_Y;
#else // DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = m_resolution.y - DEBUG_PIXEL_Y - 1;
#endif // DEBUG_FLIP_Y

    CameraRays(m_render_data, m_resolution, x, y);

#if DEBUG_RENDER_NEIGHBORHOOD
    // Rendering the neighborhood

#pragma omp parallel for schedule(dynamic)
    for (int render_y = std::max(0, y - DEBUG_NEIGHBORHOOD_SIZE); render_y <= std::min(m_resolution.y - 1, y + DEBUG_NEIGHBORHOOD_SIZE); render_y++)
        for (int render_x = std::max(0, x - DEBUG_NEIGHBORHOOD_SIZE); render_x <= std::min(m_resolution.x - 1, x + DEBUG_NEIGHBORHOOD_SIZE); render_x++)
            CameraRays(m_render_data, m_resolution, render_x, render_y);
#endif // DEBUG_RENDER_NEIGHBORHOOD
#else // DEBUG_PIXEL
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
        for (int x = 0; x < m_resolution.x; x++)
            CameraRays(m_render_data, m_resolution, x, y);
#endif // DEBUG_PIXEL
}

void CPURenderer::ReSTIR_DI()
{
    m_render_data.random_seed = m_rng.xorshift32();
    ReSTIR_DI_initial_candidates_pass();

    ReSTIRDISettings& restir_di_settings = m_render_data.render_settings.restir_di_settings;
    if (m_render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
    {
        m_render_data.random_seed = m_rng.xorshift32();

        // The input of the temporal pass is the output of last frame ReSTIR (and also the initial candidates but this is implicit
        // and "hardcoded in the shader"
        restir_di_settings.temporal_pass.input_reservoirs = restir_di_settings.restir_output_reservoirs;

        if (restir_di_settings.spatial_pass.do_spatial_reuse_pass)
            // If we're going to do spatial reuse, reuse the initial candidate reservoirs to store the output of the temporal pass.
            // The spatial reuse pass will read form that buffer
            restir_di_settings.temporal_pass.output_reservoirs = m_restir_initial_candidates_reservoirs.data();
        else
            // Else, no spatial reuse, the output of the temporal pass is going to be in its own buffer, using spatial_reuse_output1
            // here arbitrarily, could have been spatial reuse 2, doesn't matter, we just need a buffer
            restir_di_settings.temporal_pass.output_reservoirs = m_restir_spatial_output_reservoirs_1.data();
        ReSTIR_DI_temporal_candidates_pass();
    }

    if (restir_di_settings.spatial_pass.do_spatial_reuse_pass)
    {
        for (int spatial_reuse_pass = 0; spatial_reuse_pass < restir_di_settings.spatial_pass.number_of_passes; spatial_reuse_pass++)
        {
            m_render_data.random_seed = m_rng.xorshift32();

            if (spatial_reuse_pass == 0)
            {
                if (restir_di_settings.temporal_pass.do_temporal_reuse_pass)
                    // For the first spatial reuse pass, we hardcode reading from the output of the temporal pass and storing into 'spatial_reuse_output_1'
                    restir_di_settings.spatial_pass.input_reservoirs = restir_di_settings.temporal_pass.output_reservoirs;
                else
                    // If there is no temporal reuse pass, using the initial candidates as the input to the spatial reuse pass
                    restir_di_settings.spatial_pass.input_reservoirs = restir_di_settings.initial_candidates.output_reservoirs;

                restir_di_settings.spatial_pass.output_reservoirs = m_restir_spatial_output_reservoirs_1.data();
            }
            else
            {
                // And then, starting at the second spatial reuse pass, we read from the output of the previous spatial pass and store
                // in either spatial_reuse_output_1 or spatial_reuse_output_2, depending on which one isn't the input (we don't
                // want to store in the same buffers that is used for output because that's a race condition so
                // we're ping-ponging between the two outputs of the spatial reuse pass)

                if ((spatial_reuse_pass & 1) == 0)
                {
                    restir_di_settings.spatial_pass.input_reservoirs = m_restir_spatial_output_reservoirs_2.data();
                    restir_di_settings.spatial_pass.output_reservoirs = m_restir_spatial_output_reservoirs_1.data();
                }
                else
                {
                    restir_di_settings.spatial_pass.input_reservoirs = m_restir_spatial_output_reservoirs_1.data();
                    restir_di_settings.spatial_pass.output_reservoirs = m_restir_spatial_output_reservoirs_2.data();

                }
            }
        }

        ReSTIR_DI_spatial_reuse_pass();
}

    // Keeping in mind which was the buffer used last for the output of the spatial reuse pass as this is the buffer that
    // we're going to use as the input to the temporal reuse pass of the next frame
    if (restir_di_settings.spatial_pass.do_spatial_reuse_pass)
        // If there was spatial reuse, using the output of the spatial reuse pass as the input of the temporal
        // pass of next frame
        restir_di_settings.restir_output_reservoirs = restir_di_settings.spatial_pass.output_reservoirs;
    else if (restir_di_settings.temporal_pass.do_temporal_reuse_pass)
        // If there was a temporal reuse pass, using that output as the input of the next temporal reuse pass
        restir_di_settings.restir_output_reservoirs = restir_di_settings.temporal_pass.output_reservoirs;
    else
        // No spatial or temporal, the output of ReSTIR is just the output of the initial candidates pass
        restir_di_settings.restir_output_reservoirs = restir_di_settings.initial_candidates.output_reservoirs;
}

void CPURenderer::ReSTIR_DI_initial_candidates_pass()
{
#if DEBUG_PIXEL
    int x, y;
#if DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = DEBUG_PIXEL_Y;
#else // DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = m_resolution.y - DEBUG_PIXEL_Y - 1;
#endif // DEBUG_FLIP_Y

    ReSTIR_DI_InitialCandidates(m_render_data, m_resolution, x, y);

#if DEBUG_RENDER_NEIGHBORHOOD
    // Rendering the neighborhood

#pragma omp parallel for schedule(dynamic)
    for (int render_y = std::max(0, y - DEBUG_NEIGHBORHOOD_SIZE); render_y <= std::min(m_resolution.y - 1, y + DEBUG_NEIGHBORHOOD_SIZE); render_y++)
        for (int render_x = std::max(0, x - DEBUG_NEIGHBORHOOD_SIZE); render_x <= std::min(m_resolution.x - 1, x + DEBUG_NEIGHBORHOOD_SIZE); render_x++)
            ReSTIR_DI_InitialCandidates(m_render_data, m_resolution, render_x, render_y);
#endif // DEBUG_RENDER_NEIGHBORHOOD
#else // DEBUG_PIXEL
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
        for (int x = 0; x < m_resolution.x; x++)
            ReSTIR_DI_InitialCandidates(m_render_data, m_resolution, x, y);
#endif // DEBUG_PIXEL
}

void CPURenderer::ReSTIR_DI_temporal_candidates_pass()
{
#if DEBUG_PIXEL
    int x, y;
#if DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = DEBUG_PIXEL_Y;
#else // DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = m_resolution.y - DEBUG_PIXEL_Y - 1;
#endif // DEBUG_FLIP_Y

    ReSTIR_DI_TemporalReuse(m_render_data, m_resolution, x, y);

#if DEBUG_RENDER_NEIGHBORHOOD
    // Rendering the neighborhood

#pragma omp parallel for schedule(dynamic)
    for (int render_y = std::max(0, y - DEBUG_NEIGHBORHOOD_SIZE); render_y <= std::min(m_resolution.y - 1, y + DEBUG_NEIGHBORHOOD_SIZE); render_y++)
        for (int render_x = std::max(0, x - DEBUG_NEIGHBORHOOD_SIZE); render_x <= std::min(m_resolution.x - 1, x + DEBUG_NEIGHBORHOOD_SIZE); render_x++)
            ReSTIR_DI_TemporalReuse(m_render_data, m_resolution, render_x, render_y);
#endif // DEBUG_RENDER_NEIGHBORHOOD
#else // DEBUG_PIXEL
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
        for (int x = 0; x < m_resolution.x; x++)
            ReSTIR_DI_TemporalReuse(m_render_data, m_resolution, x, y);
#endif // DEBUG_PIXEL
}

void CPURenderer::ReSTIR_DI_spatial_reuse_pass()
{
#if DEBUG_PIXEL
    int x, y;
#if DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = DEBUG_PIXEL_Y;
#else // DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = m_resolution.y - DEBUG_PIXEL_Y - 1;
#endif // DEBUG_FLIP_Y

    ReSTIR_DI_SpatialReuse(m_render_data, m_resolution, x, y);

#if DEBUG_RENDER_NEIGHBORHOOD
    // Rendering the neighborhood

#pragma omp parallel for schedule(dynamic)
    for (int render_y = std::max(0, y - DEBUG_NEIGHBORHOOD_SIZE); render_y <= std::min(m_resolution.y - 1, y + DEBUG_NEIGHBORHOOD_SIZE); render_y++)
        for (int render_x = std::max(0, x - DEBUG_NEIGHBORHOOD_SIZE); render_x <= std::min(m_resolution.x - 1, x + DEBUG_NEIGHBORHOOD_SIZE); render_x++)
            ReSTIR_DI_SpatialReuse(m_render_data, m_resolution, render_x, render_y);
#endif // DEBUG_RENDER_NEIGHBORHOOD
#else // DEBUG_PIXEL
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
        for (int x = 0; x < m_resolution.x; x++)
            ReSTIR_DI_SpatialReuse(m_render_data, m_resolution, x, y);
#endif // DEBUG_PIXEL
}

void CPURenderer::tracing_pass()
{
#if DEBUG_PIXEL
    int x, y;
#if DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = DEBUG_PIXEL_Y;
#else // DEBUG_FLIP_Y
    x = DEBUG_PIXEL_X;
    y = m_resolution.y - DEBUG_PIXEL_Y - 1;
#endif // DEBUG_FLIP_Y

    FullPathTracer(m_render_data, m_resolution, x, y);

#if DEBUG_RENDER_NEIGHBORHOOD
    // Rendering the neighborhood

#pragma omp parallel for schedule(dynamic)
    for (int render_y = std::max(0, y - DEBUG_NEIGHBORHOOD_SIZE); render_y <= std::min(m_resolution.y - 1, y + DEBUG_NEIGHBORHOOD_SIZE); render_y++)
        for (int render_x = std::max(0, x - DEBUG_NEIGHBORHOOD_SIZE); render_x <= std::min(m_resolution.x - 1, x + DEBUG_NEIGHBORHOOD_SIZE); render_x++)
        {
            if (render_x == x && render_y == y)
                continue;
               
            FullPathTracer(m_render_data, m_resolution, render_x, render_y);
        }

#endif // DEBUG_RENDER_NEIGHBORHOOD
#else // DEBUG_PIXEL
    std::atomic<int> lines_completed = 0;

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
    {
        for (int x = 0; x < m_resolution.x; x++)
            FullPathTracer(m_render_data, m_resolution, x, y);

        if (omp_get_thread_num() == 0 && m_render_data.render_settings.samples_per_frame == 1)
            // Only displaying per frame progress if we're only rendering one frame
            if (m_resolution.y > 25 && lines_completed % (m_resolution.y / 25))
                std::cout << lines_completed / (float)m_resolution.y * 100 << "%" << std::endl;
    }
#endif // DEBUG_PIXEL
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
            // Scaling by sample count
            hdr_color = hdr_color / float(m_render_data.render_settings.sample_number);

            ColorRGB32F tone_mapped = ColorRGB32F(1.0f) - exp(-hdr_color * exposure);
            tone_mapped = pow(tone_mapped, 1.0f / gamma);

            m_render_data.buffers.pixels[index] = tone_mapped;
        }
    }
}
