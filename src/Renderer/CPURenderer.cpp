/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/kernels/PathTracerKernel.h"
#include "Renderer/CPURenderer.h"
#include "UI/ApplicationSettings.h"

#include <atomic>
#include <chrono>
#include <omp.h>

CPURenderer::CPURenderer(int width, int height) : m_resolution(make_int2(width, height))
{
    m_framebuffer = Image(width, height);

    // Resizing buffers + initial value
    m_debug_pixel_active_buffer.resize(width * height, 0);
    m_denoiser_albedo.resize(width * height, ColorRGB(0.0f));
    m_denoiser_normals.resize(width * height, float3{ 0.0f, 0.0f, 0.0f });
    m_pixel_sample_count.resize(width * height, 0);
    m_pixel_squared_luminance.resize(width * height, 0.0f);
}

void CPURenderer::set_scene(Scene& parsed_scene)
{
    m_render_data.geom = nullptr;

    m_render_data.buffers.emissive_triangles_count = parsed_scene.emissive_triangle_indices.size();
    m_render_data.buffers.emissive_triangles_indices = parsed_scene.emissive_triangle_indices.data();
    m_render_data.buffers.materials_buffer = parsed_scene.materials.data();
    m_render_data.buffers.material_indices = parsed_scene.material_indices.data();
    m_render_data.buffers.has_vertex_normals = parsed_scene.has_vertex_normals.data();
    m_render_data.buffers.pixels = m_framebuffer.data().data();
    m_render_data.buffers.triangles_indices = parsed_scene.triangle_indices.data();
    m_render_data.buffers.vertices_positions = parsed_scene.vertices_positions.data();
    m_render_data.buffers.vertex_normals = parsed_scene.vertex_normals.data();
    m_render_data.buffers.texcoords = parsed_scene.texcoords.data();

    m_render_data.buffers.material_textures = parsed_scene.textures.data();
    m_render_data.buffers.texture_is_srgb = parsed_scene.textures_is_srgb.data();

    m_render_data.aux_buffers.denoiser_albedo = m_denoiser_albedo.data();
    m_render_data.aux_buffers.denoiser_normals = m_denoiser_normals.data();
    m_render_data.aux_buffers.pixel_sample_count = m_pixel_sample_count.data();
    m_render_data.aux_buffers.pixel_squared_luminance = m_pixel_squared_luminance.data();

    std::cout << "Building scene BVH..." << std::endl;
    m_triangle_buffer = parsed_scene.get_triangles();
    m_bvh = std::make_shared<BVH>(&m_triangle_buffer);
    m_render_data.cpu_only.bvh = m_bvh.get();
}

void CPURenderer::set_envmap(ImageRGBA& envmap_image)
{
    m_render_data.world_settings.envmap = &envmap_image;
    m_render_data.world_settings.envmap_width = envmap_image.width;
    m_render_data.world_settings.envmap_height = envmap_image.height;
    m_render_data.world_settings.envmap_cdf = envmap_image.get_cdf().data();
}

void CPURenderer::set_camera(Camera& camera)
{
    m_hiprt_camera = camera.to_hiprt();
}

HIPRTRenderSettings& CPURenderer::get_render_settings()
{
    return m_render_data.render_settings;
}

Image& CPURenderer::get_framebuffer()
{
    return m_framebuffer;
}
#define DEBUG_PIXEL 1
#define DEBUG_EXACT_COORDINATE 0
#define DEBUG_PIXEL_X 702
#define DEBUG_PIXEL_Y 484


void CPURenderer::render()
{
    std::cout << "CPU rendering..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::atomic<int> lines_completed = 0;
#if DEBUG_PIXEL
#if DEBUG_EXACT_COORDINATE
    for (int y = DEBUG_PIXEL_Y; y < m_resolution.y; y++)
    {
        for (int x = DEBUG_PIXEL_X; x < m_resolution.x; x++)
#else
    for (int y = m_resolution.y - DEBUG_PIXEL_Y - 1; y < m_resolution.y; y++)
    {
        for (int x = DEBUG_PIXEL_X; x < m_resolution.x; x++)
#endif
#else
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
    {
        for (int x = 0; x < m_resolution.x; x++)
#endif
            PathTracerKernel(m_render_data, m_resolution, m_hiprt_camera, x, y);

        lines_completed++;

        if (omp_get_thread_num() == 0)
            if (m_resolution.y > 25 && lines_completed % (m_resolution.y / 25))
                std::cout << lines_completed / (float)m_resolution.y * 100 << "%" << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void CPURenderer::tonemap(float gamma, float exposure)
{
//#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_resolution.y; y++)
    {
        for (int x = 0; x < m_resolution.x; x++)
        {
            int index = x + y * m_resolution.x;

            ColorRGB hdr_color = m_render_data.buffers.pixels[index];
            // Scaling by sample count
            hdr_color = hdr_color / float(m_render_data.render_settings.samples_per_frame);

            ColorRGB tone_mapped = ColorRGB(1.0f) - exp(-hdr_color * exposure);
            tone_mapped = pow(tone_mapped, 1.0f / gamma);

            m_render_data.buffers.pixels[index] = tone_mapped;
        }
    }
}
