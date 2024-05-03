/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include <atomic>
#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>

#include <stb_image_write.h>

#include "Image/Envmap.h"
#include "Image/Image.h"
#include "Renderer/BVH.h"
#include "Renderer/RenderKernel.h"
#include "Renderer/Triangle.h"
#include "Scene/Camera.h"
#include "Scene/SceneParser.h"
#include "UI/RenderWindow.h"
#include "Utils/CommandlineArguments.h"
#include "Utils/Utils.h"

#include "Device/kernels/PathTracerKernel.h"

#define GPU_RENDER 0

int main(int argc, char* argv[])
{
#if GPU_RENDER
    CommandLineArguments arguments = CommandLineArguments::process_command_line_args(argc, argv);

    const int default_width = arguments.render_width, default_height = arguments.render_height;
    RenderWindow render_window(default_width, default_height);
    {
        std::cout << std::endl << "Reading scene file " << arguments.scene_file_path << " ..." << std::endl;
        Scene parsed_scene = SceneParser::parse_scene_file(arguments.scene_file_path, (float)default_width / default_height);
        std::cout << std::endl;

        Renderer& renderer = render_window.get_renderer();
        renderer.set_scene(parsed_scene);
        renderer.set_camera(parsed_scene.camera);
    }
    render_window.run();

    return 0;
#else
    CommandLineArguments cmd_arguments = CommandLineArguments::process_command_line_args(argc, argv);

    const int width = cmd_arguments.render_width;
    const int height = cmd_arguments.render_height;

    std::cout << "Reading scene file " << cmd_arguments.scene_file_path << " ..." << std::endl;
    Scene parsed_scene = SceneParser::parse_scene_file(cmd_arguments.scene_file_path);
    std::cout << std::endl;

    std::cout << "[" << width << "x" << height << "]: " << cmd_arguments.render_samples << " samples ; " << cmd_arguments.bounces << " bounces" << std::endl << std::endl;

    Image framebuffer(width, height);
    std::vector<int> debug_pixel_active_buffer(width * height, 0);
    std::vector<ColorRGB> denoiser_albedo(width * height, ColorRGB(0.0f));
    std::vector<float3> denoiser_normals(width * height, float3{ 0.0f, 0.0f, 0.0f });
    std::vector<int> pixel_sample_count(width * height, 0);
    std::vector<float> pixel_squared_luminance(width * height, 0.0f);

    HIPRTRenderData render_data;
    render_data.geom = nullptr;

    render_data.buffers.emissive_triangles_count = parsed_scene.emissive_triangle_indices.size();
    render_data.buffers.emissive_triangles_indices = parsed_scene.emissive_triangle_indices.data();
    render_data.buffers.materials_buffer = parsed_scene.materials.data();
    render_data.buffers.material_indices = parsed_scene.material_indices.data();
    render_data.buffers.normals_present = parsed_scene.normals_present.data();
    render_data.buffers.pixels = framebuffer.data().data();
    render_data.buffers.triangles_indices = parsed_scene.triangle_indices.data();
    render_data.buffers.triangles_vertices = parsed_scene.vertices_positions.data();
    render_data.buffers.vertex_normals = parsed_scene.vertex_normals.data();

    render_data.aux_buffers.debug_pixel_active = debug_pixel_active_buffer.data();
    render_data.aux_buffers.denoiser_albedo = denoiser_albedo.data();
    render_data.aux_buffers.denoiser_normals = denoiser_normals.data();
    render_data.aux_buffers.pixel_sample_count = pixel_sample_count.data();
    render_data.aux_buffers.pixel_squared_luminance = pixel_squared_luminance.data();

    render_data.world_settings.ambient_light_color = ColorRGB(0.5f);
    render_data.world_settings.use_ambient_light = true;

    render_data.render_settings.nb_bounces = cmd_arguments.bounces;
    render_data.render_settings.samples_per_frame = cmd_arguments.render_samples;

    std::vector<Triangle> triangle_buffer = parsed_scene.get_triangles();
    BVH bvh(&triangle_buffer);
    render_data.cpu_only.bvh = &bvh;

    int2 resolution{ width, height };
    HIPRTCamera camera = parsed_scene.camera.to_hiprt();

    auto start = std::chrono::high_resolution_clock::now();
    std::atomic<int> lines_completed = 0;
#if DEBUG_PIXEL
#if DEBUG_EXACT_COORDINATE
    for (int y = DEBUG_PIXEL_Y; y < m_frame_buffer.height; y++)
    {
        for (int x = DEBUG_PIXEL_X; x < m_frame_buffer.width; x++)
#else
    for (int y = m_frame_buffer.height - DEBUG_PIXEL_Y - 1; y < m_frame_buffer.height; y++)
    {
        for (int x = DEBUG_PIXEL_X; x < m_frame_buffer.width; x++)
#endif
#else
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
#endif
            PathTracerKernel(render_data, resolution, camera, x, y);

        lines_completed++;

        if (omp_get_thread_num() == 0)
            if (lines_completed % (height / 25))
                std::cout << lines_completed / (float)height * 100 << "%" << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;

    ApplicationSettings default_app_settings;
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;

            ColorRGB hdr_color = render_data.buffers.pixels[index];
            // Scaling by sample count
            hdr_color = hdr_color / float(cmd_arguments.render_samples);

            ColorRGB tone_mapped = ColorRGB(1.0f) - exp(-hdr_color * default_app_settings.tone_mapping_exposure);
            tone_mapped = pow(tone_mapped, 1.0f / default_app_settings.tone_mapping_gamma);

            render_data.buffers.pixels[index] = tone_mapped;
        }
    }

    Image image_denoised_1 = Utils::OIDN_denoise(framebuffer, width, height, 1.0f);
    Image image_denoised_075 = Utils::OIDN_denoise(framebuffer, width, height, 0.75f);
    Image image_denoised_05 = Utils::OIDN_denoise(framebuffer, width, height, 0.5f);

    framebuffer.write_image_png("RT_output.png");
    image_denoised_1.write_image_png("RT_output_denoised_1.png");
    image_denoised_075.write_image_png("RT_output_denoised_075.png");
    image_denoised_05.write_image_png("RT_output_denoised_05.png");

    return 0;
#endif
}
