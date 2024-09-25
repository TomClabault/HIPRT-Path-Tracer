/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Renderer/BVH.h"
#include "Renderer/CPURenderer.h"
#include "Renderer/GPURenderer.h"
#include "Scene/Camera.h"
#include "Scene/SceneParser.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"
#include "Utils/CommandlineArguments.h"
#include "Utils/Utils.h"

#include "stb_image_write.h"

#include <chrono>
#include <cmath>
#include <iostream>

#define GPU_RENDER 1

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

int main(int argc, char* argv[])
{
    CommandlineArguments cmd_arguments = CommandlineArguments::process_command_line_args(argc, argv);

    const int width = cmd_arguments.render_width;
    const int height = cmd_arguments.render_height;

    std::cout << "Reading scene file " << cmd_arguments.scene_file_path << " ..." << std::endl;

    std::chrono::high_resolution_clock::time_point start, start_full;
    std::chrono::high_resolution_clock::time_point stop, stop_full;
    Scene parsed_scene;
    SceneParserOptions options;

    options.nb_texture_threads = 10;
    options.override_aspect_ratio = (float)width / height;
    start = std::chrono::high_resolution_clock::now();
    start_full = std::chrono::high_resolution_clock::now();
    SceneParser::parse_scene_file(cmd_arguments.scene_file_path, parsed_scene, options);
    stop = std::chrono::high_resolution_clock::now();

    std::cout << "Scene geometry parsed in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;

    std::cout << "Reading \"" << cmd_arguments.skysphere_file_path << "\" envmap..." << std::endl;

    // TODO we only need 3 channels for the envmap but the only supported formats are 1, 2, 4 channels in HIP/CUDA, not 3
    Image32Bit envmap_image;
    ThreadManager::start_thread(ThreadManager::ENVMAP_LOAD_THREAD_KEY, ThreadFunctions::read_image_hdr, std::ref(envmap_image), cmd_arguments.skysphere_file_path, 4, true);
#if GPU_RENDER
    std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx = std::make_shared<HIPRTOrochiCtx>(0);

    RenderWindow render_window(width, height, hiprt_orochi_ctx);

    std::shared_ptr<GPURenderer> renderer = render_window.get_renderer();
    renderer->set_envmap(envmap_image);
    renderer->set_camera(parsed_scene.camera);
    renderer->set_scene(parsed_scene);
    stop_full = std::chrono::high_resolution_clock::now();
    std::cout << "Full scene parsed & built in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_full - start_full).count() << "ms" << std::endl;

    render_window.run();

#else

    std::cout << "[" << width << "x" << height << "]: " << cmd_arguments.render_samples << " samples ; " << cmd_arguments.bounces << " bounces" << std::endl << std::endl;

    CPURenderer cpu_renderer(width, height);
    cpu_renderer.set_scene(parsed_scene);
    cpu_renderer.set_envmap(envmap_image);
    cpu_renderer.set_camera(parsed_scene.camera);
    cpu_renderer.get_render_settings().nb_bounces = cmd_arguments.bounces;
    cpu_renderer.get_render_settings().samples_per_frame = cmd_arguments.render_samples;

    glm::mat4x4 rotation_matrix = glm::orientate3(glm::vec3(0.0f * 2.0f * M_PI, 1.0f * 2.0f * M_PI, 0.0f * 2.0f * M_PI));
    glm::mat4x4 rotation_matrix_inv = glm::inverse(rotation_matrix);

    cpu_renderer.get_render_data().world_settings.envmap_to_world_matrix = *reinterpret_cast<float4x4*>(&rotation_matrix);
    cpu_renderer.get_render_data().world_settings.world_to_envmap_matrix = *reinterpret_cast<float4x4*>(&rotation_matrix_inv);

    ThreadManager::join_threads(ThreadManager::TEXTURE_THREADS_KEY);
    stop_full = std::chrono::high_resolution_clock::now();
    std::cout << "Full scene & textures parsed in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_full - start_full).count() << "ms" << std::endl;
    cpu_renderer.render();
    cpu_renderer.tonemap(2.2f, 1.0f);

    Image32Bit image_denoised_1 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 1.0f);
    Image32Bit image_denoised_075 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 0.75f);
    Image32Bit image_denoised_05 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 0.5f);

    cpu_renderer.get_framebuffer().write_image_png("CPU_RT_output.png");
    image_denoised_1.write_image_png("CPU_RT_output_denoised_1.png");
    image_denoised_075.write_image_png("CPU_RT_output_denoised_075.png");
    image_denoised_05.write_image_png("CPU_RT_output_denoised_05.png");
#endif

    return 0;
}
