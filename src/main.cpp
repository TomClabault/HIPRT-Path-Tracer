/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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

extern ImGuiLogger g_imgui_logger;

#define GPU_RENDER 0

int main(int argc, char* argv[])
{   
    CommandlineArguments cmd_arguments = CommandlineArguments::process_command_line_args(argc, argv);

    int width = cmd_arguments.render_width;
    int height = cmd_arguments.render_height;

    g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Reading scene file %s...", cmd_arguments.scene_file_path.c_str());

    std::chrono::high_resolution_clock::time_point start_scene, start_full;
    std::chrono::high_resolution_clock::time_point stop_scene, stop_full;
    Scene parsed_scene;
    SceneParserOptions options(cmd_arguments.scene_file_path);

    options.override_aspect_ratio = (float)width / height;
    start_scene = std::chrono::high_resolution_clock::now();
    start_full = std::chrono::high_resolution_clock::now();
    Assimp::Importer assimp_importer;
    SceneParser::parse_scene_file(cmd_arguments.scene_file_path, assimp_importer, parsed_scene, options);
    stop_scene = std::chrono::high_resolution_clock::now();

    g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Scene geometry parsed in %ldms", std::chrono::duration_cast<std::chrono::milliseconds>(stop_scene - start_scene).count());
    g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Reading envmap %s...", cmd_arguments.skysphere_file_path.c_str());

    // TODO we only need 3 channels for the envmap but the only supported formats are 1, 2, 4 channels in HIP/CUDA, not 3
    Image32Bit envmap_image;
    ThreadManager::start_thread(ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD, ThreadFunctions::read_envmap, std::ref(envmap_image), cmd_arguments.skysphere_file_path, 4, true);
#if GPU_RENDER
    std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx = std::make_shared<HIPRTOrochiCtx>(0);

    RenderWindow render_window(width, height, hiprt_orochi_ctx);

    std::shared_ptr<GPURenderer> renderer = render_window.get_renderer();
    renderer->set_envmap(envmap_image, cmd_arguments.skysphere_file_path);
    renderer->set_camera(parsed_scene.camera);
    renderer->set_scene(parsed_scene);
    // Launching the background kernel precompilation
    renderer->precompile_kernels();

    // Joining everyone before starting the render except the precompilation threads
    ThreadManager::join_all_threads({ ThreadManager::GPU_RENDERER_PRECOMPILE_KERNELS_THREAD_KEY, ThreadManager::RENDERER_PRECOMPILE_KERNELS, ThreadManager::RESTIR_DI_PRECOMPILE_KERNELS });

    stop_full = std::chrono::high_resolution_clock::now();
    g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Full scene parsed & built in %ldms", std::chrono::duration_cast<std::chrono::milliseconds>(stop_full - start_full).count());
    renderer->get_hiprt_scene().print_statistics(std::cout);

    // We don't need the scene anymore, we can free it now (freeing the ASSIMP scene data)
    assimp_importer.FreeScene();
    // Freeing the renderer's scene data (i.e. the data converted from ASSIMP)
    parsed_scene = Scene();
    envmap_image.free();

    render_window.run();
#else

    width = 639;
    height = 346;

    g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "[%dx%d]: %d samples ; %d bounces\n\n", width, height, cmd_arguments.render_samples, cmd_arguments.bounces);

    CPURenderer cpu_renderer(width, height);
    cpu_renderer.get_render_settings().nb_bounces = cmd_arguments.bounces;
    cpu_renderer.get_render_settings().samples_per_frame = cmd_arguments.render_samples;
    cpu_renderer.set_envmap(envmap_image);
    cpu_renderer.set_camera(parsed_scene.camera);
    cpu_renderer.set_scene(parsed_scene);

    ThreadManager::join_all_threads();

    stop_full = std::chrono::high_resolution_clock::now();
    std::cout << "Full scene & textures parsed in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_full - start_full).count() << "ms" << std::endl;
    cpu_renderer.render();
    cpu_renderer.tonemap(2.2f, 1.0f);

    /*Image32Bit image_denoised_1 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 1.0f);
    Image32Bit image_denoised_075 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 0.75f);
    Image32Bit image_denoised_05 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 0.5f);*/

    cpu_renderer.get_framebuffer().write_image_png("CPU_RT_output.png");
    /*image_denoised_1.write_image_png("CPU_RT_output_denoised_1.png");
    image_denoised_075.write_image_png("CPU_RT_output_denoised_075.png");
    image_denoised_05.write_image_png("CPU_RT_output_denoised_05.png");*/
#endif

    return 0;
}
