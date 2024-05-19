/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/kernels/PathTracerKernel.h"
#include "HIPRT-Orochi/OrochiTexture.h"
#include "Image/Image.h"
#include "Renderer/BVH.h"
#include "Renderer/CPURenderer.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/Triangle.h"
#include "Scene/Camera.h"
#include "Scene/SceneParser.h"
#include "UI/RenderWindow.h"
#include "Utils/CommandlineArguments.h"
#include "Utils/Utils.h"

#include "stb_image_write.h"

#include <chrono>
#include <cmath>
#include <iostream>

#define GPU_RENDER 1

int main(int argc, char* argv[])
{
    CommandLineArguments cmd_arguments = CommandLineArguments::process_command_line_args(argc, argv);

    const int width = cmd_arguments.render_width;
    const int height = cmd_arguments.render_height;

    std::cout << std::endl << "Reading scene file " << cmd_arguments.scene_file_path << " ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    Scene parsed_scene;
    for (int i = 0; i < 1; i++) 
        parsed_scene = SceneParser::parse_scene_file(cmd_arguments.scene_file_path, (float)width / height);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Scene parsed in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
        
    std::cout << "Reading \"" << cmd_arguments.skysphere_file_path << "\" envmap..." << std::endl;
    ImageRGBA envmap_image = ImageRGBA::read_image_hdr(cmd_arguments.skysphere_file_path, /* flip Y */ true);

#if GPU_RENDER

    RenderWindow render_window(width, height);

    GPURenderer& renderer = render_window.get_renderer();
    renderer.set_scene(parsed_scene);
    renderer.set_envmap(envmap_image);
    renderer.set_camera(parsed_scene.camera);
    render_window.run();

    return 0;

#else

    std::cout << "[" << width << "x" << height << "]: " << cmd_arguments.render_samples << " samples ; " << cmd_arguments.bounces << " bounces" << std::endl << std::endl;

    CPURenderer cpu_renderer(width, height);
    cpu_renderer.set_scene(parsed_scene);
    cpu_renderer.set_envmap(envmap_image);
    cpu_renderer.set_camera(parsed_scene.camera);
    cpu_renderer.get_render_settings().nb_bounces = cmd_arguments.bounces;
    cpu_renderer.get_render_settings().samples_per_frame = cmd_arguments.render_samples;
    cpu_renderer.render();
    cpu_renderer.tonemap(2.2f, 1.0f);

    Image image_denoised_1 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 1.0f);
    Image image_denoised_075 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 0.75f);
    Image image_denoised_05 = Utils::OIDN_denoise(cpu_renderer.get_framebuffer(), width, height, 0.5f);

    cpu_renderer.get_framebuffer().write_image_png("CPU_RT_output.png");
    image_denoised_1.write_image_png("CPU_RT_output_denoised_1.png");
    image_denoised_075.write_image_png("CPU_RT_output_denoised_075.png");
    image_denoised_05.write_image_png("CPU_RT_output_denoised_05.png");

    return 0;
#endif
}
