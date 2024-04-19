/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include <iostream>
#include <chrono>
#include <cmath>

#include <stb_image_write.h>

#include "Image/envmap.h"
#include "Image/image.h"
#include "Renderer/bvh.h"
#include "Renderer/render_kernel.h"
#include "Renderer/triangle.h"
#include "Scene/camera.h"
#include "Scene/scene_parser.h"
#include "UI/render_window.h"
#include "Utils/commandline_arguments.h"
#include "Utils/utils.h"

#define GPU_RENDER 1

int main(int argc, char* argv[])
{
#if GPU_RENDER
    CommandLineArguments arguments = CommandLineArguments::process_command_line_args(argc, argv);

    const int default_width = arguments.render_width, default_height = arguments.render_height;
    RenderWindow render_window(default_width, default_height);
    {
        std::cout << "Reading scene file " << arguments.scene_file_path << " ..." << std::endl;
        Scene parsed_scene = SceneParser::parse_scene_file(arguments.scene_file_path, (float)default_width / default_height);
        std::cout << std::endl;

        Renderer& renderer = render_window.get_renderer();
        renderer.set_scene(parsed_scene);
        renderer.set_camera(parsed_scene.camera);
    }
    render_window.reset_sample_number();
    render_window.run();

    return 0;
#else
    CommandLineArguments cmd_arguments = CommandLineArguments::process_command_line_args(argc, argv);

    const int width = cmd_arguments.render_width;
    const int height = cmd_arguments.render_height;

    std::vector<Sphere> spheres;

    std::cout << "Reading scene file " << cmd_arguments.scene_file_path << " ..." << std::endl;
    Scene parsed_scene = SceneParser::parse_scene_file(cmd_arguments.scene_file_path);
    std::cout << std::endl;

    const std::vector<int>& triangle_indices = parsed_scene.triangle_indices;
    const std::vector<unsigned char>& normals_present = parsed_scene.normals_present;
    std::vector<Vector> vertex_normals(parsed_scene.vertex_normals.size());
    std::transform(parsed_scene.vertex_normals.begin(), parsed_scene.vertex_normals.end(), vertex_normals.begin(), [](hiprtFloat3 hiprt_vec) { return Vector(hiprt_vec.x, hiprt_vec.y, hiprt_vec.z); });
    std::vector<Triangle> triangle_buffer = parsed_scene.get_triangles();
    BVH bvh(&triangle_buffer);

    std::vector<RendererMaterial> materials_buffer = parsed_scene.materials;
    std::vector<int> emissive_triangle_indices_buffer = parsed_scene.emissive_triangle_indices;
    std::vector<int> materials_indices_buffer = parsed_scene.material_indices;
    std::vector<Sphere> sphere_buffer = spheres;

    std::cout << "Reading Environment Map " << cmd_arguments.skysphere_file_path << " ..." << std::endl;
    EnvironmentMap env_map = EnvironmentMap::read_from_file(cmd_arguments.skysphere_file_path);

    std::cout << "[" << width << "x" << height << "]: " << cmd_arguments.render_samples << " samples ; " << cmd_arguments.bounces << " bounces" << std::endl << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    Image image_buffer(width, height);
    auto render_kernel = RenderKernel(
        width, height,
        cmd_arguments.render_samples, cmd_arguments.bounces,
        image_buffer,
        triangle_buffer,
        triangle_indices,
        normals_present,
        vertex_normals,
        materials_buffer,
        emissive_triangle_indices_buffer,
        materials_indices_buffer,
        sphere_buffer,
        bvh,
        env_map);
    render_kernel.set_camera(parsed_scene.camera);
    render_kernel.render();

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;

    Image image_denoised_1 = Utils::OIDN_denoise(image_buffer, width, height, 1.0f);
    Image image_denoised_075 = Utils::OIDN_denoise(image_buffer, width, height, 0.75f);
    Image image_denoised_05 = Utils::OIDN_denoise(image_buffer, width, height, 0.5f);

    image_buffer.write_image_png("RT_output.png");
    image_denoised_1.write_image_png("RT_output_denoised_1.png");
    image_denoised_075.write_image_png("RT_output_denoised_075.png");
    image_denoised_05.write_image_png("RT_output_denoised_05.png");

    return 0;
#endif
}
