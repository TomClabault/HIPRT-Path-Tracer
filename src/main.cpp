#include <iostream>
#include <chrono>
#include <cmath>

#include <stb_image_write.h>

#include "Image/envmap.h"
#include "Image/image.h"
#include "Renderer/bvh.h"
#include "Renderer/render_kernel.h"
#include "Renderer/renderer_material.h"
#include "Renderer/triangle.h"
#include "Scene/camera.h"
#include "Scene/scene_parser.h"
#include "UI/app_window.h"
#include "Utils/commandline_arguments.h"
#include "Utils/utils.h"
#include "Utils/xorshift.h"

#define GPU_RENDER 0

// TODO remove the two functions when Image class is done
inline float clamp(const float x, const float min, const float max)
{
    if (x < min) return min;
    else if (x > max) return max;
    else return x;
}

bool write_image_png(const std::vector<HIPRTColor>& m_pixel_data, int m_width, int m_height, const char* filename)
{
    if (m_height * m_width == 0)
        return false;

    std::vector<unsigned char> tmp(m_height * m_width * 3);
    for (unsigned i = 0; i < m_width * m_height; i++)
    {
        HIPRTColor pixel = m_pixel_data[i] * 255;

        tmp[i * 3 + 0] = clamp(pixel.r, 0, 255);
        tmp[i * 3 + 1] = clamp(pixel.g, 0, 255);
        tmp[i * 3 + 2] = clamp(pixel.b, 0, 255);
    }

    stbi_flip_vertically_on_write(true);
    return stbi_write_png(filename, m_width, m_height, 3, tmp.data(), m_width * 3) != 0;
}

int main(int argc, char* argv[])
{
#if GPU_RENDER
    CommandLineArguments arguments = CommandLineArguments::process_command_line_args(argc, argv);

    const int default_width = arguments.render_width, default_height = arguments.render_height;
    AppWindow app_window(default_width, default_height);
    {
        std::cout << "Reading scene file " << arguments.scene_file_path << " ..." << std::endl;
        Scene parsed_scene = SceneParser::parse_scene_file(arguments.scene_file_path, (float)default_width / default_height);
        std::cout << std::endl;

        app_window.set_renderer_scene(parsed_scene);
        app_window.get_renderer().set_camera(parsed_scene.camera);
    }
    app_window.reset_sample_number();
    app_window.run();

    return 0;
#else
    CommandLineArguments cmd_arguments = CommandLineArguments::process_command_line_args(argc, argv);

    const int width = cmd_arguments.render_width;
    const int height = cmd_arguments.render_height;

    RendererMaterial sphere_material;
    sphere_material.emission = HIPRTColor(0.0f);
    sphere_material.diffuse = HIPRTColor(1.0f, 0.71, 0.29);
    sphere_material.metalness = 1.0f;
    sphere_material.roughness = 1.0e-2f;
    sphere_material.ior = 1.4f;

    //Sphere sphere = add_sphere_to_scene(parsed_scene, Point(0.0, 1, 0.3725), 0.75, sphere_material, parsed_scene.triangles.size());
    //std::vector<Sphere> spheres = { sphere };
    std::vector<Sphere> spheres;

    std::cout << "Reading scene file " << cmd_arguments.scene_file_path << " ..." << std::endl;
    Scene parsed_scene = SceneParser::parse_scene_file(cmd_arguments.scene_file_path);
    std::cout << std::endl;

    std::vector<Triangle> triangle_buffer = parsed_scene.get_triangles();
    BVH bvh(&triangle_buffer);

    std::vector<RendererMaterial> materials_buffer = parsed_scene.materials;
    std::vector<int> emissive_triangle_indices_buffer = parsed_scene.emissive_triangle_indices;
    std::vector<int> materials_indices_buffer = parsed_scene.material_indices;
    std::vector<Sphere> sphere_buffer = spheres;

    std::cout << "Reading Environment Map " << cmd_arguments.skysphere_file_path << " ..." << std::endl;
    EnvironmentMap env_map = EnvironmentMap::read_from_file(cmd_arguments.skysphere_file_path);

    std::cout << "[" << width << "x" << height << "]: " << cmd_arguments.render_samples << " samples" << std::endl << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<HIPRTColor> image_buffer(width * height);
    auto render_kernel = RenderKernel(
        width, height,
        cmd_arguments.render_samples, cmd_arguments.bounces,
        image_buffer,
        triangle_buffer,
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

    std::vector<HIPRTColor> image_denoised_1 = Utils::OIDN_denoise(image_buffer, width, height, 1.0f);
    std::vector<HIPRTColor> image_denoised_075 = Utils::OIDN_denoise(image_buffer, width, height, 0.75f);
    std::vector<HIPRTColor> image_denoised_05 = Utils::OIDN_denoise(image_buffer, width, height, 0.5f);

    write_image_png(image_buffer, width, height, "RT_output.png");
    write_image_png(image_denoised_1, width, height, "RT_output_denoised_1.png");
    write_image_png(image_denoised_075, width, height, "RT_output_denoised_0.75.png");
    write_image_png(image_denoised_05, width, height, "RT_output_denoised_0.5.png");

    return 0;
#endif
}
