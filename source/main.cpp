#include <iostream>
#include <chrono>
#include <cmath>

#include <stb_image_write.h>

#include "bvh.h"
#include "camera.h"
#include "image_io.h"
#include "render_kernel.h"
#include "simple_material.h"
#include "sphere.h"
#include "tests.h"
#include "triangle.h"
#include "utils.h"

#include "xorshift.h"

Sphere add_sphere_to_scene(ParsedScene& parsed_scene, const Point& center, float radius, const SimpleMaterial& material, int primitive_index)
{
    int material_index = parsed_scene.materials.size();

    parsed_scene.materials.push_back(material);
    parsed_scene.material_indices.push_back(material_index);

    Sphere sphere(center, radius, primitive_index);

    return sphere;
}

struct CommandLineArguments
{
    std::string scene_file_path = "data/OBJs/cornell_pbr.obj";
    std::string skysphere_file_path = "data/Skyspheres/evening_road_01_puresky_2k.hdr";

    int render_width = 512, render_height = 512;
    int render_samples = 64;
    int bounces = 8;
};

void process_command_line(int argc, char** argv, CommandLineArguments& arguments)
{
    for (int i = 1; i < argc; i++)
    {
        std::string string_argv = std::string(argv[i]);
        if (string_argv.starts_with("--sky="))
            arguments.skysphere_file_path = string_argv.substr(6);
        else if (string_argv.starts_with("--w="))
            arguments.render_width = std::atoi(string_argv.substr(4).c_str());
        else if (string_argv.starts_with("--h="))
            arguments.render_height = std::atoi(string_argv.substr(4).c_str());
        else if (string_argv.starts_with("--samples="))
            arguments.render_samples = std::atoi(string_argv.substr(10).c_str());
        else if (string_argv.starts_with("--bounces="))
            arguments.bounces = std::atoi(string_argv.substr(10).c_str());
        else
            //Assuming scene file path
            arguments.scene_file_path = string_argv;
    }
}

int main(int argc, char* argv[])
{
    CommandLineArguments arguments;
    process_command_line(argc, argv, arguments);

    const int width = arguments.render_width;
    const int height = arguments.render_height;

    std::cout << "Reading OBJ " << arguments.scene_file_path << " ..." << std::endl;
    ParsedScene parsed_scene = Utils::parse_scene_file(arguments.scene_file_path);

    //Sphere sphere = add_sphere_to_scene(parsed_obj, Point(0.3275, 0.7, 0.3725), 0.2, SimpleMaterial {Color(0.0f), Color(1.0f, 0.71, 0.29), 1.0f, 0.4f}, parsed_obj.triangles.size());
    //std::vector<Sphere> spheres = { sphere };
    std::vector<Sphere> spheres;

    BVH bvh(&parsed_scene.triangles);

    std::vector<Triangle> triangle_buffer = parsed_scene.triangles;
    std::vector<SimpleMaterial> materials_buffer = parsed_scene.materials;
    std::vector<int> emissive_triangle_indices_buffer = parsed_scene.emissive_triangle_indices;
    std::vector<int> materials_indices_buffer = parsed_scene.material_indices;
    std::vector<Sphere> sphere_buffer = spheres;

    int skysphere_width, skysphere_height;
    std::cout << "Reading Environment Map " << arguments.skysphere_file_path << " ..." << std::endl;
    Image skysphere_data = Utils::read_image_float(arguments.skysphere_file_path, skysphere_width, skysphere_height);
    std::vector<float> env_map_cdf = Utils::compute_env_map_cdf(skysphere_data);

    std::cout << "[" << width << "x" << height << "]: " << arguments.render_samples << " samples" << std::endl << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    Image image_buffer(arguments.render_width, arguments.render_height);
    auto render_kernel = RenderKernel(
        arguments.render_width, arguments.render_height,
        arguments.render_samples, arguments.bounces,
        image_buffer,
        triangle_buffer,
        materials_buffer,
        emissive_triangle_indices_buffer,
        materials_indices_buffer,
        sphere_buffer,
        bvh,
        skysphere_data,
        env_map_cdf);
    //render_kernel.set_camera(Camera::CORNELL_BOX_CAMERA);
    //render_kernel.set_camera(Camera::GANESHA_CAMERA);
    //render_kernel.set_camera(Camera::ITE_ORB_CAMERA);
    render_kernel.set_camera(Camera::PBRT_DRAGON_CAMERA);
    //render_kernel.set_camera(Camera::MIS_CAMERA);

    render_kernel.render();

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;

    Image image_denoised_1 = Utils::OIDN_denoise(image_buffer, 1.0f);
    Image image_denoised_075 = Utils::OIDN_denoise(image_buffer, 0.75f);
    Image image_denoised_05 = Utils::OIDN_denoise(image_buffer, 0.5f);

    write_image_png(image_buffer, "RT_output.png");
    write_image_png(image_denoised_1, "RT_output_denoised_1.png");
    write_image_png(image_denoised_075, "RT_output_denoised_0.75.png");
    write_image_png(image_denoised_05, "RT_output_denoised_0.5.png");

    return 0;
}
