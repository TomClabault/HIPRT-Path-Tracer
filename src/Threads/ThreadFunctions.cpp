/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/HIPKernel.h"
#include "Threads/ThreadFunctions.h"

void ThreadFunctions::compile_kernel(HIPKernel& kernel, hiprtContext& hiprt_ctx)
{
    kernel.compile(hiprt_ctx);
}

void ThreadFunctions::load_texture(Scene& parsed_scene, std::string scene_path, const std::vector<std::pair<aiTextureType, std::string>>& tex_paths, int thread_index, int nb_threads)
{
    // Preparing the scene_filepath so that it's ready to be appended with the texture name
    std::string corrected_filepath;
    corrected_filepath = scene_path;
    corrected_filepath = corrected_filepath.substr(0, corrected_filepath.rfind('/') + 1);

    while (thread_index < parsed_scene.textures.size())
    {
        std::string full_path;
        full_path = corrected_filepath + tex_paths[thread_index].second;

        ImageRGBA texture = ImageRGBA::read_image(full_path, false);
        parsed_scene.textures_dims[thread_index] = make_int2(texture.width, texture.height);
        parsed_scene.textures[thread_index] = texture;

        thread_index += nb_threads;
    }

}
