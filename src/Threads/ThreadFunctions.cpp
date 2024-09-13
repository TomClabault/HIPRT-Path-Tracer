/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Compiler/GPUKernel.h"
#include "Threads/ThreadFunctions.h"

void ThreadFunctions::compile_kernel(GPUKernel& kernel, std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx)
{
    kernel.compile(hiprt_ctx);
}

void ThreadFunctions::load_scene_texture(Scene& parsed_scene, std::string scene_path, const std::vector<std::pair<aiTextureType, std::string>>& tex_paths, const std::vector<int>& material_indices, int thread_index, int nb_threads)
{
    // Preparing the scene_filepath so that it's ready to be appended with the texture name
    std::string corrected_filepath;
    corrected_filepath = scene_path;
    corrected_filepath = corrected_filepath.substr(0, corrected_filepath.rfind('/') + 1);

    while (thread_index < parsed_scene.textures.size())
    {
        std::string full_path = corrected_filepath + tex_paths[thread_index].second;
        aiTextureType type = tex_paths[thread_index].first;
        int nb_channels;

        switch (type)
        {
        case aiTextureType_BASE_COLOR:
        case aiTextureType_DIFFUSE:
            // 4 Channels because we may want the alpha for transparency handling
            nb_channels = 4;
            break;

        case aiTextureType_NORMALS:
        case aiTextureType_HEIGHT:
            // Don't need the alpha
            // TODO we only need 3 channels here but it's tricky to handle 3 channels texture with HIP/CUDA. Supported formats are only 1, 2, 4 channels, not three
            nb_channels = 4;
            break;

        case aiTextureType_DIFFUSE_ROUGHNESS:
            if (parsed_scene.materials[material_indices[thread_index]].roughness_metallic_texture_index != -1)
            {
                // This means we have a packed metallic/roughness texture
                nb_channels = 2;

                break;
            }
            else
            {
                // Otherwise, we don't have a packed metallic/roughness texture so only 1 channel just for the roughness
                nb_channels = 1;

                break;
            }

        default:
            nb_channels = 1;
            break;
        }

        Image8Bit texture = Image8Bit::read_image(full_path, nb_channels, false);
        parsed_scene.textures_dims[thread_index] = make_int2(texture.width, texture.height);
        parsed_scene.textures[thread_index] = texture;

        thread_index += nb_threads;
    }
}

void ThreadFunctions::read_image_hdr(Image32Bit& hdr_image_out, const std::string& filepath, int wanted_channel_count, bool flip_Y)
{
    hdr_image_out = Image32Bit::read_image_hdr(filepath, wanted_channel_count, flip_Y);
}
