/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Compiler/GPUKernel.h"
#include "Threads/ThreadFunctions.h"

void ThreadFunctions::compile_kernel(std::shared_ptr<GPUKernel> kernel, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
    kernel->compile(hiprt_orochi_ctx, func_name_sets, true, false);
}

void ThreadFunctions::compile_kernel_no_func_sets(std::shared_ptr<GPUKernel> kernel, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx)
{
    kernel->compile(hiprt_orochi_ctx, {});
}

void ThreadFunctions::compile_kernel_silent(std::shared_ptr<GPUKernel> kernel, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
    kernel->compile(hiprt_orochi_ctx, func_name_sets, true, true);
}

void ThreadFunctions::precompile_kernel(const std::string& kernel_function_name, const std::string& kernel_filepath, GPUKernelCompilerOptions options, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
    OROCHI_CHECK_ERROR(oroCtxSetCurrent(hiprt_orochi_ctx->orochi_ctx));

    GPUKernel kernel(kernel_filepath, kernel_function_name);
    kernel.set_precompiled(true);
    kernel.get_kernel_options() = options;
    kernel.compile(hiprt_orochi_ctx, func_name_sets, true, true);
}

void ThreadFunctions::load_scene_texture(Scene& parsed_scene, std::string scene_path, const std::vector<std::pair<aiTextureType, std::string>>& tex_paths, const std::vector<int>& material_indices, int thread_index, int nb_threads)
{
    // Preparing the scene_filepath so that it's ready to be appended with the texture name
    std::string corrected_filepath;
    corrected_filepath = scene_path;
    corrected_filepath = corrected_filepath.substr(0, corrected_filepath.rfind('/') + 1);

    // While loop here so that a single thread can parse multiple textures
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
            if (parsed_scene.materials[material_indices[thread_index]].roughness_metallic_texture_index != MaterialUtils::NO_TEXTURE)
            {
                // This means we have a packed metallic/roughness texture
                nb_channels = 4;

                break;
            }
            else
            {
                // Otherwise, we don't have a packed metallic/roughness texture so only 1 channel just for the roughness
                nb_channels = 1;

                break;
            }

        case aiTextureType_EMISSIVE:
            // TODO we only need 3 channels here but it's tricky to handle 3 channels texture with HIP/CUDA. Supported formats are only 1, 2, 4 channels, not three
            nb_channels = 4;
            break;

        default:
            nb_channels = 1;
            break;
        }

        Image8Bit texture = Image8Bit::read_image(full_path, nb_channels, false);

        int material_index = material_indices[thread_index];
        if (type == aiTextureType_EMISSIVE)
        {
            if (texture.is_constant_color(/* threshold */ 5))
            {
                // The emissive texture is constant color, we can then just not use that texture and use 
                // the emission filed of the material to store the emission of the texture
                parsed_scene.materials[material_index].emission_texture_index = MaterialUtils::CONSTANT_EMISSIVE_TEXTURE;

                ColorRGBA32F emission_rgba = texture.sample_rgba32f(make_float2(0, 0));
                parsed_scene.materials[material_index].emission = ColorRGB32F(emission_rgba.r, emission_rgba.g, emission_rgba.b);
            }
            else
                // If not emissive texture special case, we can actually read the texture
                parsed_scene.textures[thread_index] = texture;
        }
        else
        {
            // If not emissive texture special case, we can actually read the texture

            if (type == aiTextureType_DIFFUSE || type == aiTextureType_BASE_COLOR)
            {
                // For base color textures, we're going to search for alpha transparency in the texture
                unsigned char texture_fully_opaque = texture.is_fully_opaque() ? 1 : 0;
                parsed_scene.material_has_opaque_base_color_texture[material_index] = texture_fully_opaque;
            }
            parsed_scene.textures[thread_index] = texture;
        }

        thread_index += nb_threads;
    }
}

void ThreadFunctions::load_scene_parse_emissive_triangles(const aiScene* scene, Scene& parsed_scene)
{
    // Looping over all the meshes
    int current_triangle_index = 0;
    for (int mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
    {
        aiMesh* mesh = scene->mMeshes[mesh_index];
        int material_index = mesh->mMaterialIndex;

        CPUMaterial& renderer_material = parsed_scene.materials[material_index];

        // If the mesh is emissive, we're going to add the indices of its faces to the emissive triangles
        // of the scene such that the triangles can be importance sampled (direct lighting estimation / next-event estimation)
        //
        // We are not importance sampling emissive texture so if the mesh has an emissive texture attached, we're
        // not adding its triangles to the list of emissive triangles
        bool is_mesh_emissive = renderer_material.is_emissive() && !renderer_material.emissive_texture_used;

        if (is_mesh_emissive)
        {
            for (int face_index = 0; face_index < mesh->mNumFaces; face_index++, current_triangle_index++)
                // Pushing the index of the current triangle if we're looping on an emissive mesh
                parsed_scene.emissive_triangle_indices.push_back(current_triangle_index);
        }
        else
            current_triangle_index += mesh->mNumFaces;
    }
}

//void ThreadFunctions::load_scene_parse_full_opaque_materials(const aiScene* scene, Scene& parsed_scene)
//{
//    // Looping over all the materials and setting the opaque flags for the materials 
//    // that don't have a base color texture and that are fully opaque (opacity == 1.0f)
//    //
//    // We're only interested in the materials that don't have a base color texture here because
//    // materials that  do have a base color texture have already their opaque flag set when the base
//    // color texture was read from disk (function 'load_scene_texture')
//    for (int i = 0; i < parsed_scene.materials.size(); i++)
//    {
//        CPUMaterial& material = parsed_scene.materials[i];
//
//        parsed_scene.material_has_opaque_base_color_texture[i] = true;
//        if (material.base_color_texture_index != MaterialUtils::NO_TEXTURE)
//            // The material has a texture so checking if it is fully opaque or not
//            parsed_scene.material_has_opaque_base_color_texture[i] = parsed_scene.textures[material.base_color_texture_index].is_fully_opaque();
//    }
//}

void ThreadFunctions::read_envmap(Image32Bit& hdr_image_out, const std::string& filepath, int wanted_channel_count, bool flip_Y)
{
    if (filepath.ends_with(".hdr"))
        hdr_image_out = Image32Bit::read_image_hdr(filepath, wanted_channel_count, flip_Y);
    else if (filepath.ends_with(".exr"))
        hdr_image_out = Image32Bit::read_image_exr(filepath, flip_Y);
}
