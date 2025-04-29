/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Compiler/GPUKernel.h"
#include "Threads/ThreadFunctions.h"

 // For replacing backslashes in texture paths
#include <regex>

void ThreadFunctions::compile_kernel(std::shared_ptr<GPUKernel> kernel, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
    kernel->compile(hiprt_orochi_ctx, func_name_sets, true, false);
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
    // Starting with the .GLTF/.OBJ/.whatever-scene-format file
    corrected_filepath = scene_path;
    // Removing the name of the .GLTF / .OBJ / .XXX file by looking at the *last* '/' or '\'
    if (corrected_filepath.find('/') != std::string::npos)
        corrected_filepath = corrected_filepath.substr(0, corrected_filepath.rfind('/') + 1);
    else if (corrected_filepath.find('\\') != std::string::npos)
        corrected_filepath = corrected_filepath.substr(0, corrected_filepath.rfind('\\') + 1);
    // Converting the path to absolute
    corrected_filepath = std::filesystem::absolute(corrected_filepath).string();
    // Replacing backslashes by forward slashes
    corrected_filepath = std::regex_replace(corrected_filepath, std::regex("\\\\"), "/"); // replace 'def' -> 'klm'

    // While loop here so that a single thread can parse multiple textures
    while (thread_index < parsed_scene.textures.size())
    {
        // Taking the name of the texture
        std::string texture_file_path = tex_paths[thread_index].second;
        // Adding the name of the texture to the absolute path of the scene file such that
        // we're looking for textures next to the GLTF file
        std::string full_path = corrected_filepath + texture_file_path;
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
            if (parsed_scene.materials[material_indices[thread_index]].roughness_metallic_texture_index != MaterialConstants::NO_TEXTURE)
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
                parsed_scene.materials[material_index].emission_texture_index = MaterialConstants::CONSTANT_EMISSIVE_TEXTURE;

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
    g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, " -------------------- Parsing emissive triangles -------------------");

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

void ThreadFunctions::load_scene_compute_triangle_areas(Scene& parsed_scene)
{
    g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, " -------------------- Compute triangle areas is running -------------------");

	int number_of_triangles = parsed_scene.triangles_indices.size() / 3;

	parsed_scene.triangle_areas.resize(number_of_triangles);

    for (int triangle_index = 0; triangle_index < number_of_triangles; triangle_index++)
    {
        float3 vertex_A = parsed_scene.vertices_positions[parsed_scene.triangles_indices[triangle_index * 3 + 0]];
        float3 vertex_B = parsed_scene.vertices_positions[parsed_scene.triangles_indices[triangle_index * 3 + 1]];
        float3 vertex_C = parsed_scene.vertices_positions[parsed_scene.triangles_indices[triangle_index * 3 + 2]];

        float3 AB = vertex_B - vertex_A;
        float3 AC = vertex_C - vertex_A;

        float3 normal = hippt::cross(AB, AC);
        float length_normal = hippt::length(normal);
        float area;
   //     if (length_normal <= 1.0e-6f || hippt::is_nan(length_normal))
			//// Degenerate triangle, setting area to 0
   //         area = 0.0f;
   //     else
            area = hippt::length(normal) * 0.5f;

        parsed_scene.triangle_areas[triangle_index] = area;
    }
}

void ThreadFunctions::read_envmap(Image32Bit& hdr_image_out, const std::string& filepath, int wanted_channel_count, bool flip_Y)
{
    if (filepath.ends_with(".hdr"))
        hdr_image_out = Image32Bit::read_image_hdr(filepath, wanted_channel_count, flip_Y);
    else if (filepath.ends_with(".exr"))
        hdr_image_out = Image32Bit::read_image_exr(filepath, flip_Y);

    if (hdr_image_out.width == 0 || hdr_image_out.height == 0)
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_WARNING, "Could not read envmap file: %s", filepath.c_str());
}
