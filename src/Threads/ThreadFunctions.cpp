/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernel.h"
#include "Device/includes/Texture.h"
#include "Device/includes/TriangleStructures.h"
#include "Image/Image.h"
#include "Threads/ThreadFunctions.h"

// For replacing backslashes in texture paths
#include <regex>

// For embedded texture access (aiTexture struct and stb_image)
#include "assimp/texture.h"

void ThreadFunctions::compile_kernel(std::shared_ptr<GPUKernel> kernel,
									 std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx,
									 const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	kernel->compile(hiprt_orochi_ctx, func_name_sets, true, false);
}

void ThreadFunctions::compile_kernel_silent(std::shared_ptr<GPUKernel> kernel,
											std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx,
											const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	kernel->compile(hiprt_orochi_ctx, func_name_sets, true, true);
}

void ThreadFunctions::precompile_kernel(const std::string& kernel_function_name,
										const std::string& kernel_filepath,
										GPUKernelCompilerOptions options,
										std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx,
										const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(hiprt_orochi_ctx->orochi_ctx));

	GPUKernel kernel(kernel_filepath, kernel_function_name);
	kernel.set_precompiled(true);
	kernel.get_kernel_options() = options;
	kernel.compile(hiprt_orochi_ctx, func_name_sets, true, true);
}

void ThreadFunctions::load_scene_texture(const aiScene* assimp_scene,
										 Scene& parsed_scene,
										 std::string scene_path,
										 const std::vector<std::pair<aiTextureType, std::string>>& tex_paths,
										 const std::vector<int>& material_indices,
										 int thread_index,
										 int nb_threads)
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
		aiTextureType type	  = tex_paths[thread_index].first;
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
			// TODO we only need 3 channels here but it's tricky to handle 3 channels texture with HIP/CUDA. Supported formats are only 1, 2, 4 channels, not
			// three
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
			// TODO we only need 3 channels here but it's tricky to handle 3 channels texture with HIP/CUDA. Supported formats are only 1, 2, 4 channels, not
			// three
			nb_channels = 4;
			break;

		default:
			nb_channels = 1;
			break;
		}

		Image8Bit texture;
		if (!texture_file_path.empty() && texture_file_path[0] == '*')
		{
			// Embedded texture reference (ASSIMP uses "*0", "*1", etc. as paths for embedded textures)
			int embedded_idx			  = std::stoi(texture_file_path.substr(1));
			const aiTexture* embedded_tex = assimp_scene->mTextures[embedded_idx];
			if (embedded_tex != nullptr)
			{
				if (std::string(embedded_tex->mFilename.C_Str()).contains("Labels_0_BaseColor"))
					printf("\n");

				if (embedded_tex->mHeight == 0)
				{
					// Compressed embedded texture (PNG/JPEG data in memory)
					stbi_set_flip_vertically_on_load_thread(false);

					int w, h, n;
					unsigned char* pixels =
						stbi_load_from_memory(reinterpret_cast<const unsigned char*>(embedded_tex->pcData), embedded_tex->mWidth, &w, &h, &n, nb_channels);

					if (pixels)
					{
						texture = Image8Bit(pixels, w, h, nb_channels);
						stbi_image_free(pixels);
					}
				}
				else
				{
					// Uncompressed embedded texture (raw RGBA8888 aiTexel array)
					if (nb_channels == 4)
					{
						texture = Image8Bit(reinterpret_cast<const unsigned char*>(embedded_tex->pcData), embedded_tex->mWidth, embedded_tex->mHeight, 4);
					}
					else
					{
						// Need to reduce channels from RGBA to nb_channels
						Image8Bit converted(embedded_tex->mWidth, embedded_tex->mHeight, nb_channels);
						int num_pixels				  = embedded_tex->mWidth * embedded_tex->mHeight;
						const unsigned char* src_data = reinterpret_cast<const unsigned char*>(embedded_tex->pcData);
						for (int pixel_i = 0; pixel_i < num_pixels; pixel_i++)
							for (int c = 0; c < nb_channels; c++)
								converted[pixel_i * nb_channels + c] = src_data[pixel_i * 4 + c];

						texture = std::move(converted);
					}
				}
			}
		}
		else
		{
			// External texture file on disk
			if (full_path.contains("Labels_0_BaseColor"))
				printf("\n");

			texture = Image8Bit::read_image(full_path, nb_channels, false);
		}

		int material_index = material_indices[thread_index];
		if (type == aiTextureType_EMISSIVE)
		{
			if (texture.is_constant_color(/* threshold */ 5))
			{
				// The emissive texture is constant color, we can then just not use that texture and use
				// the emission filed of the material to store the emission of the texture
				parsed_scene.materials[material_index].emission_texture_index = MaterialConstants::CONSTANT_EMISSIVE_TEXTURE;

				ColorRGBA32F emission_rgba						= texture.sample_rgba32f(make_float2(0, 0));
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
				unsigned char texture_fully_opaque									= texture.is_fully_opaque() ? 1 : 0;
				parsed_scene.material_has_opaque_base_color_texture[material_index] = texture_fully_opaque;
			}
			parsed_scene.textures[thread_index] = texture;
		}

		thread_index += nb_threads;
	}
}

void ThreadFunctions::load_scene_parse_emissive_triangles(const aiScene* scene, Scene& parsed_scene)
{
	auto start = std::chrono::high_resolution_clock::now();

	int current_triangle_index_in_whole_scene = 0;

	// If the scene contains multiple meshes, each mesh will have
	// its vertices indices starting at 0. We don't want that.
	//
	// We want indices to be continuously growing (because we don't want
	// the second mesh (with indices starting at 0, i.e its own indices) to use
	// the vertices of the first mesh that have been parsed (and that use indices 0!)
	// The offset thus offsets the indices of the meshes that come after the first one
	// to account for all the indices of the previously parsed meshes
	//
	// This is only used for the emissives triangles vertex indices
	int global_indices_offset = 0;

	unsigned int emissive_triangle_index_so_far = 0;

	// Looping over all the meshes
	for (int mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
	{
		aiMesh* mesh	   = scene->mMeshes[mesh_index];
		int material_index = mesh->mMaterialIndex;

		CPUMaterial& renderer_material = parsed_scene.materials[material_index];

		int max_emissive_mesh_index_offset = 0;
		for (int face_index = 0; face_index < mesh->mNumFaces; face_index++, current_triangle_index_in_whole_scene++)
		{
			{
				if (current_triangle_index_in_whole_scene == 3193126)
					std::cerr << "";
			}
			int index_1 = mesh->mFaces[face_index].mIndices[0];
			int index_2 = mesh->mFaces[face_index].mIndices[1];
			int index_3 = mesh->mFaces[face_index].mIndices[2];

			// Accumulating the maximum index of this mesh, this is to know
			max_emissive_mesh_index_offset = std::max(max_emissive_mesh_index_offset, std::max(index_1, std::max(index_2, index_3)));

			if (renderer_material.is_emissive())
			{
				parsed_scene.emissive_triangle_vertex_indices.push_back(index_1 + global_indices_offset);
				parsed_scene.emissive_triangle_vertex_indices.push_back(index_2 + global_indices_offset);
				parsed_scene.emissive_triangle_vertex_indices.push_back(index_3 + global_indices_offset);

				parsed_scene.emissive_triangles_primitive_indices.push_back(current_triangle_index_in_whole_scene);
				parsed_scene.emissive_triangles_primitive_indices_and_emissive_textures.push_back(current_triangle_index_in_whole_scene);

				int emissive_triangle_global_index = current_triangle_index_in_whole_scene;

				int vertex_index_1 = parsed_scene.triangles_vertex_indices[emissive_triangle_global_index * 3 + 0];
				int vertex_index_2 = parsed_scene.triangles_vertex_indices[emissive_triangle_global_index * 3 + 1];
				int vertex_index_3 = parsed_scene.triangles_vertex_indices[emissive_triangle_global_index * 3 + 2];

				float3_t vertex_1 = parsed_scene.vertices_positions[vertex_index_1];
				float3_t vertex_2 = parsed_scene.vertices_positions[vertex_index_2];
				float3_t vertex_3 = parsed_scene.vertices_positions[vertex_index_3];

				// Using the triangle class to easily compute the area of the triangle
				float3_t face_normal = hippt::cross(vertex_2 - vertex_1, vertex_3 - vertex_1);
				float face_area		 = hippt::length(face_normal) * 0.5f;
				float face_emission	 = 0.0f;

				if (renderer_material.uses_non_constant_emissive_texture())
				{
					// If the mesh has an emissive texture, we're going to approximately integrate the emissive texture power of each face of the mesh for light
					// sampling
					// and store that in the emission parameter of the material
					constexpr unsigned int sample_u_count = 16;
					constexpr unsigned int sample_v_count = 16;

					for (unsigned int sample_u = 0; sample_u < sample_u_count; sample_u++)
					{
						for (unsigned int sample_v = 0; sample_v < sample_v_count; sample_v++)
						{
							float u = (sample_u + 0.5f) / static_cast<float>(sample_u_count);
							float v = (sample_v + 0.5f) / static_cast<float>(sample_v_count);

							TriangleIndices triangle_vertex_indices = { vertex_index_1, vertex_index_2, vertex_index_3 };
							TriangleTexcoords triangle_texcoords	= load_triangle_texcoords(parsed_scene.texcoords.data(), triangle_vertex_indices);

							float2_t texcoords = uv_interpolate(triangle_texcoords, make_float2(u, v));

							ColorRGBA32F emission_rgba = parsed_scene.textures[renderer_material.emission_texture_index].sample_rgba32f(texcoords);
							ColorRGB32F emission_rgb(emission_rgba.r, emission_rgba.g, emission_rgba.b);

							face_emission += emission_rgb.luminance() / (static_cast<float>(sample_u_count) * static_cast<float>(sample_v_count));
						}
					}
				}
				else
					face_emission = renderer_material.get_total_emission().luminance();

				float face_power = face_area * face_emission;

				parsed_scene.triangles_average_emissive_luminance.push_back(face_emission);
				parsed_scene.triangles_average_emissive_power_luminance.push_back(face_power);
			}
			else
			{
				parsed_scene.triangles_average_emissive_luminance.push_back(0.0f);		 // Non emissive triangle, average emission is 0
				parsed_scene.triangles_average_emissive_power_luminance.push_back(0.0f); // Non emissive triangle, emissive power is 0
			}
		}

		global_indices_offset += max_emissive_mesh_index_offset + 1; // +1 because the indices start at 0 but 0 is already 1 index on its own so we need + 1
	}

	auto stop = std::chrono::high_resolution_clock::now();

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Parsed emissive triangles in %ldms",
							std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void ThreadFunctions::load_scene_compute_triangle_areas(Scene& parsed_scene)
{
	int number_of_triangles = parsed_scene.triangles_vertex_indices.size() / 3;

	parsed_scene.triangle_areas.resize(number_of_triangles);

#pragma omp parallel for
	for (int triangle_index = 0; triangle_index < number_of_triangles; triangle_index++)
	{
		float3_t vertex_A = parsed_scene.vertices_positions[parsed_scene.triangles_vertex_indices[triangle_index * 3 + 0]];
		float3_t vertex_B = parsed_scene.vertices_positions[parsed_scene.triangles_vertex_indices[triangle_index * 3 + 1]];
		float3_t vertex_C = parsed_scene.vertices_positions[parsed_scene.triangles_vertex_indices[triangle_index * 3 + 2]];

		float3_t AB = vertex_B - vertex_A;
		float3_t AC = vertex_C - vertex_A;

		float3_t normal		= hippt::cross(AB, AC);
		float length_normal = hippt::length(normal);
		float area			= hippt::length(normal) * 0.5f;

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
