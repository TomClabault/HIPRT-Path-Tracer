/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Scene/SceneParser.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadState.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/matrix_decompose.hpp"

#include <chrono>
#include <memory>

void SceneParser::parse_scene_file(const std::string& scene_filepath, Scene& parsed_scene, SceneParserOptions& options)
{
    Assimp::Importer importer;
    const aiScene* scene;

    scene = importer.ReadFile(scene_filepath, aiPostProcessSteps::aiProcess_PreTransformVertices | aiPostProcessSteps::aiProcess_Triangulate);
    if (scene == nullptr)
    {
        std::cerr << importer.GetErrorString() << std::endl;

        int charac = std::getchar();
        std::exit(1);
    }

    std::vector<std::pair<aiTextureType, std::string>> texture_paths;
    // Indices of the texture used by a material
    std::vector<ParsedMaterialTextureIndices> material_texture_indices;
    // Index of the material associated with the texutre
    std::vector<int> material_indices;
    // How many textures are used per mesh. This is used later when parsing the geometry
    std::vector<int> texture_per_mesh;
    // By how much to offset the indices of the textures used by a material.
    // For example, if there are 5 materials in the scene that all use a different base color
    // texture, after the callm to prepare_textures(), they will all have 0 as the index of their
    // base color texture. This is obviously wrong and it should be 0, 1, 2, 3, 4 for
    // each material since they use their own texture. This is what this vector is for, it contains
    // the offsets that are going to be used so that each material has proper texture indices
    std::vector<int> texture_indices_offsets;
    int texture_count;

    prepare_textures(scene, texture_paths, material_texture_indices, material_indices, texture_per_mesh, texture_indices_offsets, texture_count);
    parsed_scene.materials.resize(texture_per_mesh.size());
    parsed_scene.material_names.resize(texture_per_mesh.size());
    parsed_scene.textures.resize(texture_count);
    parsed_scene.textures_dims.resize(texture_count);
    dispatch_texture_loading(parsed_scene, scene_filepath, options.nb_texture_threads, texture_paths);
    assign_material_texture_indices(parsed_scene.materials, material_texture_indices, texture_indices_offsets);

    parse_camera(scene, parsed_scene, options.override_aspect_ratio);

    // If the scene contains multiple meshes, each mesh will have
    // its vertices indices starting at 0. We don't want that.
    // We want indices to be continuously growing (because we don't want
    // the second mesh (with indices starting at 0, i.e its own indices) to use
    // the vertices of the first mesh that have been parsed (and that use indices 0!)
    // The offset thus offsets the indices of the meshes that come after the first one
    // to account for all the indices of the previously parsed meshes
    int global_indices_offset = 0;
    for (int mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
    {
        aiMesh* mesh = scene->mMeshes[mesh_index];
        aiMaterial* mesh_material = scene->mMaterials[mesh->mMaterialIndex];
        std::string material_name = std::string(mesh_material->GetName().C_Str());

        RendererMaterial& renderer_material = parsed_scene.materials[mesh_index];
        read_material_properties(mesh_material, renderer_material);

        //Adding the material to the parsed scene
        bool is_mesh_emissive = renderer_material.is_emissive();

        parsed_scene.material_names[mesh_index] = material_name;
        // Inserting the normals if present
        if (mesh->HasNormals())
            parsed_scene.vertex_normals.insert(parsed_scene.vertex_normals.end(),
                reinterpret_cast<float3*>(mesh->mNormals),
                reinterpret_cast<float3*>(&mesh->mNormals[mesh->mNumVertices]));
        else
            parsed_scene.vertex_normals.insert(parsed_scene.vertex_normals.end(), mesh->mNumVertices, hiprtFloat3{0, 0, 0});

        // Inserting texcoords if present, looking at set 0 because that's where "classical" texcoords are.
        // Other sets are assumed not interesting here.
        if (mesh->HasTextureCoords(0) && texture_per_mesh[mesh_index] > 0)
            for (int i = 0; i < mesh->mNumVertices; i++)
                parsed_scene.texcoords.push_back(make_float2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y));
        else
            parsed_scene.texcoords.insert(parsed_scene.texcoords.end(), mesh->mNumVertices, float2{0.0f, 0.0f});

        // Inserting 0 or 1 depending on whether the normals are present or not.
        // These values will be used in the shader to determine whether we should do
        // smooth shading or not
        parsed_scene.has_vertex_normals.insert(parsed_scene.has_vertex_normals.end(), mesh->mNumVertices, mesh->HasNormals());

        // Inserting all the vertices of the mesh
        parsed_scene.vertices_positions.insert(parsed_scene.vertices_positions.end(), 
            reinterpret_cast<hiprtFloat3*>(&mesh->mVertices[0]), 
            reinterpret_cast<hiprtFloat3*>(&mesh->mVertices[mesh->mNumVertices]));
        
        int max_mesh_index_offset = 0;
        for (int face_index = 0; face_index < mesh->mNumFaces; face_index++)
        {
            aiFace face = mesh->mFaces[face_index];

            int index_1 = face.mIndices[0];
            int index_2 = face.mIndices[1];
            int index_3 = face.mIndices[2];

            // Accumulating the maximum index of this mesh, this is to know
            max_mesh_index_offset = std::max(max_mesh_index_offset, std::max(index_1, std::max(index_2, index_3)));

            parsed_scene.triangle_indices.push_back(index_1 + global_indices_offset);
            parsed_scene.triangle_indices.push_back(index_2 + global_indices_offset);
            parsed_scene.triangle_indices.push_back(index_3 + global_indices_offset);

            // If the face that we just pushed in the triangle buffer is emissive,
            // we're going to add its index to the emissive triangles buffer
            if (is_mesh_emissive)
                parsed_scene.emissive_triangle_indices.push_back(parsed_scene.triangle_indices.size() / 3 - 1);
        }

        // We're pushing the same material index for all the faces of this mesh
        // because all faces of a mesh have the same material (that's how ASSIMP importer's
        // do things internally). An ASSIMP mesh is basically a set of faces that all have the
        // same material.
        // If you're importing the 3D model of a car, even though you probably think of it as only one "3D mesh",
        // ASSIMP sees it as composed of as many meshes as there are different materials
        parsed_scene.material_indices.insert(parsed_scene.material_indices.end(), mesh->mNumFaces, mesh_index);

        // If the max index of the mesh was 19, we want the next to start
        // at 20, not 19, so we ++
        max_mesh_index_offset++;
        // Adding the maximum index of the mesh to our global indices offset 
        global_indices_offset += max_mesh_index_offset;
    }

    std::cout << "\t" << parsed_scene.vertices_positions.size() << " vertices" << std::endl;
    std::cout << "\t" << parsed_scene.triangle_indices.size() / 3 << " triangles" << std::endl;
    std::cout << "\t" << parsed_scene.emissive_triangle_indices.size() << " emissive triangles" << std::endl;
    std::cout << "\t" << parsed_scene.materials.size() << " materials" << std::endl;
    std::cout << "\t" << parsed_scene.textures.size() << " textures" << std::endl;
}

void SceneParser::parse_camera(const aiScene* scene, Scene& parsed_scene, float frame_aspect_override)
{
    // Taking the first camera as the camera of the scene
    if (scene->mNumCameras > 0)
    {
        aiCamera* camera = scene->mCameras[0];

        glm::vec3 camera_position = *reinterpret_cast<glm::vec3*>(&camera->mPosition);
        glm::vec3 camera_lookat = *reinterpret_cast<glm::vec3*>(&camera->mLookAt);
        glm::vec3 camera_up = *reinterpret_cast<glm::vec3*>(&camera->mUp);

        glm::mat4x4 lookat = glm::inverse(glm::lookAt(camera_position, camera_lookat, camera_up));

        glm::vec3 scale, skew, translation;
        glm::vec4 perspective;
        glm::quat orientation;
        glm::decompose(lookat, scale, orientation, translation, skew, perspective);

        parsed_scene.camera.translation = translation;
        parsed_scene.camera.rotation = orientation;

        float aspect_ratio = frame_aspect_override == -1 ? camera->mAspect : frame_aspect_override;
        float vertical_fov = 2.0f * std::atan(std::tan(camera->mHorizontalFOV / 2.0f) * aspect_ratio) + 0.425f;
        parsed_scene.camera.projection_matrix = glm::transpose(glm::perspective(vertical_fov, aspect_ratio, camera->mClipPlaneNear, camera->mClipPlaneFar));
        parsed_scene.camera.vertical_fov = vertical_fov;
        parsed_scene.camera.near_plane = camera->mClipPlaneNear;
        parsed_scene.camera.far_plane = camera->mClipPlaneFar;
    }
    else
    {
        glm::mat4x4 lookat = glm::inverse(glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0)));

        glm::vec3 scale, skew, translation;
        glm::vec4 perspective;
        glm::quat orientation;
        glm::decompose(lookat, scale, orientation, translation, skew, perspective);

        parsed_scene.camera.translation = translation;
        parsed_scene.camera.rotation = orientation;

        float aspect_ratio = 1280.0f / 720.0f;
        float horizontal_fov = 40.0f / 180 * M_PI;
        float vertical_fov = 2.0f * std::atan(std::tan(horizontal_fov / 2.0f) * aspect_ratio) + 0.425f;
        parsed_scene.camera.projection_matrix = glm::transpose(glm::perspective(vertical_fov, aspect_ratio, 0.1f, 100.0f));
        parsed_scene.camera.vertical_fov = vertical_fov;
        parsed_scene.camera.near_plane = 0.1f;
        parsed_scene.camera.far_plane = 100.0f;
    }
}

void SceneParser::prepare_textures(const aiScene* scene, std::vector<std::pair<aiTextureType, std::string>>& texture_paths, std::vector<ParsedMaterialTextureIndices>& material_texture_indices, std::vector<int>& material_indices, std::vector<int>& texture_per_mesh, std::vector<int>& texture_indices_offsets, int& texture_count)
{
    std::vector<std::pair<aiTextureType, std::string>> mesh_texture_paths;
    int global_texture_index_offset = 0;

    for (int mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
    {
        aiMesh* mesh = scene->mMeshes[mesh_index];
        aiMaterial* mesh_material = scene->mMaterials[mesh->mMaterialIndex];
        ParsedMaterialTextureIndices tex_indices;

        // Reading the paths of the textures of the mesh
        mesh_texture_paths = get_textures_paths_and_indices(mesh_material, tex_indices);
        mesh_texture_paths = normalize_texture_paths(mesh_texture_paths);

        int mesh_texture_count = mesh_texture_paths.size();

        material_indices.insert(material_indices.end(), mesh_texture_count, mesh_index);
        material_texture_indices.push_back(tex_indices);
        texture_paths.insert(texture_paths.end(), mesh_texture_paths.begin(), mesh_texture_paths.end());
        texture_per_mesh.push_back(mesh_texture_count);
        texture_indices_offsets.push_back(global_texture_index_offset);

        global_texture_index_offset += mesh_texture_count;
    }

    texture_count = texture_paths.size();
}

void SceneParser::assign_material_texture_indices(std::vector<RendererMaterial>& materials, const std::vector<ParsedMaterialTextureIndices>& material_tex_indices, const std::vector<int>& material_textures_offsets)
{
    for (int material_index = 0; material_index < material_tex_indices.size(); material_index++)
    {
        ParsedMaterialTextureIndices mat_tex_indices = material_tex_indices[material_index];
        RendererMaterial& renderer_material = materials[material_index];
        int tex_index_offset = material_textures_offsets[material_index];

        // Assigning
        renderer_material.base_color_texture_index = mat_tex_indices.base_color_texture_index;
        renderer_material.emission_texture_index = mat_tex_indices.emission_texture_index;
        renderer_material.roughness_texture_index = mat_tex_indices.roughness_texture_index;
        renderer_material.metallic_texture_index = mat_tex_indices.metallic_texture_index;
        renderer_material.roughness_metallic_texture_index = mat_tex_indices.roughness_metallic_texture_index;
        renderer_material.specular_texture_index = mat_tex_indices.specular_texture_index;
        renderer_material.clearcoat_texture_index = mat_tex_indices.clearcoat_texture_index;
        renderer_material.sheen_texture_index = mat_tex_indices.sheen_texture_index;
        renderer_material.specular_transmission_texture_index = mat_tex_indices.specular_transmission_texture_index;
        renderer_material.normal_map_texture_index = mat_tex_indices.normal_map_texture_index;

        // Offsetting
        renderer_material.base_color_texture_index += renderer_material.base_color_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.emission_texture_index += renderer_material.emission_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.roughness_texture_index += renderer_material.roughness_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.metallic_texture_index += renderer_material.metallic_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.roughness_metallic_texture_index += renderer_material.roughness_metallic_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.specular_texture_index += renderer_material.specular_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.clearcoat_texture_index += renderer_material.clearcoat_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.sheen_texture_index += renderer_material.sheen_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.specular_transmission_texture_index += renderer_material.specular_transmission_texture_index == -1 ? 0 : tex_index_offset;
        renderer_material.normal_map_texture_index += renderer_material.normal_map_texture_index == -1 ? 0 : tex_index_offset;
    }
}

void SceneParser::dispatch_texture_loading(Scene& parsed_scene, const std::string& scene_path, int nb_threads, const std::vector<std::pair<aiTextureType, std::string>>& texture_paths)
{
    if (nb_threads == -1)
        // As many threads as there are textures if -1 was given
        nb_threads = texture_paths.size();

    // Creating a state to keep the data that the threads need alive
    std::shared_ptr<TextureLoadingThreadState> texture_threads_state = std::make_shared<TextureLoadingThreadState>();
    texture_threads_state->scene_filepath = scene_path;
    texture_threads_state->texture_paths = texture_paths;

    ThreadManager::add_state(ThreadManager::TEXTURE_THREADS_KEY, texture_threads_state);

    for (int i = 0; i < nb_threads; i++)
        ThreadManager::start_thread(ThreadManager::TEXTURE_THREADS_KEY, ThreadFunctions::load_texture, std::ref(parsed_scene), texture_threads_state->scene_filepath, std::ref(texture_threads_state->texture_paths), i, nb_threads);
}

void SceneParser::read_material_properties(aiMaterial* mesh_material, RendererMaterial& renderer_material)
{
    //Getting the properties that are going to be used by the materials
    //of the application

    renderer_material.brdf_type = BRDF::Disney;

    aiReturn error_code_emissive;
    mesh_material->Get(AI_MATKEY_COLOR_DIFFUSE, *((aiColor3D*)&renderer_material.base_color));
    mesh_material->Get(AI_MATKEY_COLOR_EMISSIVE, *((aiColor3D*)&renderer_material.emission));
    mesh_material->Get(AI_MATKEY_METALLIC_FACTOR, renderer_material.metallic);
    mesh_material->Get(AI_MATKEY_ROUGHNESS_FACTOR, renderer_material.roughness);
    mesh_material->Get(AI_MATKEY_ANISOTROPY_FACTOR, renderer_material.anisotropic);
    if (!mesh_material->Get(AI_MATKEY_SHEEN_COLOR_FACTOR, *((aiColor3D*)&renderer_material.sheen_color)))
        // We did get sheen color from the parsed scene, assuming sheen is on 100%, can't do better
        renderer_material.sheen = renderer_material.sheen_tint = 1.0f;
    if (!mesh_material->Get(AI_MATKEY_SPECULAR_FACTOR, renderer_material.specular))
    {
        // We sucessfully got the specular color so we're going to assume that we the specular and tin are 100%
        renderer_material.specular_tint = 1.0f;
        renderer_material.specular_color = ColorRGB(1.0f);
    }
    mesh_material->Get(AI_MATKEY_CLEARCOAT_FACTOR, renderer_material.clearcoat);
    mesh_material->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, renderer_material.clearcoat_roughness);
    mesh_material->Get(AI_MATKEY_REFRACTI, renderer_material.ior);
    mesh_material->Get(AI_MATKEY_TRANSMISSION_FACTOR, renderer_material.specular_transmission);
    mesh_material->Get(AI_MATKEY_VOLUME_ATTENUATION_COLOR, renderer_material.absorption_color);
    mesh_material->Get(AI_MATKEY_VOLUME_ATTENUATION_DISTANCE, renderer_material.absorption_at_distance);

    if (renderer_material.is_emissive())
    {
        float emission_strength = 1.0f;
        mesh_material->Get(AI_MATKEY_EMISSIVE_INTENSITY, emission_strength);

        renderer_material.emission *= emission_strength;
    }

    renderer_material.make_safe();
    renderer_material.precompute_properties();
}

std::vector<std::pair<aiTextureType, std::string>> SceneParser::get_textures_paths_and_indices(aiMaterial* mesh_material, ParsedMaterialTextureIndices& texture_indices)
{
    std::vector<std::pair<aiTextureType, std::string>> texture_paths;

    texture_indices.base_color_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_BASE_COLOR, texture_paths);
    if (texture_indices.base_color_texture_index == -1)
        // Trying diffuse for some file formats
        // The OBJ format uses DIFFUSE instead of BASE_COLOR
        texture_indices.base_color_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_DIFFUSE, texture_paths);

    texture_indices.emission_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_EMISSION_COLOR, texture_paths);
    int roughness_index = get_first_texture_of_type(mesh_material, aiTextureType_DIFFUSE_ROUGHNESS, texture_paths);
    int metallic_index = get_first_texture_of_type(mesh_material, aiTextureType_METALNESS, texture_paths);
    if (roughness_index != -1 && metallic_index != -1 && texture_paths[roughness_index].second == texture_paths[metallic_index].second)
    {
        // The roughness and metallic textures are the same

        // Poping the metallic path because it's the same as the roughness, we only want one
        // otherwise the texture reader is going to read the same path (and the same texture) twice from disk
        texture_paths.pop_back();
        // Using the roughness index for the roughness + metallic texture
        texture_indices.roughness_metallic_texture_index = roughness_index;
    }
    else
    {
        texture_indices.roughness_texture_index = roughness_index;
        texture_indices.metallic_texture_index = metallic_index;
    }

    texture_indices.specular_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_SPECULAR, texture_paths);
    texture_indices.clearcoat_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_CLEARCOAT, texture_paths);
    texture_indices.sheen_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_SHEEN, texture_paths);
    texture_indices.specular_transmission_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_TRANSMISSION, texture_paths);

    texture_indices.normal_map_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_NORMALS, texture_paths);
    if (texture_indices.normal_map_texture_index == -1)
        // Trying HEIGHT for some file formats
        texture_indices.normal_map_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_HEIGHT, texture_paths);

    return texture_paths;
}

int SceneParser::get_first_texture_of_type(aiMaterial* mesh_material, aiTextureType type, std::vector<std::pair<aiTextureType, std::string>>& texture_path_list)
{
    int tex_count = mesh_material->GetTextureCount(type);
    if (tex_count == 0)
        return -1;
    else
    {
        aiString aiPath;
        mesh_material->GetTexture(type, 0, &aiPath);

        std::string string_path = std::string(aiPath.data);
        if (string_path.empty())
            return -1;

        texture_path_list.push_back(std::make_pair(type, string_path));

        return texture_path_list.size() - 1;
    }
}

std::vector<std::pair<aiTextureType, std::string>> SceneParser::normalize_texture_paths(std::vector<std::pair<aiTextureType, std::string>>& paths)
{
    std::vector<std::pair<aiTextureType, std::string>> normalized_paths;
    normalized_paths.reserve(paths.size());

    for (auto pair : paths)
    {
        size_t find_index = pair.second.find("%20");
        while (find_index != (size_t)-1)
        {
            pair.second = pair.second.replace(find_index, 3, " ");
            find_index = pair.second.find("%20");
        }

        normalized_paths.push_back(pair);
    }

    return normalized_paths;
}

RendererMaterial SceneParser::offset_textures_indices(const RendererMaterial& renderer_material, int offset)
{
    RendererMaterial out_mat = renderer_material;

   out_mat.emission_texture_index += (renderer_material.emission_texture_index == -1) ? 0 : offset;
   out_mat.base_color_texture_index += (renderer_material.base_color_texture_index == -1) ? 0 : offset;

   out_mat.roughness_texture_index += (renderer_material.roughness_texture_index == -1) ? 0 : offset;
   out_mat.oren_sigma_texture_index += (renderer_material.oren_sigma_texture_index == -1) ? 0 : offset;
   out_mat.subsurface_texture_index += (renderer_material.subsurface_texture_index == -1) ? 0 : offset;

   out_mat.metallic_texture_index += (renderer_material.metallic_texture_index == -1) ? 0 : offset;
   out_mat.specular_texture_index += (renderer_material.specular_texture_index == -1) ? 0 : offset;
   out_mat.specular_tint_texture_index += (renderer_material.specular_tint_texture_index == -1) ? 0 : offset;
   out_mat.specular_color_texture_index += (renderer_material.specular_color_texture_index == -1) ? 0 : offset;

   out_mat.anisotropic_texture_index += (renderer_material.anisotropic_texture_index == -1) ? 0 : offset;
   out_mat.anisotropic_rotation_texture_index += (renderer_material.anisotropic_rotation_texture_index == -1) ? 0 : offset;

   out_mat.clearcoat_texture_index += (renderer_material.clearcoat_texture_index == -1) ? 0 : offset;
   out_mat.clearcoat_roughness_texture_index += (renderer_material.clearcoat_roughness_texture_index == -1) ? 0 : offset;
   out_mat.clearcoat_ior_texture_index += (renderer_material.clearcoat_ior_texture_index == -1) ? 0 : offset;

   out_mat.sheen_texture_index += (renderer_material.sheen_texture_index == -1) ? 0 : offset;
   out_mat.sheen_tint_color_texture_index += (renderer_material.sheen_tint_color_texture_index == -1) ? 0 : offset;
   out_mat.sheen_color_texture_index += (renderer_material.sheen_color_texture_index == -1) ? 0 : offset;
    
   out_mat.specular_transmission_texture_index += (renderer_material.specular_transmission_texture_index == -1) ? 0 : offset;

   return out_mat;
}
