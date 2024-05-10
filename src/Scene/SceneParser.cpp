/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Scene/SceneParser.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/matrix_decompose.hpp"

#include <chrono>

Scene SceneParser::parse_scene_file(const std::string& filepath, float frame_aspect_override)
{
    Scene parsed_scene;

    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(filepath, aiPostProcessSteps::aiProcess_PreTransformVertices | aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_FixInfacingNormals);
    if (scene == nullptr)
    {
        std::cerr << importer.GetErrorString() << std::endl;

        int charac = std::getchar();
        std::exit(1);
    }

    // Taking the first camera as the camera of the scene
    if (scene->mNumCameras > 0)
    {
        aiCamera* camera = scene->mCameras[0];

        glm::vec3 camera_position = *reinterpret_cast<glm::vec3*>(&camera->mPosition);
        glm::vec3 camera_lookat = *reinterpret_cast<glm::vec3*>(&camera->mLookAt);
        glm::vec3 camera_up = *reinterpret_cast<glm::vec3*>(&camera->mUp);

        // We need to inverse to view matrix here, not sure why
        // TODO investigate why the perspective matrix multiplied by a point using matrix_X_point
        // gives a different result than the point multiplied by the matrix using the glm * operator
        // This may be why we need to transpose the perspective matrix below and this may also explain
        // why we need to inverse the view matrix below
        glm::mat4x4 lookat = glm::inverse(glm::lookAt(camera_position, camera_lookat, camera_up));

        glm::vec3 scale, skew, translation;
        glm::vec4 perspective;
        glm::quat orientation;
        glm::decompose(lookat, scale, orientation, translation, skew, perspective);

        parsed_scene.camera.translation = translation;
        parsed_scene.camera.rotation = orientation;

        // TODO + 0.425f is here to correct the FOV from a GLTF Blender export. After the export, 
        // the scene in the renderer is view as if the FOV was smaller. We're correcting this by adding a fix
        // +0.425 to try and get to same view as in Blender. THIS PROBABLY SHOULDN'T BE HERE AS THIS IS
        // VERY SUS
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

        // TODO + 0.425f is here to correct the FOV from a GLTF Blender export. After the export, 
        // the scene in the renderer is view as if the FOV was smaller. We're correcting this by adding a fix
        // +0.425 to try and get to same view as in Blender. THIS PROBABLY SHOULDN'T BE HERE AS THIS IS
        // VERY SUS
        float aspect_ratio = 1280.0f / 720.0f;
        float horizontal_fov = 40.0f / 180 * M_PI;
        float vertical_fov = 2.0f * std::atan(std::tan(horizontal_fov / 2.0f) * aspect_ratio) + 0.425f;
        parsed_scene.camera.projection_matrix = glm::transpose(glm::perspective(vertical_fov, aspect_ratio, 0.1f, 100.0f));
        parsed_scene.camera.vertical_fov = vertical_fov;
        parsed_scene.camera.near_plane = 0.1f;
        parsed_scene.camera.far_plane = 100.0f;
    }

    // If the scene contains multiple meshes, each mesh will have
    // its vertices indices starting at 0. We don't want that.
    // We want indices to be continuously growing (because we don't want
    // the second mesh (with indices starting at 0, i.e its own indices) to use
    // the vertices of the first mesh that have been parsed (and that use indices 0!)
    // The offset thus offsets the indices of the meshes that come after the first one
    // to account for all the indices of the previously parsed meshes
    int global_indices_offset = 0;
    // Same thing for the textures
    int global_texture_indices_offset = 0;
    for (int mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
    {
        int max_mesh_index_offset = 0;

        aiMesh* mesh = scene->mMeshes[mesh_index];
        aiMaterial* mesh_material = scene->mMaterials[mesh->mMaterialIndex];

        RendererMaterial renderer_material = read_material_properties(mesh_material);
        std::vector<std::pair<aiTextureType, std::string>> texture_paths = get_textures_paths(mesh_material, renderer_material);
        normalize_texture_paths(texture_paths);
        std::vector<ImageRGBA> textures = read_textures(filepath, texture_paths);
        parsed_scene.textures.insert(parsed_scene.textures.end(), textures.begin(), textures.end());
        std::transform(texture_paths.begin(), texture_paths.end(), std::back_inserter(parsed_scene.textures_is_srgb), 
            [](const std::pair<aiTextureType, std::string>& pair) {return pair.first == aiTextureType_BASE_COLOR || pair.first == aiTextureType_NORMALS; });
        offset_textures_indices(renderer_material, global_texture_indices_offset);
        global_texture_indices_offset += textures.size();

        //Adding the material to the parsed scene
        parsed_scene.materials.push_back(renderer_material);
        int material_index = parsed_scene.materials.size() - 1;
        bool is_mesh_emissive = renderer_material.is_emissive();

        // Inserting the normals if present
        if (mesh->HasNormals())
            parsed_scene.vertex_normals.insert(parsed_scene.vertex_normals.end(),
                reinterpret_cast<float3*>(mesh->mNormals),
                reinterpret_cast<float3*>(&mesh->mNormals[mesh->mNumVertices]));
        else
            parsed_scene.vertex_normals.insert(parsed_scene.vertex_normals.end(), mesh->mNumVertices, hiprtFloat3{0, 0, 0});

        // Inserting texcoords if present
        if (mesh->HasTextureCoords(0))
        {
            for (int i = 0; i < mesh->mNumVertices; i++)
            {
                aiVector3D texcoord_3D = mesh->mTextureCoords[0][i];
                parsed_scene.texcoords.push_back(make_float2(texcoord_3D.x, texcoord_3D.y));
            }
        }

        // Inserting 0 or 1 depending on whether the normals are present or not.
        // These values will be used in the shader to determine whether we should do
        // smooth shading or not
        parsed_scene.has_vertex_normals.insert(parsed_scene.has_vertex_normals.end(), mesh->mNumVertices, mesh->HasNormals());

        // Inserting all the vertices of the mesh
        parsed_scene.vertices_positions.insert(parsed_scene.vertices_positions.end(), 
            reinterpret_cast<hiprtFloat3*>(&mesh->mVertices[0]), 
            reinterpret_cast<hiprtFloat3*>(&mesh->mVertices[mesh->mNumVertices]));

        for (int face_index = 0; face_index < mesh->mNumFaces; face_index++)
        {
            aiFace face = mesh->mFaces[face_index];

            int index_1 = face.mIndices[0];
            int index_2 = face.mIndices[1];
            int index_3 = face.mIndices[2];

            // Accumulating the maximum index of this mesh
            max_mesh_index_offset = std::max(max_mesh_index_offset, std::max(index_1, std::max(index_2, index_3)));

            parsed_scene.triangle_indices.push_back(index_1 + global_indices_offset);
            parsed_scene.triangle_indices.push_back(index_2 + global_indices_offset);
            parsed_scene.triangle_indices.push_back(index_3 + global_indices_offset);

            // If the face that we just pushed in the triangle buffer is emissive,
            // we're going to add its index to the emissive triangles buffer
            if (is_mesh_emissive)
                parsed_scene.emissive_triangle_indices.push_back(parsed_scene.triangle_indices.size() / 3 - 1);

            // We're pushing the same material index for all the faces of this mesh
            // because all faces of a mesh have the same material (that's how ASSIMP importer's
            // do things internally). An ASSIMP mesh is basically a set of faces that all have the
            // same material.
            // If you're importing the 3D model of a car, even though you probably think of it as only one "3D mesh",
            // ASSIMP sees it as composed of as many meshes as there are different materials
            parsed_scene.material_indices.push_back(material_index);
        }

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

    return parsed_scene;
}

RendererMaterial SceneParser::read_material_properties(aiMaterial* mesh_material)
{
    //Getting the properties that are going to be used by the materials
    //of the application
    RendererMaterial renderer_material;
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

    if (renderer_material.is_emissive())
    {
        float emission_strength = 1.0f;
        mesh_material->Get(AI_MATKEY_EMISSIVE_INTENSITY, emission_strength);

        renderer_material.emission *= emission_strength;
    }

    renderer_material.make_safe();
    renderer_material.precompute_properties();

    return renderer_material;
}

std::vector<std::pair<aiTextureType, std::string>> SceneParser::get_textures_paths(aiMaterial* mesh_material, RendererMaterial& renderer_material)
{
    std::vector<std::pair<aiTextureType, std::string>> texture_paths;

    renderer_material.base_color_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_BASE_COLOR, texture_paths);
    renderer_material.emission_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_EMISSION_COLOR, texture_paths);
    int roughness_index = get_first_texture_of_type(mesh_material, aiTextureType_DIFFUSE_ROUGHNESS, texture_paths);
    int metallic_index = get_first_texture_of_type(mesh_material, aiTextureType_METALNESS, texture_paths);
    if (roughness_index != -1 && metallic_index != -1 && texture_paths[roughness_index].second == texture_paths[metallic_index].second)
    {
        // The roughness and metallic textures are the same

        // Poping the metallic path because it's the same as the roughness, we only want one
        // otherwise the texture reader is going to read the same path (and the same texture) twice from disk
        texture_paths.pop_back();
        // Using the roughness index for the roughness + metallic texture
        renderer_material.roughnes_metallic_texture_index = roughness_index;
    }
    renderer_material.specular_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_SPECULAR, texture_paths);
    renderer_material.clearcoat_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_CLEARCOAT, texture_paths);
    renderer_material.sheen_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_SHEEN, texture_paths);
    renderer_material.specular_transmission_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_TRANSMISSION, texture_paths);

    renderer_material.normal_map_texture_index = get_first_texture_of_type(mesh_material, aiTextureType_NORMALS, texture_paths);

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
        texture_path_list.push_back(std::make_pair(type, std::string(aiPath.data)));

        return texture_path_list.size() - 1;
    }
}

void SceneParser::normalize_texture_paths(std::vector<std::pair<aiTextureType, std::string>>& paths)
{
    for (auto& pair : paths)
    {
        size_t find_index = pair.second.find("%20");
        while (find_index != (size_t)-1)
        {
            pair.second = pair.second.replace(find_index, 3, " ");
            find_index = pair.second.find("%20");
        }
    }
}

std::vector<ImageRGBA> SceneParser::read_textures(const std::string& filepath, const std::vector<std::pair<aiTextureType, std::string>>& texture_paths)
{
    // Preparing the filepath so that it's ready to be appended with the texture name
    std::string corrected_filepath = filepath;
    corrected_filepath = corrected_filepath.substr(0, corrected_filepath.rfind('/') + 1);

    std::vector<ImageRGBA> images(texture_paths.size());

//#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < texture_paths.size(); i++)
    {
        std::string texture_path = corrected_filepath + texture_paths[i].second;
        images[i] = ImageRGBA::read_image(texture_path, false);
    }

    return images;
}

void SceneParser::offset_textures_indices(RendererMaterial& renderer_material, int offset)
{
    renderer_material.emission_texture_index += (renderer_material.emission_texture_index == -1) ? 0 : offset;
    renderer_material.base_color_texture_index += (renderer_material.base_color_texture_index == -1) ? 0 : offset;

    renderer_material.roughness_texture_index += (renderer_material.roughness_texture_index == -1) ? 0 : offset;
    renderer_material.oren_sigma_texture_index += (renderer_material.oren_sigma_texture_index == -1) ? 0 : offset;
    renderer_material.subsurface_texture_index += (renderer_material.subsurface_texture_index == -1) ? 0 : offset;

    renderer_material.metallic_texture_index += (renderer_material.metallic_texture_index == -1) ? 0 : offset;
    renderer_material.specular_texture_index += (renderer_material.specular_texture_index == -1) ? 0 : offset;
    renderer_material.specular_tint_texture_index += (renderer_material.specular_tint_texture_index == -1) ? 0 : offset;
    renderer_material.specular_color_texture_index += (renderer_material.specular_color_texture_index == -1) ? 0 : offset;

    renderer_material.anisotropic_texture_index += (renderer_material.anisotropic_texture_index == -1) ? 0 : offset;
    renderer_material.anisotropic_rotation_texture_index += (renderer_material.anisotropic_rotation_texture_index == -1) ? 0 : offset;

    renderer_material.clearcoat_texture_index += (renderer_material.clearcoat_texture_index == -1) ? 0 : offset;
    renderer_material.clearcoat_roughness_texture_index += (renderer_material.clearcoat_roughness_texture_index == -1) ? 0 : offset;
    renderer_material.clearcoat_ior_texture_index += (renderer_material.clearcoat_ior_texture_index == -1) ? 0 : offset;

    renderer_material.sheen_texture_index += (renderer_material.sheen_texture_index == -1) ? 0 : offset;
    renderer_material.sheen_tint_color_texture_index += (renderer_material.sheen_tint_color_texture_index == -1) ? 0 : offset;
    renderer_material.sheen_color_texture_index += (renderer_material.sheen_color_texture_index == -1) ? 0 : offset;
    
    renderer_material.ior_texture_index += (renderer_material.ior_texture_index == -1) ? 0 : offset;
    renderer_material.specular_transmission_texture_index += (renderer_material.specular_transmission_texture_index == -1) ? 0 : offset;
}
