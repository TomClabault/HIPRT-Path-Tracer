/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Scene/scene_parser.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/matrix_decompose.hpp"

RendererMaterial SceneParser::ai_mat_to_renderer_mat(aiMaterial* mesh_material)
{
    // Only for debug purposes to know what material index we're at globally in the scene
    static int debug_counter = 0;

    //Getting the properties that are going to be used by the materials
    //of the application
    aiColor3D diffuse_color;
    aiColor3D emissive_color;
    float metallic, roughness;
    float ior, specular_transmission;

    aiReturn error_code_transmission, error_code_emissive, error_code_ior;
    mesh_material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse_color);
    error_code_emissive = mesh_material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive_color);
    mesh_material->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
    mesh_material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
    error_code_ior = mesh_material->Get(AI_MATKEY_REFRACTI, ior);
    error_code_transmission = mesh_material->Get(AI_MATKEY_TRANSMISSION_FACTOR, specular_transmission);

    //Creating the material used by the application from the properties read
    RendererMaterial renderer_material;
    renderer_material.base_color = ColorRGB(diffuse_color.r, diffuse_color.g, diffuse_color.b);
    if (error_code_emissive == AI_SUCCESS)
    {
        if (emissive_color.r > 0 || emissive_color.g > 0 || emissive_color.b > 0)
        {
            float emission_strength;
            mesh_material->Get(AI_MATKEY_EMISSIVE_INTENSITY, emission_strength);

            renderer_material.emission = ColorRGB(emissive_color.r, emissive_color.g, emissive_color.b) * emission_strength;
        }
    }
    else
        renderer_material.emission = ColorRGB(0.0f, 0.0f, 0.0f);
    renderer_material.metallic = metallic;
    renderer_material.roughness = roughness;
    renderer_material.anisotropic = 0.0f; // TODO read from the file instead of hardcoded
    renderer_material.sheen_tint = 1.0f; // TODO read from the file instead of hardcoded
    renderer_material.sheen_color = ColorRGB(1.0f, 1.0f, 1.0f); // TODO read from the file instead of hardcoded
    renderer_material.ior = error_code_transmission == AI_SUCCESS ? ior : 1.45f;;
    renderer_material.specular_transmission = error_code_transmission == AI_SUCCESS ? specular_transmission : 0.0f;
    renderer_material.brdf_type = BRDF::Disney;

    renderer_material.make_safe();
    renderer_material.precompute_properties();

    debug_counter++;
    return renderer_material;
}

Scene SceneParser::parse_scene_file(const std::string& filepath, float frame_aspect_override)
{
    Scene parsed_scene;

    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(filepath, aiPostProcessSteps::aiProcess_PreTransformVertices | aiPostProcessSteps::aiProcess_Triangulate);
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
    for (int mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
    {
        int max_mesh_index_offset = 0;

        aiMesh* mesh = scene->mMeshes[mesh_index];
        aiMaterial* mesh_material = scene->mMaterials[mesh->mMaterialIndex];
        RendererMaterial renderer_material = ai_mat_to_renderer_mat(mesh_material);

        //Adding the material to the parsed scene
        parsed_scene.materials.push_back(renderer_material);
        int material_index = parsed_scene.materials.size() - 1;
        bool is_mesh_emissive = renderer_material.is_emissive();

        // Inserting the normals if present
        if (mesh->HasNormals())
            parsed_scene.vertex_normals.insert(parsed_scene.vertex_normals.end(),
                reinterpret_cast<hiprtFloat3*>(mesh->mNormals),
                reinterpret_cast<hiprtFloat3*>(&mesh->mNormals[mesh->mNumVertices]));
        else
            parsed_scene.vertex_normals.insert(parsed_scene.vertex_normals.end(), mesh->mNumVertices, hiprtFloat3{0, 0, 0});

        // Inserting 0 or 1 depending on whether the normals are present or not.
        // These values will be used in the shader to determine whether we should do
        // smooth shading or not
        parsed_scene.normals_present.insert(parsed_scene.normals_present.end(), mesh->mNumVertices, mesh->HasNormals());

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

            // The face that we just pushed in the triangle buffer is emissive
            // We're going to add its index to the emissive triangles buffer
            if (is_mesh_emissive)
                parsed_scene.emissive_triangle_indices.push_back(parsed_scene.triangle_indices.size() / 3 - 1);

            // We're pushing the same material index for all the faces of this mesh
            // because all faces of a mesh have the same material (that's how ASSIMP importer's
            // do things internally)
            parsed_scene.material_indices.push_back(material_index);
        }

        // If the max index of the mesh was 19, we want the next to start
        // at 20, not 19, so we ++
        max_mesh_index_offset++;
        // Adding the maximum index of the mesh to our global indices offset 
        global_indices_offset += max_mesh_index_offset;
    }

    std::cout << parsed_scene.vertices_positions.size() << " vertices ; " << parsed_scene.triangle_indices.size() / 3 << " triangles" << std::endl;

    return parsed_scene;
}