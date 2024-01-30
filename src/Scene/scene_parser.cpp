#include "Scene/scene_parser.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/matrix_decompose.hpp"

RendererMaterial SceneParser::mesh_mat_to_renderer_mat(aiMaterial* mesh_material)
{
    //Getting the properties that are going to be used by the materials
    //of the application
    aiColor3D diffuse_color;
    aiColor3D emissive_color;
    float metalness, roughness;
    float ior, transmission_factor;

    aiReturn error_code_transmission_factor, error_code_emissive;
    mesh_material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse_color);
    error_code_emissive = mesh_material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive_color);
    mesh_material->Get(AI_MATKEY_METALLIC_FACTOR, metalness);
    mesh_material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
    mesh_material->Get(AI_MATKEY_REFRACTI, ior);
    error_code_transmission_factor = mesh_material->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission_factor);

    //Creating the material used by the application from the properties read
    RendererMaterial renderer_material;
    renderer_material.diffuse = Color(diffuse_color.r, diffuse_color.g, diffuse_color.b, 1.0f);
    if (error_code_emissive == AI_SUCCESS)
        renderer_material.emission = Color(emissive_color.r, emissive_color.g, emissive_color.b, 1.0f);
    else
        renderer_material.emission = Color(0.0f, 0.0f, 0.0f, 1.0f);
    renderer_material.metalness = metalness;
    renderer_material.roughness = std::max(1.0e-2f, roughness); // Clamping the roughness to avoid edge cases when roughness == 0.0f
    renderer_material.ior = ior;
    renderer_material.transmission_factor = error_code_transmission_factor == AI_SUCCESS ? transmission_factor : 0.0f;
    if (renderer_material.transmission_factor > 0.0f)
        renderer_material.brdf_type = BRDF::SpecularFresnel;
    else
        renderer_material.brdf_type = BRDF::CookTorrance;

    return renderer_material;
}

Scene SceneParser::parse_scene_file(const std::string& filepath)
{
    Scene parsed_scene;

    Assimp::Importer importer;

    //TODO check perf of aiPostProcessSteps::aiProcess_ImproveCacheLocality
    const aiScene* scene = importer.ReadFile(filepath, aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_PreTransformVertices | aiPostProcessSteps::aiProcess_Triangulate);
    if (scene == nullptr)
    {
        std::cerr << importer.GetErrorString() << std::endl;
        std::exit(1);
    }

    // Taking the first camera as the camera of the scene
    if (scene->mNumCameras > 0)
    {
        //aiMatrix4x4 camera_matrix;
        //scene->mCameras[0]->GetCameraMatrix(camera_matrix);

        //aiQuaternion ai_rotation;
        //aiVector3f ai_translation;
        //camera_matrix.DecomposeNoScaling(ai_rotation, ai_translation);
        //std::cout << std::endl;


        //parsed_scene.camera.rotation = *reinterpret_cast<glm::quat*>(&ai_rotation);
        ////parsed_scene.camera.rotation = parsed_scene.camera.rotation * Camera::DEFAULT_COORDINATES_SYSTEM;
        //parsed_scene.camera.translation = *reinterpret_cast<glm::vec3*>(&ai_translation);
        //parsed_scene.camera.translation.z *= -1;


        glm::vec3 camera_position = *reinterpret_cast<glm::vec3*>(&scene->mCameras[0]-> mPosition);
        glm::vec3 camera_lookat = *reinterpret_cast<glm::vec3*>(&scene->mCameras[0]->mLookAt);
        glm::vec3 camera_up = *reinterpret_cast<glm::vec3*>(&scene->mCameras[0]->mUp);

        glm::mat4x4 view_matrix = glm::lookAt(camera_position, camera_lookat, camera_up);
        view_matrix = Camera::DEFAULT_COORDINATES_SYSTEM * view_matrix;

        glm::quat rotation;
        glm::vec3 translation, scale, skew;
        glm::vec4 perspective;

        glm::decompose(view_matrix, scale, rotation, translation, skew, perspective);
        parsed_scene.camera.rotation = rotation;
        parsed_scene.camera.translation = translation;

        //camera_matrix = camera_matrix * Camera::DEFAULT_COORDINATES_SYSTEM;
        //camera_matrix = glm::transpose(camera_matrix);

        /*glm::quat rotation;
        glm::vec3 translation, scale, skew;
        glm::vec4 perspective;

        glm::decompose(camera_matrix, scale, rotation, translation, skew, perspective);
        parsed_scene.camera.rotation = rotation;
        parsed_scene.camera.translation = translation;*/
        //glm::vec3 camera_position = *reinterpret_cast<glm::vec3*>(&scene->mCameras[0]-> mPosition);
        //glm::vec3 camera_lookat = *reinterpret_cast<glm::vec3*>(&scene->mCameras[0]->mLookAt);
        //glm::vec3 camera_up = *reinterpret_cast<glm::vec3*>(&scene->mCameras[0]->mUp);


        //// fov in radians
        //float fov = scene->mCameras[0]->mHorizontalFOV;
        //// TODO The +5.0f here is there to account for the slight difference of FOV between the
        //// rendered and Blender. This probably shouldn't be here
        //float degrees_fov = fov / M_PI * 180.0f + 5.0f;
        //float full_degrees_fov = degrees_fov / 2.0f;

        //parsed_scene.camera = Camera(camera_position, camera_lookat, camera_up, full_degrees_fov);
        //parsed_scene.has_camera = true;
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
        RendererMaterial renderer_material = mesh_mat_to_renderer_mat(mesh_material);

        //Adding the material to the parsed scene
        parsed_scene.materials.push_back(renderer_material);
        int material_index = parsed_scene.materials.size() - 1;
        bool is_mesh_emissive = renderer_material.is_emissive();

        // Inserting all the vertices of the mesh
        parsed_scene.vertices_positions.insert(parsed_scene.vertices_positions.end(), reinterpret_cast<hiprtFloat3*>(&mesh->mVertices[0]), reinterpret_cast<hiprtFloat3*>(&mesh->mVertices[mesh->mNumVertices]));
        for (int face_index = 0; face_index < mesh->mNumFaces; face_index++)
        {
            aiFace face = mesh->mFaces[face_index];

            int index_1 = face.mIndices[0];
            int index_2 = face.mIndices[1];
            int index_3 = face.mIndices[2];

            // Accumulating the maximum index of this mesh
            max_mesh_index_offset = std::max(max_mesh_index_offset, std::max(index_1, std::max(index_2, index_3)));

            parsed_scene.vertices_indices.push_back(index_1 + global_indices_offset);
            parsed_scene.vertices_indices.push_back(index_2 + global_indices_offset);
            parsed_scene.vertices_indices.push_back(index_3 + global_indices_offset);

            // The face that we just pushed in the triangle buffer is emissive
            // We're going to add its index to the emissive triangles buffer
            if (is_mesh_emissive)
                parsed_scene.emissive_triangle_indices.push_back(parsed_scene.material_indices.size() - 3); // TODO this may be incorrect

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

    return parsed_scene;
}