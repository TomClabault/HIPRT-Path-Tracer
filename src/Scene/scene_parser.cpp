#include "Scene/scene_parser.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

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
        Point camera_position = Point(*((Vector*)&scene->mCameras[0]->mPosition));
        Point camera_lookat = Point(*((Vector*)&scene->mCameras[0]->mLookAt));
        Vector camera_up = *((Vector*)&scene->mCameras[0]->mUp);

        // fov in radians
        float fov = scene->mCameras[0]->mHorizontalFOV;
        // TODO The +5.0f here is there to account for the slight difference of FOV between the
        // rendered and Blender. This probably shouldn't be here
        float degrees_fov = fov / M_PI * 180.0f + 5.0f;
        float full_degrees_fov = degrees_fov / 2.0f;

        parsed_scene.camera = Camera(camera_position, camera_lookat, camera_up, full_degrees_fov);
        parsed_scene.has_camera = true;
    }

    int total_face_count = 0;
    for (int mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
    {
        aiMesh* mesh = scene->mMeshes[mesh_index];
        total_face_count += mesh->mNumFaces;
    }

    parsed_scene.triangles.reserve(total_face_count);
    for (int mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
    {
        aiMesh* mesh = scene->mMeshes[mesh_index];
        aiMaterial* mesh_material = scene->mMaterials[mesh->mMaterialIndex];

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
        renderer_material.emission = Color(emissive_color.r, emissive_color.g, emissive_color.b, 1.0f);
        renderer_material.metalness = metalness;
        renderer_material.roughness = std::max(1.0e-2f, roughness); // Clamping the roughness to avoid edge cases when roughness == 0.0f
        renderer_material.ior = ior;
        renderer_material.transmission_factor = error_code_transmission_factor == AI_SUCCESS ? transmission_factor : 0.0f;
        if (renderer_material.transmission_factor > 0.0f)
            renderer_material.brdf_type = BRDF::SpecularFresnel;
        else
            renderer_material.brdf_type = BRDF::CookTorrance;

        //Adding the material to the parsed scene
        parsed_scene.materials.push_back(renderer_material);
        int material_index = parsed_scene.materials.size() - 1;

        //If the mesh is emissive
        bool is_mesh_emissive = false;
        if (error_code_emissive == AI_SUCCESS && (emissive_color.r != 0.0f || emissive_color.g != 0.0f || emissive_color.b != 0.0f))
            is_mesh_emissive = true;

        for (int face_index = 0; face_index < mesh->mNumFaces; face_index++)
        {
            aiFace face = mesh->mFaces[face_index];

            //All faces should be triangles so we're assuming exactly 3 vertices
            Point vertex_a = *(Point*)(&mesh->mVertices[face.mIndices[0]]);
            Point vertex_b = *(Point*)(&mesh->mVertices[face.mIndices[1]]);
            Point vertex_c = *(Point*)(&mesh->mVertices[face.mIndices[2]]);
            parsed_scene.triangles.push_back(Triangle(vertex_a, vertex_b, vertex_c));

            //The face that we just pushed in the triangle buffer is emissive
            //We're going to add its index to the emissive triangles buffer
            if (is_mesh_emissive)
                parsed_scene.emissive_triangle_indices.push_back(parsed_scene.triangles.size() - 1);

            //We're pushing the same material index for all the faces of this mesh
            //because all faces of a mesh have the same material (that's how ASSIMP importer's
            //do things internally)
            parsed_scene.material_indices.push_back(material_index);
        }
    }

    return parsed_scene;
}