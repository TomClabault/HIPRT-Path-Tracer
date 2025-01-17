/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "HostDeviceCommon/Material/Material.h"
#include "Image/Image.h"
#include "Scene/BoundingBox.h"
#include "Scene/Camera.h"
#include "Renderer/Sphere.h"
#include "Renderer/Triangle.h"

#include <thread>
#include <vector>

/**
 * Structure that holds the indices of the textures of a material during scene parsing
 */
struct ParsedMaterialTextureIndices
{
    int base_color_texture_index = MaterialUtils::NO_TEXTURE;
    int emission_texture_index = MaterialUtils::NO_TEXTURE;

    int roughness_texture_index = MaterialUtils::NO_TEXTURE;
    int metallic_texture_index = MaterialUtils::NO_TEXTURE;
    int roughness_metallic_texture_index = MaterialUtils::NO_TEXTURE;

    int specular_texture_index = MaterialUtils::NO_TEXTURE;
    int coat_texture_index = MaterialUtils::NO_TEXTURE;
    int sheen_texture_index = MaterialUtils::NO_TEXTURE;
    int specular_transmission_texture_index = MaterialUtils::NO_TEXTURE;

    int normal_map_texture_index = MaterialUtils::NO_TEXTURE;
};

struct SceneParserOptions
{
    float override_aspect_ratio;

    // How many CPU threads to use when loading the textures of the scene.
    // 
    // Note that blindly defaulting to 1 thread per texture may not be the
    // best idea, especially on HDDs. This is because with one thread per texture,
    // all textures will be loading at the same time. Although this may utilize the
    // CPU very well, this will cause A LOT of random read accesses on the drive
    // can SIGNIFICANTLY degrade performance. This is mostly applicable to HDDs but
    // to SSDs too to some extent. You may want to use a higher thread count for SSDs
    // though to be sure to feed enough work to the CPU to keep up with the fast SSD.
    // 
    // -1 to use one thread per texture.
    //
    // 16 seemed to be a good arbitrary number to avoid trashing the disks on my setup 
    // (tested on the Amazon Lumberyard Bistro on both HDD and SSD)
    int nb_texture_threads = 4;
};

struct SceneMetadata
{
    // The material names are used for displaying in the material editor of ImGui
    std::vector<std::string> material_names;
    // Names of the objects in the scene
    std::vector<std::string> mesh_names;
    // For a given mesh index, its material index
    std::vector<int> mesh_material_indices;

    // AABBs of the meshes of the scene
    std::vector<BoundingBox> mesh_bounding_boxes;

    // AABB of the whole scene
    BoundingBox scene_bounding_box;
};

struct Scene
{
    SceneMetadata metadata;

    std::vector<CPUMaterial> materials;
    // Material textures. Needs to be index by a material index. 
    std::vector<Image8Bit> textures;

    std::vector<int> triangle_indices;
    std::vector<float3> vertices_positions;
    std::vector<unsigned char> has_vertex_normals;
    std::vector<float3> vertex_normals;
    std::vector<float2> texcoords;
    std::vector<int> emissive_triangle_indices;
    std::vector<int> material_indices;
    std::vector<bool> material_has_opaque_base_color_texture;

    bool has_camera = false;
    Camera camera;

    Sphere add_sphere(const float3& center, float radius, const CPUMaterial& material, int primitive_index)
    {
        int material_index = materials.size();

        materials.push_back(material);
        material_indices.push_back(material_index);

        Sphere sphere(center, radius, primitive_index);

        return sphere;
    }

    std::vector<Triangle> get_triangles()
    {
        std::vector<Triangle> triangles;

        for (int i = 0; i < triangle_indices.size(); i += 3)
        {
            triangles.push_back(Triangle(*reinterpret_cast<float3*>(&vertices_positions[triangle_indices[i + 0]]),
                                         *reinterpret_cast<float3*>(&vertices_positions[triangle_indices[i + 1]]),
                                         *reinterpret_cast<float3*>(&vertices_positions[triangle_indices[i + 2]])));
        }

        return triangles;
    }
};

class SceneParser
{
public:
    /**
     * Parses the scene file at @filepath and stores the parsed data in the parsed_scene parameter.
     * All formats supported by the ASSIMP library are supported by the renderer.
     * 
     * If provided, the @frame_aspect_override parameter in the options structure is meant to override 
     * the aspect ratio of the camera of the scene file (if any). This is useful because the renderer
     * uses a default aspect ratio of 16:9 but the camera of the scene file may not use the same aspect. 
     * Without this parameter, this would result in rendering the scene with an aspect different of 16:9 in the default 
     * framebuffer of the renderer which is 16:9, resulting in deformations.
     */
    static void parse_scene_file(const std::string& filepath, Assimp::Importer& assimp_importer, Scene& parsed_scene, SceneParserOptions& options);

private:

    static void parse_camera(const aiScene* scene, Scene& parsed_scene, float frame_aspect_override);

    /** 
     * Prepares all the necessary data for multithreaded texture-loading
     * 
     * @ scene is the scene to parse the textures from
     * @ textures_paths is a list of pair of <texture_type -> the path to the texture>.
     * @ material_texture_indices is a list that is as long as there are unique materials
     *      in the scene. Each field of the stucture contains the index of the texture used
     *      by that material. -1 if the material doesn't have that type of texture
     *      (if structure.base_color_texture_index == MaterialUtils::NO_TEXTURE for example, that means
     *      that the material doesn't have a base color texture)
     * @ material_indices is a vector which is 'number of textures' long and contains the
     *      index of the material that the texture belongs to.
     *      If material_indices[3] == 2, this means that the texture 3 (the fourth texure)
     *      is used by material 2 (which is the third material)
     * @ texture_per_mesh is a list that is 'number of mesh' long and that gives the number
     *      of textures used per mesh
     * @ texture_indices_offset /By how much to offset the indices of the textures used by a material.
     *      For example, if there are 5 materials in the scene that all use a different base color
     *      texture, after the call to prepare_textures(), they will all have 0 as the index of their
     *      base color texture. This is obviously wrong and it should be 0, 1, 2, 3, 4 for
     *      each material since they use their own texture. This is what this vector is for, it contains
     *      the offsets that are going to be used so that each material has proper texture indices.
     * @ texture_count How many texture are in the scene
     */
    static void prepare_textures(const aiScene* scene, std::vector<std::pair<aiTextureType, std::string>>& texture_paths, std::vector<ParsedMaterialTextureIndices>& material_texture_indices, std::vector<int>& material_indices, std::vector<int>& texture_per_mesh, std::vector<int>& texture_indices_offsets, int& texture_count);
    static void assign_material_texture_indices(std::vector<CPUMaterial>& materials, const std::vector<ParsedMaterialTextureIndices>& material_tex_indices, const std::vector<int>& material_textures_offsets);
    static void dispatch_texture_loading(Scene& parsed_scene, const std::string& scene_path, int nb_threads, const std::vector<std::pair<aiTextureType, std::string>>& texture_paths, const std::vector<int>& material_indices);

    static void read_material_properties(aiMaterial* mesh_material, CPUMaterial& renderer_material);
    /**
     * Check if the mesh material has a texture of the given type. If so, returns the index of the
     * texture within texturePathList and appends the path of the texture to the list. If the material
     * doesn't have the required texture, returns -1
     */
    static int get_first_texture_of_type(aiMaterial* mesh_material, aiTextureType type, std::vector<std::pair<aiTextureType, std::string>>& texture_path_list);
    static std::vector<std::pair<aiTextureType, std::string>> get_textures_paths_and_indices(aiMaterial* mesh_material, ParsedMaterialTextureIndices& texture_indices);
    static std::vector<std::pair<aiTextureType, std::string>> normalize_texture_paths(std::vector<std::pair<aiTextureType, std::string>>& paths);
    static CPUMaterial offset_textures_indices(const CPUMaterial& renderer_material, int offset);
};

#endif
