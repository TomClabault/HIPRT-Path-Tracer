/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "HostDeviceCommon/Material.h"
#include "Scene/Camera.h"
#include "Renderer/Sphere.h"
#include "Renderer/Triangle.h"

#include <vector>

struct Scene
{
    std::vector<RendererMaterial> materials;
    std::vector<ImageRGBA> textures;
    // This vector of is sRGB indicates whether the texture is sRGB 
    // (and will need to be converted to linear in the shader or not)
    std::vector<unsigned char> textures_is_srgb;

    std::vector<int> triangle_indices;
    std::vector<float3> vertices_positions;
    std::vector<unsigned char> has_vertex_normals;
    std::vector<float3> vertex_normals;
    std::vector<float2> texcoords;
    std::vector<int> emissive_triangle_indices;
    std::vector<int> material_indices;

    bool has_camera = false;
    Camera camera;

    Sphere add_sphere(const float3& center, float radius, const RendererMaterial& material, int primitive_index)
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
     * Parses the scene file at @filepath and returns a scene appropriate for the renderer.
     * All formats supported by the ASSIMP library are supported by the renderer
     * 
     * If provided, the @frame_aspect_override parameter is meant to override the aspect ratio of the camera
     * of the scene file (if any). This is useful because the renderer uses a default aspect ratio
     * of 16:9 but the camera of the scene file ma not use the same aspect. Without this parameter,
     * this would result in rendering the scene with an aspect different of 16:9 in the defualt 
     * framebuffer of the renderer which is 16:9, resulting in deformations.
     */
    static Scene parse_scene_file(const std::string& filepath, float frame_aspect_override = -1.0f);

private:
    static RendererMaterial read_material_properties(aiMaterial* mesh_material);

    /**
     * Check if the mesh material has a texture of the given type. If so, returns the index of the
     * texture within texturePathList and appends the path of the texture to the list. If the material
     * doesn't have the required texture, returns -1
     */
    static int get_first_texture_of_type(aiMaterial* mesh_material, aiTextureType type, std::vector<std::pair<aiTextureType, std::string>>& texture_path_list);
    static std::vector<std::pair<aiTextureType, std::string>> get_textures_paths(aiMaterial* mesh_material, RendererMaterial& renderer_material);
    static void normalize_texture_paths(std::vector<std::pair<aiTextureType, std::string>>& paths);
    static std::vector<ImageRGBA> read_textures(const std::string& filepath, const std::vector<std::pair<aiTextureType, std::string>>& texture_paths);
    static void offset_textures_indices(RendererMaterial& renderer_material, int offset);
};

#endif
