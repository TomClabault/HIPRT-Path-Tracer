/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef SCENE_PARSED_EMISSIVE_MESHES_H
#define SCENE_PARSED_EMISSIVE_MESHES_H

#include <vector>

#include "HostDeviceCommon/Math.h"

struct ParsedEmissiveMesh
{
    static const std::vector<float3> BinningNormals;

    // Alias table built on the power of all the emissive triangles of the mesh
    std::vector<float> alias_probas;
    std::vector<int> alias_aliases;

    // The faces of the emissive mesh are going to be binned according to their orientation
    // for better sampling later.
    //
    // All the indices of the faces are going to be found in 'binned_faces_indices'
    //
    // There are as many bins as 'binned_faces_start_index.size()' (equivalently as many
    // bins as there are BinningNormals)
    //
    // In the binned_faces_indices (which is a big concatenated buffer):
    // Bin[0] starts at 'binned_faces_start_index[0]' and contains 'binned_faces_counts[0]' faces
    // Bin[1] starts at 'binned_faces_start_index[1]' and contains 'binned_faces_counts[1]' faces
    // ...
    std::vector<unsigned int> binned_faces_indices;
    std::vector<unsigned int> binned_faces_start_index;
    std::vector<unsigned int> binned_faces_counts;
    std::vector<float> binned_faces_total_power;
    std::vector<unsigned int> binned_faces_mesh_face_index_to_bin_index;

    // Average of all the vertices of the emissive mesh
    float3 average_mesh_point = make_float3(0.0f, 0.0f, 0.0f);

    float total_mesh_emissive_power = 0.0f;
    unsigned int emissive_triangle_count = 0;
};

struct ParsedEmissiveMeshes
{
    // Contains the list of all emissive meshes of the scene. This list is going to be used by some light
    // sampling scheme such as ReGIR
    // 
    // Any emissive mesh that contains emissive textures is NOT in that list because emissive textures
    // aren't importance sampled
    std::vector<ParsedEmissiveMesh> emissive_meshes;

    // PDF that a given triangle in a given emissive mesh is sampled by the sampler
    // that samples triangles in meshes.
    //
    // For example, if the emissive mesh [0] of the scene has 5 emissive triangles
    // then entries [0], [1], ... [4] of this vector will contain the PDF that triangles
    // 0, 1, ..., 4 are sampled within mesh [0]
    //
    // The PDF is assumed to be power proportional
    std::vector<float> emissive_meshes_triangles_PDFs;
};

#endif
