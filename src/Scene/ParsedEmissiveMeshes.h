/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef SCENE_PARSED_EMISSIVE_MESHES_H
#define SCENE_PARSED_EMISSIVE_MESHES_H

#include "Renderer/CPUGPUCommonDataStructures/EmissiveMeshHost.h"

struct ParsedEmissiveMeshes
{
    // Contains the list of all emissive meshes of the scene. This list is going to be used by some light
    // sampling scheme such as ReGIR
    // 
    // Any emissive mesh that contains emissive textures is NOT in that list because emissive textures
    // aren't importance sampled
    std::vector<EmissiveMeshHost<std::vector>> emissive_meshes;

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
