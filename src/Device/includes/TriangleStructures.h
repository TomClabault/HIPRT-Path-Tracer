/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_TRIANGLE_STRUCTURES_H
#define DEVICE_TRIANGLE_STRUCTURES_H

#include "HostDeviceCommon/Math.h"

/**
 * Structure that contains the vertex index (in the vertex buffer) of the 3 vertices of a triangle
 */
struct TriangleIndices
{
    int x; // vertex A
    int y; // vertex B
    int z; // vertex C
};

HIPRT_HOST_DEVICE HIPRT_INLINE TriangleIndices load_triangle_vertex_indices(int* triangle_indices_buffer, int primitive_index)
{
    int primitive_index_3 = primitive_index * 3;

    return TriangleIndices
    {
        triangle_indices_buffer[primitive_index_3 + 0],
        triangle_indices_buffer[primitive_index_3 + 1],
        triangle_indices_buffer[primitive_index_3 + 2]
    };
}

/**
 * Structure that contains the UV texcoords a the 3 vertices of a triangle
 */
struct TriangleTexcoords
{
    float2 x; // vertex A
    float2 y; // vertex B
    float2 z; // vertex C
};

HIPRT_HOST_DEVICE HIPRT_INLINE TriangleTexcoords load_triangle_texcoords(float2* texcoords_buffer, TriangleIndices triangle_vertex_indices)
{
    return TriangleTexcoords
    {
        texcoords_buffer[triangle_vertex_indices.x],
        texcoords_buffer[triangle_vertex_indices.y],
        texcoords_buffer[triangle_vertex_indices.z]
    };
}

#endif
