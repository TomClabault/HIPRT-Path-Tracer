/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Scene/ParsedEmissiveMeshes.h"

const std::vector<float3> ParsedEmissiveMesh::BinningNormals = {
	// Six faces of a cube
	make_float3(1.0f, 0.0f, 0.0f),
	make_float3(0.0f, 1.0f, 0.0f),
	make_float3(0.0f, 0.0f, 1.0f),
	make_float3(-1.0f, 0.0f, 0.0f),
	make_float3(0.0f, -1.0f, 0.0f),
	make_float3(0.0f, 0.0f, -1.0f),

	// Normals at eight corners of a cube (from the center to the corner)
	// 
	// Computed assuming a cube whose center is at 0, 0, 0 and 
	// normalizing the position of its corners
	hippt::normalize(make_float3(-1.0f, -1.0f, -1.0f)),
	hippt::normalize(make_float3(1.0f, -1.0f, -1.0f)),
	hippt::normalize(make_float3(1.0f, 1.0f, -1.0f)),
	hippt::normalize(make_float3(-1.0f, 1.0f, -1.0f)),

	hippt::normalize(make_float3(-1.0f, -1.0f, 1.0f)),
	hippt::normalize(make_float3(1.0f, -1.0f, 1.0f)),
	hippt::normalize(make_float3(1.0f, 1.0f, 1.0f)),
	hippt::normalize(make_float3(-1.0f, 1.0f, 1.0f))
};
