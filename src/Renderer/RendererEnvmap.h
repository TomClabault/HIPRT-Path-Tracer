/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ENVMAP_H
#define RENDERER_ENVMAP_H

#include "HIPRT-Orochi/OrochiEnvmap.h"

class GPURenderer;

class RendererEnvmap
{
public:
	float rotation_X = 0.0f;
	float rotation_Y = 0.0f;
	float rotation_Z = 0.0f;

	bool animate = false;
	// How many degrees per second will the envmap rotate if 'animate' is true
	float animation_speed_X = 0.0f;
	float animation_speed_Y = 0.0f;
	float animation_speed_Z = 0.0f;

	float4x4 envmap_to_world_matrix;
	float4x4 world_to_envmap_matrix;

	/**
	 * Recomputes the envmap matrices if necessary based on
	 the current values of rotation_X, rotation_Y and rotation_Z
	 */
	void update(GPURenderer* renderer);

	OrochiEnvmap& get_orochi_envmap();

private:
	/**
	 * Updates the world settings, envmap itself, etc... of the renderer
	 */
	void update_renderer(GPURenderer* renderer);

	// Cached values of rotation_X, Y, Z so that we don't recompute the matrices
	// if the rotations haven't changed
	float prev_rotation_X = -1.0f;
	float prev_rotation_Y = -1.0f;
	float prev_rotation_Z = -1.0f;

	// This object contains the memory data of the envmap
	OrochiEnvmap m_orochi_envmap;
};

#endif
