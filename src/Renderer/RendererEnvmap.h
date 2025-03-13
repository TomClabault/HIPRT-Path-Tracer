/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ENVMAP_H
#define RENDERER_ENVMAP_H

#include "Image/EnvmapRGBE9995.h"

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
	float animation_speed_Y = 8.0f;
	float animation_speed_Z = 0.0f;

	float3x3 envmap_to_world_matrix;
	float3x3 world_to_envmap_matrix;

	/**
	 * Uploads the data from 'image' to a texture on the GPU and remembers the path
	 * of the envmap if needed later (when recomputing the CDF/alias table for example
	 * since we do not keep the data of the envmap in memory (envmaps can be quite big), 
	 * we'll have to read it again from the disk)
	 */
	void init_from_image(const Image32Bit& image, const std::string& envmap_filepath);

	/**
	 * - Updates the animation of the envmap
	 * - Recomputes the sampling data structure (CDF for binary search sampling, 
	 *		alias table for alias table sampling) if necessary
	 */
	void update(GPURenderer* renderer, float delta_time);

	/**
	 * Computes the CDF or alias table of the envmap based of the envmap sampling strategy used
	 * by the renderer.
	 * 
	 * The data structure that is unused will also be freed to free some VRAM.
	 */
	void recompute_sampling_data_structure(GPURenderer* renderer, const Image32Bit* = nullptr);

	RGBE9995Packed* get_packed_data_pointer();
	void get_alias_table_device_pointers(float*& out_probas_pointer, int*& out_alias_pointer);
	float* get_cdf_device_pointer();

	unsigned int get_width();
	unsigned int get_height();

	/**
	 * Returns the VRAM used by the sampling structure of the envmap in MB
	 */
	float get_sampling_structure_VRAM_usage() const;

private:
	void recompute_CDF(const Image32Bit* image);
	void recompute_alias_table(const Image32Bit* image);

	/**
	 * Recomputes the envmap matrices if necessary based on
	 * the current values of rotation_X, rotation_Y and rotation_Z
	 * 
     * The 'delta_time' parameter should be how much time passed, in milliseconds, since the last
     * call to do_animation()
	 */
	void do_animation(GPURenderer* renderer, float delta_time);

	/**
	 * Updates the world settings, envmap itself, etc... of the renderer
	 */
	void update_renderer(GPURenderer* renderer);

	// Cached values of rotation_X, Y, Z so that we don't recompute the matrices
	// if the rotations haven't changed
	float prev_rotation_X = -1.0f;
	float prev_rotation_Y = -1.0f;
	float prev_rotation_Z = -1.0f;

	// The envmap path is saved if we need to load the envmap data again
	// 
	// This requires reading from the disk again but this saves memory because
	// we don't have to store the envmap, we can just read it from the disk again.
	// And envmaps are heavy so we're actually saving a lot of memory there
	std::string m_envmap_filepath;

	// This object contains the memory data of the envmap
	RGBE9995Envmap<true> m_envmap_data;
	unsigned int m_width = 0;
	unsigned int m_height = 0;

	// CDF / Alias table for sampling the envmap
	OrochiBuffer<float> m_cdf;
	OrochiBuffer<float> m_alias_table_probas;
	OrochiBuffer<int> m_alias_table_alias;
	float m_luminance_total_sum = 0.0f;
};

#endif
