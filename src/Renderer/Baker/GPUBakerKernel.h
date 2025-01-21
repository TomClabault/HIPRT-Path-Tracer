/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_BAKER_KERNEL_H
#define GPU_BAKER_KERNEL_H

#include "Renderer/GPURenderer.h"

#include <string>

class GPUBakerKernel
{
public:
	GPUBakerKernel() {}
	GPUBakerKernel(std::shared_ptr<GPURenderer> renderer, oroStream_t bake_stream, std::shared_ptr<std::mutex> compiler_priority_mutex, 
		const std::string& kernel_filepath, const std::string& kernel_function, const std::string& kernel_title);

	/**
	 * Starts the baking process
	 */
	void bake_internal(int3 bake_resolution, const void* bake_settings_pointer, int nb_kernel_iterations, std::string output_filename);

	/**
	 * Is the baking process complete?
	 */
	bool is_complete() const;

private:
	bool m_bake_complete = true;

	std::shared_ptr<GPURenderer> m_renderer = nullptr;
	oroStream_t m_bake_stream = nullptr;
	// Mutex so that if we're baking multiple textures at the same time,
	// we don't run into issue with the compilers wanting to take the priority
	// (over background compiling kerneks) at the same time
	std::shared_ptr<std::mutex> m_compiler_priority_mutex = nullptr;

	// Filepath and function within this file that will be launched
	// when the baking of the kernel starts
	std::string m_kernel_filepath = "";
	std::string m_kernel_function = "";
	// String used for logging infos
	std::string m_kernel_title = "";
	// State for baking GGX conductors directional albedo
	GPUKernel m_bake_kernel;

	// Buffer for holding the baked data
	OrochiBuffer<float> m_bake_buffer;
};

#endif
