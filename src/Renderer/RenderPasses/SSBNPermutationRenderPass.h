/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_SSBN_PERMUTATION_RENDER_PASS_H
#define RENDERER_SSBN_PERMUTATION_RENDER_PASS_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "HostDeviceCommon/RenderData.h"
#include "Renderer/RenderPasses/RenderPass.h"
#include "UI/ApplicationSettings.h"

class GPURenderer;

class SSBNPermutationRenderPass : public RenderPass
{
public:
	static const std::string SSBN_PERMUTATION_RENDER_PASS_NAME;
	static const std::string SSBN_PERMUTATION_SORTING_PASS;
	static const std::string SSBN_PERMUTATION_RETARGETING_PASS;
	static const std::string SSBN_PERMUTATION_REFRESH_SEEDS_PASS;
	static const int DEFAULT_MAX_RETARGETING_RADIUS = 5;

	SSBNPermutationRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;
	void reload_blue_noise_texture(int new_width, int new_height);
	void reload_retargeting_data(int new_max_retargeting_radius);
	std::string get_permutation_file_path_no_extension(int retarget_radius = -1);

	/**
	 * Allocates/deallocates the buffers used by GMoN.
	 *
	 * Returns true or false depending on whether or not the render buffer data have been invalidated
	 */
	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;

	virtual bool is_render_pass_used() const override;
	bool& get_do_retargeting();

	int& get_blue_noise_texture_width();
	int& get_blue_noise_texture_height();
	int& get_max_retargeting_radius();
	int& get_refresh_seeds_sample_interval();

private:
	bool m_using_ssbn_permutation = true;
	bool m_do_retargeting		  = true;

	OrochiBuffer<unsigned int> m_sorted_seeds_buffer;
	OrochiBuffer<unsigned char> m_blue_noise_dither_texture_buffer;
	OrochiBuffer<int> m_blue_noise_retargeting_texture_buffer;

	OrochiBuffer<uint3_t> m_screen_space_hash_grid_buffer;
	OrochiBuffer<int> m_screen_space_hash_grid_cell_offsets_buffer;
	unsigned int m_different_hash_count = 0;

	int m_blue_noise_texture_width		= 4096;
	int m_blue_noise_texture_height		= 2048;
	int m_max_retargeting_radius		= DEFAULT_MAX_RETARGETING_RADIUS;
	int m_refresh_seeds_sample_interval = 64;
};

#endif
