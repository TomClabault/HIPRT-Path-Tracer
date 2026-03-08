/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/RenderPasses/SSBNPermutationRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_RENDER_PASS_NAME = "SSBN Permutation Render Pass";
const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS	   = "SSBN Permutation Sorting Pass";
const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS = "SSBN Permutation Retargeting Pass";

SSBNPermutationRenderPass::SSBNPermutationRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(renderer, options, SSBNPermutationRenderPass::SSBN_PERMUTATION_RENDER_PASS_NAME)
{
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS] = std::make_shared<GPUKernel>();
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/SSBNPermutation/SortingPass.h");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->set_kernel_function_name("SSBNPermutationSortingPass");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->synchronize_options_with(m_compiler_options, {});

	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS] = std::make_shared<GPUKernel>();
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																								  "/SSBNPermutation/RetargetingPass.h");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->set_kernel_function_name("SSBNPermutationRetargetingPass");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->synchronize_options_with(m_compiler_options, {});

	Image32Bit blue_noise_texture = Image32Bit::read_image(SSBN_PERMUTATION_DATA_DIRECTORY "/blueNoiseTile_512x512_R=dither_GB=retarget.png", 3, false);

	std::vector<unsigned char> blue_noise_dither_data(blue_noise_texture.width * blue_noise_texture.height);
	std::vector<unsigned char> blue_noise_retargeting_data(blue_noise_texture.width * blue_noise_texture.height * 2);

	for (int i = 0; i < blue_noise_texture.width * blue_noise_texture.height; i++)
	{
		blue_noise_dither_data[i]			   = static_cast<unsigned char>(blue_noise_texture.get_pixel_ColorRGB32F(i).r * 255.0f);
		blue_noise_retargeting_data[i * 2]	   = static_cast<unsigned char>(blue_noise_texture.get_pixel_ColorRGB32F(i).g * 255.0f);
		blue_noise_retargeting_data[i * 2 + 1] = static_cast<unsigned char>(blue_noise_texture.get_pixel_ColorRGB32F(i).b * 255.0f);
	}

	m_blue_noise_dither_texture_buffer		= OrochiBuffer<unsigned char>(blue_noise_dither_data);
	m_blue_noise_retargeting_texture_buffer = OrochiBuffer<unsigned char>(blue_noise_retargeting_data);
}

void SSBNPermutationRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_sorted_seeds_buffer.resize(new_width * new_height);
}

bool SSBNPermutationRenderPass::pre_render_update(float delta_time)
{
	return false;
}

bool SSBNPermutationRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	return false;
}

void SSBNPermutationRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	unsigned char* blue_noise_texture_buffer_pointer = m_blue_noise_dither_texture_buffer.get_device_pointer();
	unsigned int* sorted_seeds_buffer_pointer		 = m_sorted_seeds_buffer.get_device_pointer();

	void* launch_args_sorting[] = { &render_data, &blue_noise_texture_buffer_pointer, &render_data.buffers.random_seeds };
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->launch_asynchronous(
							SSBNPermutationBlockSize, SSBNPermutationBlockSize, render_data.render_settings.render_resolution.x,
							render_data.render_settings.render_resolution.y, launch_args_sorting, m_renderer->get_main_stream());

	/*void* launch_args_retargeting[] = { &render_data, &sorted_seeds_buffer_pointer };
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->launch_asynchronous(
							SSBNPermutationBlockSize, SSBNPermutationBlockSize, render_data.render_settings.render_resolution.x,
							render_data.render_settings.render_resolution.y, launch_args_retargeting, m_renderer->get_main_stream());*/
}

void SSBNPermutationRenderPass::reset(bool reset_by_camera_movement) {}

void SSBNPermutationRenderPass::update_render_data() {}

bool SSBNPermutationRenderPass::is_render_pass_used() const
{
	return m_using_ssbn_permutation;
}

std::map<std::string, std::shared_ptr<GPUKernel>> SSBNPermutationRenderPass::get_tracing_kernels()
{
	return {};
}
