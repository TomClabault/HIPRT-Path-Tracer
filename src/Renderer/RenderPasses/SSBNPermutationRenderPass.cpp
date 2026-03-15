/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/SSBNPermutationSimulatedAnnealing.h"
#include "Renderer/RenderPasses/SSBNPermutationRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_RENDER_PASS_NAME = "SSBN Permutation Render Pass";
const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS	   = "SSBN Permutation Sorting Pass";
const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS = "SSBN Permutation Retargeting Pass";

const unsigned int SSBNPermutationRenderPass::SSBN_PERMUTATION_BLUE_NOISE_TEXTURE_WIDTH	 = 512;
const unsigned int SSBNPermutationRenderPass::SSBN_PERMUTATION_BLUE_NOISE_TEXTURE_HEIGHT = 512;

SSBNPermutationRenderPass::SSBNPermutationRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(renderer, options, SSBNPermutationRenderPass::SSBN_PERMUTATION_RENDER_PASS_NAME)
{
	int max_radius						= 15;
	std::string blue_noise_texture_path = SSBN_PERMUTATION_DATA_DIRECTORY "/noise" + std::to_string(SSBN_PERMUTATION_BLUE_NOISE_TEXTURE_WIDTH) + "x" +
										  std::to_string(SSBN_PERMUTATION_BLUE_NOISE_TEXTURE_HEIGHT) + ".png";
	std::string permutation_file_path_no_extension = SSBN_PERMUTATION_DATA_DIRECTORY "/permutation" +
													 std::to_string(SSBN_PERMUTATION_BLUE_NOISE_TEXTURE_WIDTH) + "x" +
													 std::to_string(SSBN_PERMUTATION_BLUE_NOISE_TEXTURE_HEIGHT) + "-r" + std::to_string(max_radius);

	/*Image8Bit input_image = Image8Bit::read_image(blue_noise_texture_path, 1, false);

	SSBNPermutationSimulatedAnnealing annealing(input_image, 15);
	annealing.compute_permutation();
	annealing.write_permutations_to_file(permutation_file_path_no_extension + ".bin");
	annealing.write_permutation_visualization_image(permutation_file_path_no_extension + ".png");*/

	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS] = std::make_shared<GPUKernel>();
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/SSBNPermutation/SortingPass.h");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->set_kernel_function_name("SSBNPermutationSortingPass");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->synchronize_options_with(m_compiler_options, {});

	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS] = std::make_shared<GPUKernel>();
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																								  "/SSBNPermutation/RetargetingPass.h");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->set_kernel_function_name("SSBNPermutationRetargetingPass");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->synchronize_options_with(m_compiler_options, {});

	Image8Bit blue_noise_texture = Image8Bit::read_image(blue_noise_texture_path, 1, false);

	std::vector<unsigned char> blue_noise_dither_data(blue_noise_texture.width * blue_noise_texture.height);
	for (int i = 0; i < blue_noise_texture.width * blue_noise_texture.height; i++)
		blue_noise_dither_data[i] = static_cast<unsigned char>(std::round(blue_noise_texture.data()[i * blue_noise_texture.channels + 0]));

	std::ifstream blue_noise_retargeting_file(permutation_file_path_no_extension + ".bin", std::ios::binary);
	if (!blue_noise_retargeting_file.is_open())
	{
		std::cerr << "Error opening blue noise retargeting file: " << permutation_file_path_no_extension + ".bin" << std::endl;

		throw std::runtime_error("Error opening blue noise retargeting file: " + permutation_file_path_no_extension + ".bin");
	}

	std::vector<int> blue_noise_retargeting_data(blue_noise_texture.width * blue_noise_texture.height);
	blue_noise_retargeting_file.read(reinterpret_cast<char*>(blue_noise_retargeting_data.data()),
									 blue_noise_texture.width * blue_noise_texture.height * sizeof(int));

	m_blue_noise_dither_texture_buffer		= OrochiBuffer<unsigned char>(blue_noise_dither_data);
	m_blue_noise_retargeting_texture_buffer = OrochiBuffer<int>(blue_noise_retargeting_data);
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
	if (!is_render_pass_used())
		return;

	unsigned char* blue_noise_texture_buffer_pointer = m_blue_noise_dither_texture_buffer.get_device_pointer();
	unsigned int* sorted_seeds_buffer_pointer		 = m_sorted_seeds_buffer.get_device_pointer();
	unsigned int blue_noise_texture_width			 = SSBN_PERMUTATION_BLUE_NOISE_TEXTURE_WIDTH;
	unsigned int blue_noise_texture_height			 = SSBN_PERMUTATION_BLUE_NOISE_TEXTURE_HEIGHT;

	unsigned int block_size = compiler_options.get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE);
	if (m_do_retargeting)
	{
		void* launch_args_sorting[] = { &render_data,
										&blue_noise_texture_buffer_pointer,
										&blue_noise_texture_width,
										&blue_noise_texture_height,
										&render_data.buffers.get_input_random_seeds_pointer(),
										&sorted_seeds_buffer_pointer };

		m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->launch_asynchronous(
								block_size, block_size, render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y,
								launch_args_sorting, m_renderer->get_main_stream());

		int* blue_noise_retargeting_texture_buffer_pointer = m_blue_noise_retargeting_texture_buffer.get_device_pointer();
		void* launch_args_retargeting[]					   = { &render_data,
															   &blue_noise_retargeting_texture_buffer_pointer,
															   &blue_noise_texture_width,
															   &blue_noise_texture_height,
															   &sorted_seeds_buffer_pointer,
															   &render_data.buffers.get_input_random_seeds_pointer() };
		m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->launch_asynchronous(32, 32, render_data.render_settings.render_resolution.x,
																									 render_data.render_settings.render_resolution.y,
																									 launch_args_retargeting, m_renderer->get_main_stream());
	}
	else
	{
		void* launch_args_sorting[] = { &render_data,
										&blue_noise_texture_buffer_pointer,
										&blue_noise_texture_width,
										&blue_noise_texture_height,
										&render_data.buffers.get_input_random_seeds_pointer(),
										&render_data.buffers.get_input_random_seeds_pointer() };

		m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->launch_asynchronous(
								block_size, block_size, render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y,
								launch_args_sorting, m_renderer->get_main_stream());
	}
}

void SSBNPermutationRenderPass::reset(bool reset_by_camera_movement) {}

void SSBNPermutationRenderPass::update_render_data() {}

bool SSBNPermutationRenderPass::is_render_pass_used() const
{
	return m_compiler_options->get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_ENABLED) == KERNEL_OPTION_TRUE;
}

bool& SSBNPermutationRenderPass::get_do_retargeting()
{
	return m_do_retargeting;
}

std::map<std::string, std::shared_ptr<GPUKernel>> SSBNPermutationRenderPass::get_tracing_kernels()
{
	return {};
}
