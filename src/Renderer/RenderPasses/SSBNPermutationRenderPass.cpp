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

SSBNPermutationRenderPass::SSBNPermutationRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(renderer, options, SSBNPermutationRenderPass::SSBN_PERMUTATION_RENDER_PASS_NAME)
{
	// Execute annealing simulation block for generating permutations
	if (false)
	{
		std::string blue_noise_texture_path = SSBN_PERMUTATION_DATA_DIRECTORY "/noise" + std::to_string(m_blue_noise_texture_width) + "x" +
											  std::to_string(m_blue_noise_texture_height) + ".png";
		std::string permutation_file_path_no_extension = SSBN_PERMUTATION_DATA_DIRECTORY "/permutation" + std::to_string(m_blue_noise_texture_width) + "x" +
														 std::to_string(m_blue_noise_texture_height) + "-r" + std::to_string(m_max_retargeting_radius);

		Image8Bit input_image = Image8Bit::read_image(blue_noise_texture_path, 1, false);

		SSBNPermutationSimulatedAnnealing annealing(input_image, m_max_retargeting_radius, 600);
		annealing.compute_permutation();
		annealing.write_permutations_to_file(permutation_file_path_no_extension + ".bin");
	}

	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS] = std::make_shared<GPUKernel>();
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/SSBNPermutation/SortingPass.h");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->set_kernel_function_name("SSBNPermutationSortingPass");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->synchronize_options_with(m_compiler_options, {});

	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS] = std::make_shared<GPUKernel>();
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																								  "/SSBNPermutation/RetargetingPass.h");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->set_kernel_function_name("SSBNPermutationRetargetingPass");
	m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->synchronize_options_with(m_compiler_options, {});

	reload_blue_noise_texture(m_blue_noise_texture_width, m_blue_noise_texture_height);
	reload_retargeting_data(m_max_retargeting_radius);
}

void SSBNPermutationRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used())
		return;

	unsigned int padded_width  = (new_width + m_blue_noise_texture_width - 1) / m_blue_noise_texture_width * m_blue_noise_texture_width;
	unsigned int padded_height = (new_height + m_blue_noise_texture_height - 1) / m_blue_noise_texture_height * m_blue_noise_texture_height;

	m_sorted_seeds_buffer.resize(padded_width * padded_height);
	m_screen_space_hash_grid_buffer.resize(new_width * new_height);
	m_screen_space_hash_grid_cell_offsets_buffer.resize(new_width * new_height);
}

void SSBNPermutationRenderPass::reload_blue_noise_texture(int new_width, int new_height)
{
	if (!is_render_pass_used())
		return;

	m_blue_noise_texture_width	= new_width;
	m_blue_noise_texture_height = new_height;

	std::string blue_noise_texture_path = SSBN_PERMUTATION_DATA_DIRECTORY "/noise" + std::to_string(m_blue_noise_texture_width) + "x" +
										  std::to_string(m_blue_noise_texture_height) + ".png";

	Image8Bit blue_noise_texture = Image8Bit::read_image(blue_noise_texture_path, 1, false);
	if (blue_noise_texture.width != m_blue_noise_texture_width || blue_noise_texture.height != m_blue_noise_texture_height)
	{
		std::cerr << "Error: Blue noise texture has wrong dimensions. Expected " << m_blue_noise_texture_width << "x" << m_blue_noise_texture_height << ", got "
				  << blue_noise_texture.width << "x" << blue_noise_texture.height << std::endl;

		throw std::runtime_error("Error: Blue noise texture has wrong dimensions. Expected " + std::to_string(m_blue_noise_texture_width) + "x" +
								 std::to_string(m_blue_noise_texture_height) + ", got " + std::to_string(blue_noise_texture.width) + "x" +
								 std::to_string(blue_noise_texture.height));
	}

	std::vector<unsigned char> blue_noise_dither_data(blue_noise_texture.width * blue_noise_texture.height);
	for (int i = 0; i < blue_noise_texture.width * blue_noise_texture.height; i++)
		blue_noise_dither_data[i] = blue_noise_texture.data()[i * blue_noise_texture.channels + 0];

	m_blue_noise_dither_texture_buffer = OrochiBuffer<unsigned char>(blue_noise_dither_data);

	unsigned int render_resolution_x		= m_renderer->get_render_data().render_settings.render_resolution.x;
	unsigned int render_resolution_y		= m_renderer->get_render_data().render_settings.render_resolution.y;
	unsigned int padded_render_resolution_x = (render_resolution_x + m_blue_noise_texture_width - 1) / m_blue_noise_texture_width * m_blue_noise_texture_width;
	unsigned int padded_render_resolution_y =
							(render_resolution_y + m_blue_noise_texture_height - 1) / m_blue_noise_texture_height * m_blue_noise_texture_height;

	m_sorted_seeds_buffer.resize(padded_render_resolution_x * padded_render_resolution_y);

	reload_retargeting_data(m_max_retargeting_radius);
}

void SSBNPermutationRenderPass::reload_retargeting_data(int new_max_retargeting_radius)
{
	if (!is_render_pass_used())
		return;

	m_max_retargeting_radius = new_max_retargeting_radius;

	std::string permutation_file_path_no_extension = get_permutation_file_path_no_extension();

	std::ifstream blue_noise_retargeting_file(permutation_file_path_no_extension + ".bin", std::ios::binary);
	if (!blue_noise_retargeting_file.is_open())
	{
		std::cerr << "Error opening blue noise retargeting file: " << permutation_file_path_no_extension + ".bin" << std::endl;

		throw std::runtime_error("Error opening blue noise retargeting file: " + permutation_file_path_no_extension + ".bin");
	}

	std::vector<int> blue_noise_retargeting_data(m_blue_noise_texture_width * m_blue_noise_texture_height);
	blue_noise_retargeting_file.read(reinterpret_cast<char*>(blue_noise_retargeting_data.data()),
									 m_blue_noise_texture_width * m_blue_noise_texture_height * sizeof(int));

	m_blue_noise_retargeting_texture_buffer = OrochiBuffer<int>(blue_noise_retargeting_data);
}

std::string SSBNPermutationRenderPass::get_permutation_file_path_no_extension(int retarget_radius)
{
	if (retarget_radius == -1)
		retarget_radius = m_max_retargeting_radius;

	return SSBN_PERMUTATION_DATA_DIRECTORY "/permutation" + std::to_string(m_blue_noise_texture_width) + "x" + std::to_string(m_blue_noise_texture_height) +
		   "-r" + std::to_string(retarget_radius);
}

bool SSBNPermutationRenderPass::pre_render_update(float delta_time)
{
	bool updated = false;
	if (!is_render_pass_used() && m_sorted_seeds_buffer.size() != 0)
	{
		m_sorted_seeds_buffer.free();
		m_screen_space_hash_grid_buffer.free();
		m_screen_space_hash_grid_cell_offsets_buffer.free();

		updated = true;
	}
	else if (m_sorted_seeds_buffer.size() == 0)
	{
		unsigned int resolution_x = m_renderer->get_render_data().render_settings.render_resolution.x;
		unsigned int resolution_y = m_renderer->get_render_data().render_settings.render_resolution.y;

		unsigned int padded_width = (resolution_x + m_blue_noise_texture_width - 1) /
									m_blue_noise_texture_width * m_blue_noise_texture_width;
		unsigned int padded_height = (resolution_y + m_blue_noise_texture_height - 1) /
									 m_blue_noise_texture_height * m_blue_noise_texture_height;

		m_sorted_seeds_buffer.resize(padded_width * padded_height);
		m_screen_space_hash_grid_buffer.resize(padded_width * padded_height);
		m_screen_space_hash_grid_cell_offsets_buffer.resize(resolution_x * resolution_y);

		updated = true;
	}

	return updated;
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
	unsigned int padded_render_solution_x = (render_data.render_settings.render_resolution.x + m_blue_noise_texture_width - 1) / m_blue_noise_texture_width *
											m_blue_noise_texture_width;
	unsigned int padded_render_solution_y = (render_data.render_settings.render_resolution.y + m_blue_noise_texture_height - 1) / m_blue_noise_texture_height *
											m_blue_noise_texture_height;

	unsigned int different_hash_count = 0;
	{
		// CPU sorting for now
		std::vector<uint3_t> screen_space_hash_grid_data = m_screen_space_hash_grid_buffer.download_data();
		std::sort(screen_space_hash_grid_data.begin(), screen_space_hash_grid_data.end(), [](const uint3_t& a, const uint3_t& b) { return a.x < b.x; });

		// Computing the number of hash cells and the offsets of each cell in the sorted hash grid
		std::vector<int> offsets(padded_render_solution_x * padded_render_solution_y);
		offsets[0] = 0;
		for (size_t i = 1; i < screen_space_hash_grid_data.size(); i++)
		{
			if (screen_space_hash_grid_data[i].x != screen_space_hash_grid_data[i - 1].x)
			{
				different_hash_count++;
				offsets[different_hash_count] = i;
			}
		}

		m_screen_space_hash_grid_buffer.upload_data(screen_space_hash_grid_data);
		m_screen_space_hash_grid_cell_offsets_buffer.upload_data(offsets);
	}

	unsigned int block_size = compiler_options.get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE);

	if (m_do_retargeting)
	{
		int* hash_grid_offsets_buffer_pointer = m_screen_space_hash_grid_cell_offsets_buffer.get_device_pointer();
		void* launch_args_sorting[]			  = { &render_data,
												  &blue_noise_texture_buffer_pointer,
												  &m_blue_noise_texture_width,
												  &m_blue_noise_texture_height,
												  &render_data.buffers.get_input_random_seeds_pointer(),
												  &render_data.buffers.get_input_random_seeds_pointer(),
												  &hash_grid_offsets_buffer_pointer };

		m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->launch_asynchronous(block_size * block_size, 1,
																								 block_size * block_size * different_hash_count, 1,
																								 launch_args_sorting, m_renderer->get_main_stream());

		int* blue_noise_retargeting_texture_buffer_pointer = m_blue_noise_retargeting_texture_buffer.get_device_pointer();
		void* launch_args_retargeting[]					   = { &render_data,
															   &blue_noise_retargeting_texture_buffer_pointer,
															   &m_blue_noise_texture_width,
															   &m_blue_noise_texture_height,
															   &sorted_seeds_buffer_pointer,
															   &render_data.buffers.get_input_random_seeds_pointer() };

		m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS]->launch_asynchronous(32, 32, padded_render_solution_x, padded_render_solution_y,
																									 launch_args_retargeting, m_renderer->get_main_stream());
	}
	else
	{
		int* hash_grid_offsets_buffer_pointer = m_screen_space_hash_grid_cell_offsets_buffer.get_device_pointer();
		void* launch_args_sorting[]			  = { &render_data,
												  &blue_noise_texture_buffer_pointer,
												  &m_blue_noise_texture_width,
												  &m_blue_noise_texture_height,
												  &render_data.buffers.get_input_random_seeds_pointer(),
												  &render_data.buffers.get_input_random_seeds_pointer(),
												  &hash_grid_offsets_buffer_pointer };

		m_kernels[SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS]->launch_asynchronous(block_size * block_size, 1,
																								 block_size * block_size * different_hash_count, 1,
																								 launch_args_sorting, m_renderer->get_main_stream());
	}
}

void SSBNPermutationRenderPass::reset(bool reset_by_camera_movement) {}

void SSBNPermutationRenderPass::update_render_data()
{
	if (!is_render_pass_used())
		return;

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.ssbn_settings.blue_noise_texture_width	= m_blue_noise_texture_width;
	render_data.ssbn_settings.blue_noise_texture_height = m_blue_noise_texture_height;
	render_data.ssbn_settings.screen_space_hash_grid	= m_screen_space_hash_grid_buffer.get_device_pointer();
}

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

int& SSBNPermutationRenderPass::get_blue_noise_texture_width()
{
	return m_blue_noise_texture_width;
}

int& SSBNPermutationRenderPass::get_blue_noise_texture_height()
{
	return m_blue_noise_texture_height;
}

int& SSBNPermutationRenderPass::get_max_retargeting_radius()
{
	return m_max_retargeting_radius;
}
