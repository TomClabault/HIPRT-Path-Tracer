/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/Baker/GPUBakerKernel.h"
#include "Renderer/Baker/GPUBakerConstants.h"
#include "Threads/ThreadManager.h"

GPUBakerKernel::GPUBakerKernel(std::shared_ptr<GPURenderer> renderer, oroStream_t bake_stream, std::shared_ptr<std::mutex> compiler_priority_mutex,
	const std::string& kernel_filepath, const std::string& kernel_function, const std::string& kernel_title)
{
	m_renderer = renderer;
	m_bake_stream = bake_stream;
	m_compiler_priority_mutex = compiler_priority_mutex;

	m_kernel_filepath = kernel_filepath;
	m_kernel_function = kernel_function;
	m_kernel_title = kernel_title;
}

void GPUBakerKernel::bake_internal(int3 bake_resolution, const void* bake_settings_pointer, int nb_kernel_iterations, std::string output_filename)
{
	m_bake_complete = false;

	struct ThreadData
	{
		int3 bake_resolution;
		const void* bake_settings_pointer;
		std::string output_filename;
	};

	// Allocating that on the heap so that it stays alive for the thread even
	// when we return from this function
	std::shared_ptr<ThreadData> thread_data = std::make_shared<ThreadData>();
	thread_data->bake_resolution = bake_resolution;
	thread_data->bake_settings_pointer = bake_settings_pointer;
	thread_data->output_filename = output_filename;

	// Starting everything on a thread to avoid blocking to UI (during the compilation
	// of the kernel mainly)
	ThreadManager::start_thread("kernel_bake_" + m_kernel_title, [this, thread_data, nb_kernel_iterations] {
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

		int3& bake_resolution = thread_data->bake_resolution;
		const void* bake_settings_pointer = thread_data->bake_settings_pointer;
		std::string& output_filename = thread_data->output_filename;

		if (!m_bake_kernel.has_been_compiled())
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "%s", ("Compiling " + m_kernel_title + " kernel...").c_str());

			// Taking the priority for the compilation as otherwise, the kernels
			// precompiling in the background are going to have the hand and we'll
			// never be able to compile our bake kernel and we'll never start baking
			// (until all kernels are precompiled in background of course but that's
			// going to take a long time)
			std::lock_guard<std::mutex> lock(*m_compiler_priority_mutex);
			m_renderer->take_kernel_compilation_priority();

			m_bake_kernel = GPUKernel(m_kernel_filepath, m_kernel_function);
			m_bake_kernel.compile(m_renderer->get_hiprt_orochi_ctx());

			m_renderer->release_kernel_compilation_priority();
		}

		m_bake_buffer.resize(bake_resolution.x * bake_resolution.y * bake_resolution.z);

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "%s", ("Launching " + m_kernel_title + " baking...").c_str());

		int3 tile_size;
		if (bake_resolution.z > 1)
			// 3D launch
			tile_size = make_int3(4, 4, 4);
		else
			// 2D launch
			tile_size = make_int3(8, 8, 1);

		float kernel_duration;
		oroEvent_t start, stop;
		OROCHI_CHECK_ERROR(oroEventCreate(&start));
		OROCHI_CHECK_ERROR(oroEventCreate(&stop));
		OROCHI_CHECK_ERROR(oroEventRecord(start, m_bake_stream));

		// Zeroing the buffer that we're going to accumulate the bake data into
		m_bake_buffer.memset_whole_buffer(0);

		// Launching many "small" kernels to avoid driver timeouts
		int iterations_per_kernel = floor(hippt::max(1.0f, (float)GPUBakerConstants::COMPUTE_ELEMENT_PER_BAKE_KERNEL_LAUNCH / (bake_resolution.x * bake_resolution.y * bake_resolution.z)));
		int nb_kernel_launch = ceil(nb_kernel_iterations / (float)iterations_per_kernel);

		void* non_const_setting = const_cast<void*>(bake_settings_pointer);
		for (int i = 0; i < nb_kernel_launch; i++)
		{
			// The current iteration variable is used in the kernel to shuffle the random
			// so that we get different random numbers per each kernel launch
			int current_iteration = i + 1;
			float* device_buffer = m_bake_buffer.get_device_pointer();
			void* bake_kernel_launch_args[] = { &iterations_per_kernel, &current_iteration, non_const_setting, &device_buffer };
			m_bake_kernel.launch_asynchronous_3D(
				tile_size.x, tile_size.y, tile_size.z,
				bake_resolution.x, bake_resolution.y, bake_resolution.z,
				bake_kernel_launch_args, m_bake_stream);
		}

		OROCHI_CHECK_ERROR(oroEventRecord(stop, m_bake_stream));
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_bake_stream));
		OROCHI_CHECK_ERROR(oroEventElapsedTime(&kernel_duration, start, stop));
		OROCHI_CHECK_ERROR(oroEventDestroy(start));
		OROCHI_CHECK_ERROR(oroEventDestroy(stop));

		std::string unit_suffix = kernel_duration < 1000.0f ? "ms!" : "s!";
		kernel_duration = kernel_duration > 1000.0f ? kernel_duration / 1000.0f : kernel_duration;
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "%s", (m_kernel_title + " completed in " + std::to_string(kernel_duration) + unit_suffix).c_str());

		if (bake_resolution.z > 1)
		{
			// 3D texture
			std::vector<float> baked_data = m_bake_buffer.download_data();
			for (int i = 0; i < bake_resolution.z; i++)
			{
				Image32Bit image = Image32Bit(baked_data.data() + i * bake_resolution.x * bake_resolution.y, bake_resolution.x, bake_resolution.y, /* nb channels */ 1);

				std::string final_filename = std::to_string(i) + output_filename;
				image.write_image_hdr(final_filename.c_str(), false);
			}
		}
		else
		{
			// A single 2D image
			std::vector<float> baked_data = m_bake_buffer.download_data();
			Image32Bit image = Image32Bit(baked_data, bake_resolution.x, bake_resolution.y, 1);
			image.write_image_hdr(output_filename.c_str(), false);
		}


		m_bake_buffer.free();
		m_bake_complete = true;
	});

	ThreadManager::detach_threads("kernel_bake_" + m_kernel_title);
}

bool GPUBakerKernel::is_complete() const
{
	return m_bake_complete;
}
