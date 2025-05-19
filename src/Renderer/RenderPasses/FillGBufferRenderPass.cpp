/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/FillGBufferRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

#include <memory>

const std::string FillGBufferRenderPass::FILL_GBUFFER_RENDER_PASS_NAME = "Fill G-Buffer Render Pass";
const std::string FillGBufferRenderPass::FILL_GBUFFER_KERNEL = "Fill G-Buffer";

FillGBufferRenderPass::FillGBufferRenderPass() : FillGBufferRenderPass(nullptr) {}
FillGBufferRenderPass::FillGBufferRenderPass(GPURenderer* renderer) : RenderPass(renderer, FillGBufferRenderPass::FILL_GBUFFER_RENDER_PASS_NAME)
{
	m_render_resolution = m_renderer->m_render_resolution;

	m_kernels[FillGBufferRenderPass::FILL_GBUFFER_KERNEL] = std::make_shared<GPUKernel>();
	m_kernels[FillGBufferRenderPass::FILL_GBUFFER_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/CameraRays.h");
	m_kernels[FillGBufferRenderPass::FILL_GBUFFER_KERNEL]->set_kernel_function_name("CameraRays");
	m_kernels[FillGBufferRenderPass::FILL_GBUFFER_KERNEL]->synchronize_options_with(m_renderer->get_global_compiler_options(), GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[FillGBufferRenderPass::FILL_GBUFFER_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[FillGBufferRenderPass::FILL_GBUFFER_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
}

void FillGBufferRenderPass::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	// Configuring the kernel that will be used to retrieve the size of the RayVolumeState structure.
	// This size will be needed to resize the 'ray_volume_states' buffer in the GBuffer if the nested dielectrics
	// stack size changes
	//
	// We're compiling it serially so that we're sure that we can retrieve the RayVolumeState size on the GPU after the
	// GPURenderer is constructed (because this renderer pass is compiled during the construction of the GPURenderer)

	m_ray_volume_state_byte_size_kernel = std::make_shared<GPUKernel>();
	m_ray_volume_state_byte_size_kernel->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Utils/RayVolumeStateSize.h");
	m_ray_volume_state_byte_size_kernel->set_kernel_function_name("RayVolumeStateSize");
	m_ray_volume_state_byte_size_kernel->synchronize_options_with(m_renderer->get_global_compiler_options(), GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	ThreadManager::start_serial_thread(ThreadManager::COMPILE_RAY_VOLUME_STATE_SIZE_KERNEL_KEY, ThreadFunctions::compile_kernel_silent, m_ray_volume_state_byte_size_kernel, hiprt_orochi_ctx, std::ref(func_name_sets));

	RenderPass::compile(hiprt_orochi_ctx, func_name_sets);
}

void FillGBufferRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_g_buffer.resize(new_width * new_height, get_ray_volume_state_byte_size());

	if (m_renderer->get_render_data().render_settings.use_prev_frame_g_buffer(m_renderer))
		m_g_buffer_prev_frame.resize(new_width * new_height, get_ray_volume_state_byte_size());

	m_render_resolution = m_renderer->m_render_resolution;
}

bool FillGBufferRenderPass::pre_render_update(float delta_time)
{
	if (m_renderer->get_ReGIR_render_pass()->is_render_pass_used())
		m_renderer->get_ReGIR_render_pass()->update_cell_alive_count();

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (render_data.render_settings.use_prev_frame_g_buffer(m_renderer))
	{
		// If at least one buffer has a size of 0, we assume that this means that the whole G-buffer is deallocated
		// and so we're going to have to reallocate it
		bool prev_frame_g_buffer_needs_resize = m_g_buffer_prev_frame.first_hit_prim_index.size() == 0;

		if (prev_frame_g_buffer_needs_resize)
		{
			m_g_buffer_prev_frame.resize(m_render_resolution.x * m_render_resolution.y, get_ray_volume_state_byte_size());
			return true;
		}
	}
	else
	{
		// If we're not using the G-buffer, indicating that in use_last_frame_g_buffer so that the shader doesn't
		// try to use it

		if (m_g_buffer_prev_frame.first_hit_prim_index.size() > 0)
		{
			// If the buffers aren't freed already
			m_g_buffer_prev_frame.free();
			return true;
		}
	}

	return false;
}

bool FillGBufferRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	render_data.random_number = m_renderer->get_rng_generator().xorshift32();

	void* launch_args[] = { &render_data };

	m_kernels[FillGBufferRenderPass::FILL_GBUFFER_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_renderer->get_main_stream());

	return true;
}

void FillGBufferRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.g_buffer = m_g_buffer.get_device_g_buffer();

	if (render_data.render_settings.use_prev_frame_g_buffer(m_renderer))
		// Only setting the pointers of the buffers if we're actually using the g-buffer of the previous frame
		render_data.g_buffer_prev_frame = m_g_buffer_prev_frame.get_device_g_buffer();
	else
	{
		render_data.g_buffer_prev_frame.materials = nullptr;
		render_data.g_buffer_prev_frame.geometric_normals = nullptr;
		render_data.g_buffer_prev_frame.shading_normals = nullptr;
		render_data.g_buffer_prev_frame.primary_hit_position = nullptr;
	}
}

size_t FillGBufferRenderPass::get_ray_volume_state_byte_size()
{
	OrochiBuffer<size_t> out_size_buffer(1);
	size_t* out_size_buffer_pointer = out_size_buffer.get_device_pointer();

	ThreadManager::join_threads(ThreadManager::COMPILE_RAY_VOLUME_STATE_SIZE_KERNEL_KEY);

	void* launch_args[] = { &out_size_buffer_pointer };
	m_ray_volume_state_byte_size_kernel->launch_synchronous(1, 1, 1, 1, launch_args, 0);
	OROCHI_CHECK_ERROR(oroStreamSynchronize(0));

	//std::cout << out_size_buffer.download_data()[0] << " GPU";
	//std::cout << sizeof(RayVolumeState) << " CPU" << std::endl;
	//std::exit(0);
	//return 0;
	size_t size = out_size_buffer.download_data()[0];

	return size;
}

void FillGBufferRenderPass::resize_g_buffer_ray_volume_states()
{
	m_renderer->synchronize_all_kernels();

	m_g_buffer.ray_volume_states.resize(m_render_resolution.x * m_render_resolution.y, get_ray_volume_state_byte_size());
	if (m_renderer->get_render_data().render_settings.use_prev_frame_g_buffer())
		m_g_buffer_prev_frame.ray_volume_states.resize(m_render_resolution.x * m_render_resolution.y, get_ray_volume_state_byte_size());

	m_renderer->invalidate_render_data_buffers();
}
