/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"
#include "UI/RenderWindow.h"

const std::string MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME = "Megakernel Render Pass";
const std::string MegaKernelRenderPass::MEGAKERNEL_KERNEL = "Megakernel (1 SPP)";
const std::string MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION = "Megakernel (ReGIR interactivity)";

MegaKernelRenderPass::MegaKernelRenderPass() : MegaKernelRenderPass(nullptr) {}
MegaKernelRenderPass::MegaKernelRenderPass(GPURenderer* renderer) : MegaKernelRenderPass(renderer, MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME) {}
MegaKernelRenderPass::MegaKernelRenderPass(GPURenderer* renderer, const std::string& name) : RenderPass(renderer, name)
{
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL] = std::make_shared<GPUKernel>();
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Megakernel.h");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_function_name("MegaKernel");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->synchronize_options_with(m_renderer->get_global_compiler_options(), GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	std::unordered_set<std::string> options_not_synchronized = GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED;
	options_not_synchronized.insert(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION] = std::make_shared<GPUKernel>();
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Megakernel.h");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->set_kernel_function_name("MegaKernel");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->synchronize_options_with(m_renderer->get_global_compiler_options(), options_not_synchronized);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, LSS_BASE_LIGHT_TREE_SG);
}

bool MegaKernelRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool updated = false;

	if (!m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->has_been_compiled())
	{
		updated = true;
		m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->has_been_compiled() && m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR)
	{
		updated = true;
		m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	return updated;
}

void MegaKernelRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool MegaKernelRenderPass::pre_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used())
		return false;

	// Resetting this flag as this is a new frame
	render_data.render_settings.do_update_status_buffers = false;

	if (!render_data.render_settings.accumulate)
		render_data.render_settings.sample_number = 0;

	return false;
}

bool MegaKernelRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return false;

	render_data.random_number = m_renderer->get_rng_generator().xorshift32();

	void* launch_args[] = { &render_data };

	if (compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR && m_render_window->is_interacting())
		// If we're using ReGIR and we're interacting with the camera, using another kernel which uses a light
		// tree for light sampling because ReGIR isn't good for interactivity
		m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_renderer->get_main_stream());
	else
		m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_renderer->get_main_stream());

	return true;
}

void MegaKernelRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used())
		return;

	if (render_data.render_settings.accumulate)
		if (m_renderer->get_application_settings()->auto_sample_per_frame)
			render_data.render_settings.samples_per_frame = 1;

	render_data.render_settings.denoiser_AOV_accumulation_counter = 0;

	render_data.render_settings.sample_number = 0;
}

bool MegaKernelRenderPass::is_render_pass_used() const
{
	// Only active if we're not using ReSTIR GI because if we are using ReSTIR, the path tracing is done in
	// the initial candidates kernel
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) != PSS_RESTIR_GI;
}

std::map<std::string, std::shared_ptr<GPUKernel>> MegaKernelRenderPass::get_all_kernels()
{
	std::map<std::string, std::shared_ptr<GPUKernel>> kernels = m_kernels;

	if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) != LSS_BASE_REGIR)
		kernels.erase(MegaKernelRenderPass::MEGAKERNEL_KERNEL_REGIR_INTERACTION);

	return kernels;
}
