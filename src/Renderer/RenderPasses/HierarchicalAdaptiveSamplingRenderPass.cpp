/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/RenderPasses/HierarchicalAdaptiveSamplingRenderPass.h"

#include "Renderer/GPURenderer.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

#include <algorithm>

const std::string HierarchicalAdaptiveSamplingRenderPass::RENDER_PASS_NAME		 = "Hierarchical Adaptive Sampling";
const std::string HierarchicalAdaptiveSamplingRenderPass::COMPUTE_ERROR_KERNEL	 = "Hierarchical adaptive sampling compute error";
const std::string HierarchicalAdaptiveSamplingRenderPass::BUILD_ROWS_KERNEL		 = "Hierarchical adaptive sampling build SAT rows";
const std::string HierarchicalAdaptiveSamplingRenderPass::BUILD_COLUMNS_KERNEL	 = "Hierarchical adaptive sampling build SAT columns";
const std::string HierarchicalAdaptiveSamplingRenderPass::BUILD_HIERARCHY_KERNEL = "Hierarchical adaptive sampling build hierarchy";
const std::string HierarchicalAdaptiveSamplingRenderPass::RESOLVE_MASK_KERNEL	 = "Hierarchical adaptive sampling resolve mask";

HierarchicalAdaptiveSamplingRenderPass::HierarchicalAdaptiveSamplingRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(HierarchicalAdaptiveSamplingRenderPass::RENDER_PASS_NAME, renderer, options)
{
	m_render_data_host_pinned.resize_host_pinned_mem(1);

	configure_kernel(COMPUTE_ERROR_KERNEL, "HierarchicalAdaptiveSamplingComputeError", "ComputeError.h");
	configure_kernel(BUILD_ROWS_KERNEL, "HierarchicalAdaptiveSamplingBuildRows", "BuildRows.h");
	configure_kernel(BUILD_COLUMNS_KERNEL, "HierarchicalAdaptiveSamplingBuildColumns", "BuildColumns.h");
	configure_kernel(BUILD_HIERARCHY_KERNEL, "HierarchicalAdaptiveSamplingBuildHierarchy", "BuildHierarchy.h");
	configure_kernel(RESOLVE_MASK_KERNEL, "HierarchicalAdaptiveSamplingResolveMask", "ResolveMask.h");
}

void HierarchicalAdaptiveSamplingRenderPass::configure_kernel(const std::string& kernel_id,
															  const std::string& function_name,
															  const std::string& kernel_file_name)
{
	m_kernels[kernel_id] = std::make_shared<GPUKernel>(this->get_name() + "::" + kernel_id);
	m_kernels[kernel_id]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/HierarchicalAdaptiveSampling/" + kernel_file_name);
	m_kernels[kernel_id]->set_kernel_function_name(function_name);
	m_kernels[kernel_id]->synchronize_options_with(m_compiler_options, {});
}

void HierarchicalAdaptiveSamplingRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool HierarchicalAdaptiveSamplingRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
																		  const std::vector<hiprtFuncNameSet>& func_name_sets,
																		  bool silent,
																		  bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool updated = false;
	for (std::map<std::string, std::shared_ptr<GPUKernel>>::value_type& name_to_kernel : m_kernels)
	{
		if (name_to_kernel.second->has_been_compiled())
			continue;

		name_to_kernel.second->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
		updated = true;
	}

	return updated;
}

void HierarchicalAdaptiveSamplingRenderPass::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
													   const std::vector<hiprtFuncNameSet>& func_name_sets,
													   bool silent,
													   bool use_cache)
{
	// Recompile this pass even when it is disabled. Its HIPRTRenderData ABI and compiler options must
	// remain synchronized before the user enables it after changing another rendering option.
	for (std::map<std::string, std::shared_ptr<GPUKernel>>::value_type& name_to_kernel : m_kernels)
		ThreadManager::start_thread(ThreadManager::COMPILE_KERNELS_THREAD_KEY, ThreadFunctions::compile_kernel, name_to_kernel.second, hiprt_orochi_ctx,
									std::ref(func_name_sets));
}

bool HierarchicalAdaptiveSamplingRenderPass::pre_frame_render_update(float delta_time)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_data().render_settings;
	if (!is_render_pass_used(*m_compiler_options))
	{
		bool had_buffers = m_error.size() != 0;
		free_buffers();

		return had_buffers;
	}

	bool resized						   = false;
	unsigned int pixel_count			   = render_settings.render_resolution.x * render_settings.render_resolution.y;
	unsigned int requested_node_capacity   = static_cast<unsigned int>(std::max(1, render_settings.hierarchical_adaptive_sampling_max_node_count));
	unsigned int maximum_useful_node_count = pixel_count > 0 ? pixel_count * 2u - 1u : 1u;
	unsigned int node_capacity			   = std::min(requested_node_capacity, maximum_useful_node_count);
	unsigned int maximum_depth			   = static_cast<unsigned int>(std::max(1, render_settings.hierarchical_adaptive_sampling_max_depth));
	unsigned int build_command_count	   = maximum_depth * 2u + 4u;

	if (m_build_depths_host_pinned.size() != build_command_count)
	{
		m_build_depths_host_pinned.resize_host_pinned_mem(build_command_count);

		unsigned int* build_depths = m_build_depths_host_pinned.get_host_pinned_pointer();
		build_depths[0]			   = static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::INITIALIZE);

		for (unsigned int depth = 0; depth <= maximum_depth; depth++)
		{
			build_depths[depth * 2u + 1u] = static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::PREPARE_LEVEL);
			build_depths[depth * 2u + 2u] = depth;
		}

		build_depths[build_command_count - 1u] = static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::FINALIZE);

		resized = true;
	}

	if (m_error.size() != pixel_count)
	{
		m_error.resize(pixel_count);
		m_error.memset_whole_buffer(0);
		m_summed_area.resize(pixel_count);
		resized = true;
	}

	if (m_nodes.size() != node_capacity)
	{
		m_nodes.resize(node_capacity);
		resized = true;
	}

	if (m_node_count.size() == 0)
	{
		m_node_count.resize(1);
		resized = true;
	}

	if (m_level_node_count.size() == 0)
	{
		m_level_node_count.resize(1);
		resized = true;
	}

	return resized;
}

void HierarchicalAdaptiveSamplingRenderPass::free_buffers()
{
	if (m_error.size() != 0)
	{
		m_error.free();
		m_summed_area.free();
		m_nodes.free();
		m_node_count.free();
		m_level_node_count.free();
	}
}

void HierarchicalAdaptiveSamplingRenderPass::upload_render_data(const std::string& kernel_id, HIPRTRenderData& render_data)
{
	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	*host_pinned_render_data				 = render_data;

	m_kernels[kernel_id]->upload_to_module_global("HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData),
												  m_renderer->get_main_stream());
}

bool HierarchicalAdaptiveSamplingRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options) || !render_data.render_settings.use_hierarchical_adaptive_sampling())
		return false;

	unsigned int completed_sample_count = render_data.render_settings.sample_number + 1u;
	unsigned int minimum_sample_count	= static_cast<unsigned int>(std::max(2, render_data.render_settings.adaptive_sampling_min_samples));
	unsigned int rebuild_interval		= static_cast<unsigned int>(std::max(2, render_data.render_settings.hierarchical_adaptive_sampling_rebuild_interval));

	if (completed_sample_count < minimum_sample_count || completed_sample_count % rebuild_interval != 0u)
		return false;

	upload_render_data(COMPUTE_ERROR_KERNEL, render_data);
	m_kernels[COMPUTE_ERROR_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, nullptr,
														 m_renderer->get_main_stream());

	upload_render_data(BUILD_ROWS_KERNEL, render_data);
	m_kernels[BUILD_ROWS_KERNEL]->launch_asynchronous(64, 1, m_render_resolution.y, 1, nullptr, m_renderer->get_main_stream());

	upload_render_data(BUILD_COLUMNS_KERNEL, render_data);
	m_kernels[BUILD_COLUMNS_KERNEL]->launch_asynchronous(64, 1, m_render_resolution.x, 1, nullptr, m_renderer->get_main_stream());

	upload_render_data(BUILD_HIERARCHY_KERNEL, render_data);
	unsigned int maximum_depth		 = static_cast<unsigned int>(std::max(1, render_data.render_settings.hierarchical_adaptive_sampling_max_depth));
	unsigned int build_command_count = maximum_depth * 2u + 4u;
	unsigned int* build_depths		 = m_build_depths_host_pinned.get_host_pinned_pointer();
	for (unsigned int command_index = 0; command_index < build_command_count; command_index++)
	{
		unsigned int build_depth  = build_depths[command_index];
		unsigned int thread_count = 1u;
		void* launch_args[]		  = { &build_depth };
		if (build_depth <= maximum_depth)
		{
			unsigned int maximum_thread_count = static_cast<unsigned int>(m_nodes.size());
			thread_count					  = maximum_thread_count;
			if (build_depth < 30u)
			{
				unsigned int maximum_nodes_through_depth = (1u << (build_depth + 1u)) - 1u;
				thread_count							 = std::min(maximum_thread_count, maximum_nodes_through_depth);
			}
		}

		m_kernels[BUILD_HIERARCHY_KERNEL]->launch_asynchronous(64, 1, thread_count, 1, launch_args, m_renderer->get_main_stream());
	}

	upload_render_data(RESOLVE_MASK_KERNEL, render_data);
	m_kernels[RESOLVE_MASK_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, nullptr,
														m_renderer->get_main_stream());

	return true;
}

void HierarchicalAdaptiveSamplingRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();
	if (!render_data.render_settings.enable_hierarchical_adaptive_sampling || m_error.size() == 0)
	{
		render_data.aux_buffers.hierarchical_adaptive_sampling_error			= nullptr;
		render_data.aux_buffers.hierarchical_adaptive_sampling_summed_area		= nullptr;
		render_data.aux_buffers.hierarchical_adaptive_sampling_nodes			= nullptr;
		render_data.aux_buffers.hierarchical_adaptive_sampling_node_count		= nullptr;
		render_data.aux_buffers.hierarchical_adaptive_sampling_level_node_count = nullptr;
		render_data.aux_buffers.hierarchical_adaptive_sampling_node_capacity	= 0;
		return;
	}

	render_data.aux_buffers.hierarchical_adaptive_sampling_error			= m_error.get_device_pointer();
	render_data.aux_buffers.hierarchical_adaptive_sampling_summed_area		= m_summed_area.get_device_pointer();
	render_data.aux_buffers.hierarchical_adaptive_sampling_nodes			= m_nodes.get_device_pointer();
	render_data.aux_buffers.hierarchical_adaptive_sampling_node_count		= m_node_count.get_atomic_device_pointer();
	render_data.aux_buffers.hierarchical_adaptive_sampling_level_node_count = m_level_node_count.get_device_pointer();
	render_data.aux_buffers.hierarchical_adaptive_sampling_node_capacity	= static_cast<unsigned int>(m_nodes.size());
}

void HierarchicalAdaptiveSamplingRenderPass::reset(bool reset_by_camera_movement)
{
	if (m_error.size() != 0)
		m_error.memset_whole_buffer(0);
}

bool HierarchicalAdaptiveSamplingRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return m_renderer != nullptr && m_renderer->get_render_data().render_settings.enable_hierarchical_adaptive_sampling;
}
