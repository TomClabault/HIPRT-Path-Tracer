/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/WavefrontRenderPass.h"
#include "Renderer/RenderPasses/WavefrontRoutingConfiguration.h"

#include <algorithm>

const std::string WavefrontRenderPass::WAVEFRONT_RENDER_PASS_NAME						 = "Wavefront Render Pass";
const std::string WavefrontRenderPass::SHADE_PRIMARY_PATHS_KERNEL						 = "Wavefront - Shade Primary Paths";
const std::string WavefrontRenderPass::SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL				 = "Wavefront - Shade Primary Diffuse Paths";
const std::string WavefrontRenderPass::SHADE_PRIMARY_PATHS_GLASS_KERNEL					 = "Wavefront - Shade Primary Glass Paths";
const std::string WavefrontRenderPass::SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL		 = "Wavefront - Shade Primary Single Metallic Paths";
const std::string WavefrontRenderPass::SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL		 = "Wavefront - Shade Primary Specular Diffuse Paths";
const std::string WavefrontRenderPass::SHADE_PATHS_KERNEL								 = "Wavefront - Shade Paths";
const std::string WavefrontRenderPass::SHADE_PATHS_DIFFUSE_KERNEL						 = "Wavefront - Shade Diffuse Paths";
const std::string WavefrontRenderPass::SHADE_PATHS_GLASS_KERNEL							 = "Wavefront - Shade Glass Paths";
const std::string WavefrontRenderPass::SHADE_PATHS_SINGLE_METALLIC_KERNEL				 = "Wavefront - Shade Single Metallic Paths";
const std::string WavefrontRenderPass::SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL				 = "Wavefront - Shade Specular Diffuse Paths";
const std::string WavefrontRenderPass::TRACE_PATHS_KERNEL								 = "Wavefront - Trace Paths";
const std::string WavefrontRenderPass::TRACE_SHADOW_RAYS_KERNEL							 = "Wavefront - Trace Shadow Rays";
const std::string WavefrontRenderPass::NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL				 = "Wavefront - NEE Deferred MIS Context Size";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL = "Wavefront - Classify Secondary Material Families";

const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_KERNEL				   = "Wavefront - Complete Terminated Paths";
const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL		   = "Wavefront - Complete Terminated Diffuse Paths";
const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_GLASS_KERNEL			   = "Wavefront - Complete Terminated Glass Paths";
const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL  = "Wavefront - Complete Terminated Single Metallic Paths";
const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL = "Wavefront - Complete Terminated Specular Diffuse Paths";

WavefrontRenderPass::WavefrontRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(WAVEFRONT_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[SHADE_PRIMARY_PATHS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PRIMARY_PATHS_KERNEL);
	m_kernels[SHADE_PRIMARY_PATHS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePrimaryPaths.h");
	m_kernels[SHADE_PRIMARY_PATHS_KERNEL]->set_kernel_function_name("WavefrontShadePrimaryPaths");

	m_kernels[SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL);
	m_kernels[SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePrimaryPaths.h");
	m_kernels[SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL]->set_kernel_function_name("WavefrontShadePrimaryPaths");

	m_kernels[SHADE_PRIMARY_PATHS_GLASS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PRIMARY_PATHS_GLASS_KERNEL);
	m_kernels[SHADE_PRIMARY_PATHS_GLASS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePrimaryPaths.h");
	m_kernels[SHADE_PRIMARY_PATHS_GLASS_KERNEL]->set_kernel_function_name("WavefrontShadePrimaryPaths");

	m_kernels[SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL);
	m_kernels[SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePrimaryPaths.h");
	m_kernels[SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL]->set_kernel_function_name("WavefrontShadePrimaryPaths");

	m_kernels[SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL);
	m_kernels[SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePrimaryPaths.h");
	m_kernels[SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL]->set_kernel_function_name("WavefrontShadePrimaryPaths");

	m_kernels[SHADE_PATHS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PATHS_KERNEL);
	m_kernels[SHADE_PATHS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePaths.h");
	m_kernels[SHADE_PATHS_KERNEL]->set_kernel_function_name("WavefrontShadePaths");

	m_kernels[SHADE_PATHS_DIFFUSE_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PATHS_DIFFUSE_KERNEL);
	m_kernels[SHADE_PATHS_DIFFUSE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePaths.h");
	m_kernels[SHADE_PATHS_DIFFUSE_KERNEL]->set_kernel_function_name("WavefrontShadePaths");

	m_kernels[SHADE_PATHS_GLASS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PATHS_GLASS_KERNEL);
	m_kernels[SHADE_PATHS_GLASS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePaths.h");
	m_kernels[SHADE_PATHS_GLASS_KERNEL]->set_kernel_function_name("WavefrontShadePaths");

	m_kernels[SHADE_PATHS_SINGLE_METALLIC_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PATHS_SINGLE_METALLIC_KERNEL);
	m_kernels[SHADE_PATHS_SINGLE_METALLIC_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePaths.h");
	m_kernels[SHADE_PATHS_SINGLE_METALLIC_KERNEL]->set_kernel_function_name("WavefrontShadePaths");

	m_kernels[SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL);
	m_kernels[SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ShadePaths.h");
	m_kernels[SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL]->set_kernel_function_name("WavefrontShadePaths");

	m_kernels[TRACE_PATHS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + TRACE_PATHS_KERNEL);
	m_kernels[TRACE_PATHS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/TracePaths.h");
	m_kernels[TRACE_PATHS_KERNEL]->set_kernel_function_name("WavefrontTracePaths");

	m_kernels[TRACE_SHADOW_RAYS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + TRACE_SHADOW_RAYS_KERNEL);
	m_kernels[TRACE_SHADOW_RAYS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/TraceShadowRays.h");
	m_kernels[TRACE_SHADOW_RAYS_KERNEL]->set_kernel_function_name("WavefrontTraceShadowRays");

	m_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL);
	m_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Utils/NEEDeferredMISContextSize.h");
	m_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL]->set_kernel_function_name("NEEDeferredMISContextSize");

	// Primary hits are classified by CameraRays and secondary hits by TracePaths. This separate pass only runs when the family-routed deferred
	// completion still needs the family queues of the shaded vertex after TracePaths
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																					   "/Wavefront/ClassifySecondaryMaterialFamilies.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->set_kernel_function_name("ClassifySecondaryMaterialFamilies");

	m_kernels[COMPLETE_DEFERRED_PATHS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + COMPLETE_DEFERRED_PATHS_KERNEL);
	m_kernels[COMPLETE_DEFERRED_PATHS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/CompleteDeferredPaths.h");
	m_kernels[COMPLETE_DEFERRED_PATHS_KERNEL]->set_kernel_function_name("WavefrontCompleteDeferredPaths");

	m_kernels[COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL);
	m_kernels[COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/CompleteDeferredPaths.h");
	m_kernels[COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL]->set_kernel_function_name("WavefrontCompleteDeferredPaths");

	m_kernels[COMPLETE_DEFERRED_PATHS_GLASS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + COMPLETE_DEFERRED_PATHS_GLASS_KERNEL);
	m_kernels[COMPLETE_DEFERRED_PATHS_GLASS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/CompleteDeferredPaths.h");
	m_kernels[COMPLETE_DEFERRED_PATHS_GLASS_KERNEL]->set_kernel_function_name("WavefrontCompleteDeferredPaths");

	m_kernels[COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL);
	m_kernels[COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/CompleteDeferredPaths.h");
	m_kernels[COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL]->set_kernel_function_name("WavefrontCompleteDeferredPaths");

	m_kernels[COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL);
	m_kernels[COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/CompleteDeferredPaths.h");
	m_kernels[COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL]->set_kernel_function_name("WavefrontCompleteDeferredPaths");

	for (std::pair<const std::string, std::shared_ptr<GPUKernel>>& name_to_kernel : m_kernels)
	{
		name_to_kernel.second->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
		name_to_kernel.second->get_kernel_options().set_macro_value_independently(GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION,
																				  KernelMaterialSpecializationAll);
	}
	m_kernels[SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationDiffuse);
	m_kernels[SHADE_PRIMARY_PATHS_GLASS_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationGlass);
	m_kernels[SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationSingleMetallic);
	m_kernels[SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationSpecularDiffuse);
	m_kernels[SHADE_PATHS_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value_independently(GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION,
																							  KernelMaterialSpecializationDiffuse);
	m_kernels[SHADE_PATHS_GLASS_KERNEL]->get_kernel_options().set_macro_value_independently(GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION,
																							KernelMaterialSpecializationGlass);
	m_kernels[SHADE_PATHS_SINGLE_METALLIC_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationSingleMetallic);
	m_kernels[SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationSpecularDiffuse);

	m_kernels[COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationDiffuse);
	m_kernels[COMPLETE_DEFERRED_PATHS_GLASS_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationGlass);
	m_kernels[COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationSingleMetallic);
	m_kernels[COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value_independently(
		GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, KernelMaterialSpecializationSpecularDiffuse);

	m_kernels[SHADE_PRIMARY_PATHS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[SHADE_PRIMARY_PATHS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
	m_kernels[SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
																						KERNEL_OPTION_TRUE);
	m_kernels[SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
	m_kernels[SHADE_PRIMARY_PATHS_GLASS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
																					  KERNEL_OPTION_TRUE);
	m_kernels[SHADE_PRIMARY_PATHS_GLASS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
	m_kernels[SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
																								KERNEL_OPTION_TRUE);
	m_kernels[SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
	m_kernels[SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
																								 KERNEL_OPTION_TRUE);
	m_kernels[SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
	m_kernels[SHADE_PATHS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[SHADE_PATHS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
	for (unsigned int family_index = 1; family_index < KernelMaterialSpecializationCount; family_index++)
	{
		const std::string& kernel_name = get_secondary_shading_kernel_name(family_index);
		m_kernels[kernel_name]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
		m_kernels[kernel_name]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
	}
	m_kernels[TRACE_PATHS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[TRACE_PATHS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
	m_kernels[TRACE_SHADOW_RAYS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[TRACE_SHADOW_RAYS_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);

	// The forked families are the small ones, the dominant family runs on the main stream. Giving the family streams
	// the greatest priority lets their workgroups be dispatched as soon as the dominant family kernel frees some slots
	int least_stream_priority	 = 0;
	int greatest_stream_priority = 0;
	OROCHI_CHECK_ERROR(oroDeviceGetStreamPriorityRange(&least_stream_priority, &greatest_stream_priority));

	// The events only order the streams, they are never used for timing
	OROCHI_CHECK_ERROR(oroEventCreateWithFlags(&m_material_family_fork_event, oroEventDisableTiming));
	for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
	{
		OROCHI_CHECK_ERROR(oroStreamCreateWithPriority(&m_material_family_streams[family_index], 0, greatest_stream_priority));
		OROCHI_CHECK_ERROR(oroEventCreateWithFlags(&m_material_family_join_events[family_index], oroEventDisableTiming));
	}
}

void WavefrontRenderPass::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
									const std::vector<hiprtFuncNameSet>& func_name_sets,
									bool silent,
									bool use_cache)
{
	// Kernel options can change the NEE deferred MIS context layout. Query its newly compiled size
	// before deciding whether the wavefront staging buffers need to be resized.
	m_nee_deferred_mis_context_byte_size_dirty = true;

	RenderPass::recompile(hiprt_orochi_ctx, func_name_sets, silent, use_cache);
}

bool WavefrontRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
													   const std::vector<hiprtFuncNameSet>& func_name_sets,
													   bool silent,
													   bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))

		return false;

	bool updated													 = false;
	std::map<std::string, std::shared_ptr<GPUKernel>> active_kernels = get_all_kernels();
	for (std::pair<const std::string, std::shared_ptr<GPUKernel>>& name_to_kernel : active_kernels)
	{
		if (!name_to_kernel.second->has_been_compiled())
		{
			updated = true;
			name_to_kernel.second->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

			if (name_to_kernel.first == NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL)
				m_nee_deferred_mis_context_byte_size_dirty = true;
		}
	}

	return updated;
}

void WavefrontRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool WavefrontRenderPass::pre_frame_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
	{
		bool had_buffers = m_staging_buffers_allocated;
		free_staging_buffers();

		return had_buffers;
	}

	bool resized = resize_staging_buffers();

	// Resetting this flag as this is a new frame
	render_data.render_settings.do_update_status_buffers = false;

	if (!render_data.render_settings.accumulate)
		render_data.render_settings.sample_number = 0;

	return resized;
}

bool WavefrontRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))

		return false;
	if (!kernels_ready() || !m_staging_buffers_allocated)
	{
		if (m_staging_buffers_allocated)
			// The camera rays pass may already have appended this frame's primary hits to the material family queues
			reset_material_family_counts_async(m_renderer->get_main_stream());

		return false;
	}

	render_data.wavefront_data				= get_wavefront_data_device();
	bool trace_classifies_material_families = render_data.wavefront_data.trace_classifies_material_families != 0;

	unsigned int path_capacity = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	unsigned int bounce_count  = static_cast<unsigned int>(render_data.render_settings.nb_bounces);
	if (render_data.render_settings.do_render_low_resolution())
		bounce_count = std::min(3u, bounce_count);

	oroStream_t main_stream = m_renderer->get_main_stream();

	unsigned int queue_block_size	   = KernelBlockWidthHeight * KernelBlockWidthHeight;
	unsigned int wavefront_block_size  = 64;
	unsigned int wavefront_block_count = std::max(1u, std::min((path_capacity + 63u) / 64u, 2048u));
	unsigned int shadow_block_count	   = (path_capacity + 63u) / 64u;

	int nee_estimator = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	WavefrontRoutingConfiguration routing_configuration =
		get_wavefront_routing_configuration(compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL), nee_estimator);
	unsigned int material_family_count	   = routing_configuration.material_family_count;
	bool has_deferred_bsdf_sampling		   = routing_configuration.requires_terminal_trace_completion;
	bool route_deferred_by_material_family = routing_configuration.route_deferred_completion_by_material_family;
	bool defer_nee_shadow_traversal = compiler_options.get_macro_value(GPUKernelCompilerOptions::WAVEFRONT_DEFER_NEE_SHADOW_TRAVERSAL) == KERNEL_OPTION_TRUE;
	bool run_shadow_pass			= defer_nee_shadow_traversal && m_wavefront_data.shadow_work_capacity() > 0;

	// HIPRT assigns the global BVH traversal stacks statically from the block index: two family kernels running
	// concurrently would share the stacks of their blocks with the same index. Each family is given its own slice
	// of the global stack buffer and the grid of the family kernels is limited to the size of that slice
	hiprtGlobalStackBuffer global_stack_buffer = render_data.global_traversal_stack_buffer;
	unsigned int family_stack_count			   = global_stack_buffer.stackCount / material_family_count;
	unsigned int family_block_count			   = std::min(wavefront_block_count, family_stack_count / wavefront_block_size);
	// The families fall back to running one after the other on the main stream with the whole stack buffer if
	// it is too small to be split between them. A single family has nothing to overlap with
	bool overlap_family_kernels = material_family_count > 1 && global_stack_buffer.stackData != nullptr && family_block_count > 0;
	if (!overlap_family_kernels)
		family_block_count = wavefront_block_count;

	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	host_pinned_render_data[0]				 = render_data;
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		HIPRTRenderData& family_render_data = host_pinned_render_data[1 + family_index];
		family_render_data					= render_data;
		if (!overlap_family_kernels)
			continue;

		unsigned int family_slice_stack_count = family_block_count * wavefront_block_size;
		std::size_t family_slice_entry_offset =
			static_cast<std::size_t>(family_index) * family_slice_stack_count * static_cast<std::size_t>(global_stack_buffer.stackSize);

		// The global stack buffer is created with integer entries
		family_render_data.global_traversal_stack_buffer.stackData	= static_cast<unsigned int*>(global_stack_buffer.stackData) + family_slice_entry_offset;
		family_render_data.global_traversal_stack_buffer.stackCount = family_slice_stack_count;
	}

	m_kernels[TRACE_PATHS_KERNEL]->upload_to_module_global("WAVEFRONT_TRACE_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
	m_kernels[TRACE_SHADOW_RAYS_KERNEL]->upload_to_module_global("WAVEFRONT_TRACE_SHADOW_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData),
																 main_stream);
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		HIPRTRenderData* family_render_data = host_pinned_render_data + 1 + family_index;

		m_kernels[get_primary_shading_kernel_name(family_index)]->upload_to_module_global("WAVEFRONT_SHADE_PRIMARY_RENDER_DATA", family_render_data,
																						  sizeof(HIPRTRenderData), main_stream);
		m_kernels[get_secondary_shading_kernel_name(family_index)]->upload_to_module_global("WAVEFRONT_SHADE_RENDER_DATA", family_render_data,
																							sizeof(HIPRTRenderData), main_stream);
	}

	if (!trace_classifies_material_families)
		m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->upload_to_module_global("WAVEFRONT_CLASSIFY_SECONDARY_MATERIAL_FAMILIES_RENDER_DATA",
																							  host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
	if (route_deferred_by_material_family)
	{
		for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
			m_kernels[get_deferred_completion_kernel_name(family_index)]->upload_to_module_global(
				"WAVEFRONT_COMPLETE_DEFERRED_RENDER_DATA", host_pinned_render_data + 1 + family_index, sizeof(HIPRTRenderData), main_stream);
	}
	else
		m_kernels[COMPLETE_DEFERRED_PATHS_KERNEL]->upload_to_module_global("WAVEFRONT_COMPLETE_DEFERRED_RENDER_DATA", host_pinned_render_data,
																		   sizeof(HIPRTRenderData), main_stream);

	unsigned int* zero = m_zero_host_pinned.get_host_pinned_pointer();
	zero[0]			   = 0;

	// The dominant material family only matters when the families are overlapped: it is the one that runs on the main stream
	bool read_back_material_family_counts		   = overlap_family_kernels && m_material_family_count_readback_frame_counter == 0;
	m_material_family_count_readback_frame_counter = (m_material_family_count_readback_frame_counter + 1) % MATERIAL_FAMILY_COUNT_READBACK_PERIOD;

	for (unsigned int bounce_index = 0; bounce_index <= bounce_count; bounce_index++)
	{
		m_wavefront_data.get_queue_count_buffer(1).upload_data_async(zero, main_stream);
		if (run_shadow_pass)
		{
			m_wavefront_data.get_shadow_work_count_buffer().upload_data_async(zero, main_stream);
			m_wavefront_data.get_shadow_queue_overflow_count_buffer().upload_data_async(zero, main_stream);
		}

		void* shade_launch_args[] = { &bounce_count };
		if (bounce_index == 0 && path_capacity > 0)
		{
			// The camera rays pass has already appended the primary hits to the material family queues
			launch_material_family_kernels_async(MaterialFamilyKernelType::PRIMARY_SHADING, shade_launch_args, material_family_count, family_block_count,
												 overlap_family_kernels);
			if (read_back_material_family_counts)
				read_back_material_family_counts_async(MaterialFamilyKernelType::PRIMARY_SHADING);
		}
		else if (bounce_index > 0 && path_capacity > 0)
		{
			if (!trace_classifies_material_families)
			{
				// The family-routed deferred completion of the previous bounce has consumed the family queues, they can be rebuilt from queue 0
				reset_material_family_counts_async(main_stream);
				m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, nullptr,
																												 main_stream);
			}

			launch_material_family_kernels_async(MaterialFamilyKernelType::SECONDARY_SHADING, shade_launch_args, material_family_count, family_block_count,
												 overlap_family_kernels);
			// The first secondary bounce is representative of the following ones
			if (read_back_material_family_counts && bounce_index == 1)
				read_back_material_family_counts_async(MaterialFamilyKernelType::SECONDARY_SHADING);
		}

		if (run_shadow_pass)
			m_kernels[TRACE_SHADOW_RAYS_KERNEL]->launch_asynchronous_3D_block_count(shadow_block_count, 1, 1, 64, 1, 1, nullptr, main_stream);

		if (bounce_index >= bounce_count && !has_deferred_bsdf_sampling)
			continue;

		if (trace_classifies_material_families)
			// TracePaths appends the new hits to the material family queues, which the shading passes of this bounce have already consumed.
			// Queue 0 is not used in that case
			reset_material_family_counts_async(main_stream);
		else
			m_wavefront_data.get_queue_count_buffer(0).upload_data_async(zero, main_stream);
		m_wavefront_data.get_queue_count_buffer(WAVEFRONT_COMPLETION_QUEUE_INDEX).upload_data_async(zero, main_stream);
		m_kernels[TRACE_PATHS_KERNEL]->launch_asynchronous(queue_block_size, 1, path_capacity, 1, nullptr, main_stream);

		bool deferred_completion_can_emit_shadow = run_shadow_pass && nee_estimator == LSS_RIS_BSDF_AND_LIGHT;
		if (deferred_completion_can_emit_shadow)
			m_wavefront_data.get_shadow_work_count_buffer().upload_data_async(zero, main_stream);

		// Family-routed deferred completion reuses these queues and filters paths by terminal/miss state.
		bool complete_by_family		 = route_deferred_by_material_family;
		void* complete_launch_args[] = { &complete_by_family };
		if (route_deferred_by_material_family)
			launch_material_family_kernels_async(MaterialFamilyKernelType::DEFERRED_COMPLETION, complete_launch_args, material_family_count, family_block_count,
												 overlap_family_kernels);
		else
			m_kernels[COMPLETE_DEFERRED_PATHS_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, complete_launch_args,
																						  main_stream);

		if (deferred_completion_can_emit_shadow)
			m_kernels[TRACE_SHADOW_RAYS_KERNEL]->launch_asynchronous_3D_block_count(shadow_block_count, 1, 1, 64, 1, 1, nullptr, main_stream);
	}

	// The camera rays pass of the next frame appends the primary hits to the material family queues
	reset_material_family_counts_async(main_stream);

	return true;
}

void WavefrontRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (is_render_pass_used(*m_compiler_options) && m_staging_buffers_allocated)
		render_data.wavefront_data = get_wavefront_data_device();
	else
		render_data.wavefront_data = WavefrontDataDevice();
}

WavefrontDataDevice WavefrontRenderPass::get_wavefront_data_device()
{
	WavefrontDataDevice wavefront_data_device = m_wavefront_data.to_device();

	// The family-routed deferred completion runs after TracePaths and still needs the family queues of the shaded vertex.
	// TracePaths cannot overwrite them with the families of the new hits in that case so a separate classification pass runs instead
	if (wavefront_data_device.material_family_routing_enabled && !uses_deferred_material_family_routing(*m_compiler_options))
		wavefront_data_device.trace_classifies_material_families = 1;

	return wavefront_data_device;
}

void WavefrontRenderPass::reset_material_family_counts_async(oroStream_t stream)
{
	if (!m_staging_buffers_allocated)
		return;

	m_wavefront_data.get_material_family_count_buffer().upload_data_async(m_zero_host_pinned.get_host_pinned_pointer(), stream);
}

const std::string& WavefrontRenderPass::get_material_family_kernel_name(MaterialFamilyKernelType kernel_type, unsigned int family_index) const
{
	switch (kernel_type)
	{
	case MaterialFamilyKernelType::PRIMARY_SHADING:

		return get_primary_shading_kernel_name(family_index);
	case MaterialFamilyKernelType::SECONDARY_SHADING:

		return get_secondary_shading_kernel_name(family_index);
	case MaterialFamilyKernelType::DEFERRED_COMPLETION:
	default:

		return get_deferred_completion_kernel_name(family_index);
	}
}

unsigned int WavefrontRenderPass::get_material_family_count_readback_slot(MaterialFamilyKernelType kernel_type) const
{
	// The deferred completion consumes the family queues of the secondary hits
	return kernel_type == MaterialFamilyKernelType::PRIMARY_SHADING ? 0 : 1;
}

unsigned int WavefrontRenderPass::get_dominant_material_family(MaterialFamilyKernelType kernel_type, unsigned int material_family_count) const
{
	unsigned int readback_slot	 = get_material_family_count_readback_slot(kernel_type);
	const unsigned int* counts	 = m_material_family_counts_host_pinned.get_host_pinned_pointer() + readback_slot * KernelMaterialSpecializationCount;
	unsigned int dominant_family = 0;

	// Before the first read back has completed, all the counts are 0 and the first family is picked
	for (unsigned int family_index = 1; family_index < material_family_count; family_index++)
		if (counts[family_index] > counts[dominant_family])
			dominant_family = family_index;

	return dominant_family;
}

void WavefrontRenderPass::read_back_material_family_counts_async(MaterialFamilyKernelType kernel_type)
{
	unsigned int readback_slot = get_material_family_count_readback_slot(kernel_type);
	unsigned int* destination  = m_material_family_counts_host_pinned.get_host_pinned_pointer() + readback_slot * KernelMaterialSpecializationCount;

	// Enqueued on the main stream after the join of the family kernels: the counters still hold the counts that the
	// family kernels have just consumed, they are only reset before the next pass that appends to the family queues
	m_wavefront_data.get_material_family_count_buffer().download_data_async(destination, m_renderer->get_main_stream());
}

void WavefrontRenderPass::launch_material_family_kernels_async(
	MaterialFamilyKernelType kernel_type, void** launch_args, unsigned int material_family_count, unsigned int block_count, bool overlap)
{
	oroStream_t main_stream = m_renderer->get_main_stream();

	if (!overlap)
	{
		for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
			m_kernels[get_material_family_kernel_name(kernel_type, family_index)]->launch_asynchronous_3D_block_count(block_count, 1, 1, 64, 1, 1, launch_args,
																													  main_stream);

		return;
	}

	unsigned int dominant_family_index = get_dominant_material_family(kernel_type, material_family_count);

	// The family kernels must not start before the work already queued on the main stream: render data uploads,
	// queue counter resets and the kernels that produced the family queues
	OROCHI_CHECK_ERROR(oroEventRecord(m_material_family_fork_event, main_stream));

	// The dominant family is launched on the main stream: it has no cross-stream wait so the worker thread of the main stream
	// submits it ahead of time and it starts right after the kernel that produced its queue. The worker threads of the
	// other streams wait on the CPU for the fork event before they can submit their kernels (see the comment on the join below).
	// It still uses its own render data and its slice of the global stack buffer since the other families run concurrently
	m_kernels[get_material_family_kernel_name(kernel_type, dominant_family_index)]->launch_asynchronous_3D_block_count(block_count, 1, 1, 64, 1, 1, launch_args,
																													   main_stream);

	// The families shade disjoint sets of paths and only append to the shared queues with atomics so they can run concurrently
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		if (family_index == dominant_family_index)
			continue;

		oroStream_t family_stream = m_material_family_streams[family_index];

		OROCHI_CHECK_ERROR(oroStreamWaitEvent(family_stream, m_material_family_fork_event, 0));
		m_kernels[get_material_family_kernel_name(kernel_type, family_index)]->launch_asynchronous_3D_block_count(block_count, 1, 1, 64, 1, 1, launch_args,
																												  family_stream);
		OROCHI_CHECK_ERROR(oroEventRecord(m_material_family_join_events[family_index], family_stream));
	}

	// The following passes of the main stream consume the outputs of all the families.
	// The dominant family is already ordered on the main stream so it is not joined.
	//
	// The HIP runtime has no direct dispatch on Windows (PAL backend): each stream has a worker thread that handles cross-stream
	// waits on the CPU, the app thread does not block here. When the worker of the main stream reaches the wait of a family that
	// has not finished, HostQueue::loop() first waits for all the work already submitted on the main stream, which includes the
	// dominant family kernel, and then for the family. The next passes are only submitted after that so the GPU idles
	// at each join (see memory/performance/OPTIMIZATIONS_SUCCESS.md)
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
		if (family_index != dominant_family_index)
			OROCHI_CHECK_ERROR(oroStreamWaitEvent(main_stream, m_material_family_join_events[family_index], 0));
}

void WavefrontRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))

		return;

	if (render_data.render_settings.accumulate)
		if (m_renderer->get_application_settings()->auto_sample_per_frame)
			render_data.render_settings.samples_per_frame = 1;

	render_data.render_settings.denoiser_AOV_accumulation_counter = 0;
	render_data.render_settings.sample_number					  = 0;
}

bool WavefrontRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	int nee_estimator	  = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	int sampling_strategy = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY);
	bool nisml_enabled	  = ILLUMINATION_AWARE_KD_TREE_IS_NISML(nee_estimator, sampling_strategy);

	return wavefront_path_strategy_uses_pass(compiler_options.get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY), nisml_enabled);
}

unsigned int WavefrontRenderPass::get_material_family_count(const GPUKernelCompilerOptions& compiler_options) const
{
	WavefrontRoutingConfiguration configuration =
		get_wavefront_routing_configuration(compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL),
											compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR));

	return configuration.material_family_count;
}

bool WavefrontRenderPass::uses_deferred_material_family_routing(const GPUKernelCompilerOptions& compiler_options) const
{
	WavefrontRoutingConfiguration configuration =
		get_wavefront_routing_configuration(compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL),
											compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR));

	return configuration.route_deferred_completion_by_material_family;
}

const std::string& WavefrontRenderPass::get_primary_shading_kernel_name(unsigned int family_index) const
{
	switch (family_index)
	{
	case KernelMaterialSpecializationDiffuse:

		return SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL;
	case KernelMaterialSpecializationGlass:

		return SHADE_PRIMARY_PATHS_GLASS_KERNEL;
	case KernelMaterialSpecializationSingleMetallic:

		return SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL;
	case KernelMaterialSpecializationSpecularDiffuse:

		return SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL;
	default:

		return SHADE_PRIMARY_PATHS_KERNEL;
	}
}

const std::string& WavefrontRenderPass::get_secondary_shading_kernel_name(unsigned int family_index) const
{
	switch (family_index)
	{
	case KernelMaterialSpecializationDiffuse:

		return SHADE_PATHS_DIFFUSE_KERNEL;
	case KernelMaterialSpecializationGlass:

		return SHADE_PATHS_GLASS_KERNEL;
	case KernelMaterialSpecializationSingleMetallic:

		return SHADE_PATHS_SINGLE_METALLIC_KERNEL;
	case KernelMaterialSpecializationSpecularDiffuse:

		return SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL;
	default:

		return SHADE_PATHS_KERNEL;
	}
}

const std::string& WavefrontRenderPass::get_deferred_completion_kernel_name(unsigned int family_index) const
{
	switch (family_index)
	{
	case KernelMaterialSpecializationDiffuse:

		return COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL;
	case KernelMaterialSpecializationGlass:

		return COMPLETE_DEFERRED_PATHS_GLASS_KERNEL;
	case KernelMaterialSpecializationSingleMetallic:

		return COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL;
	case KernelMaterialSpecializationSpecularDiffuse:

		return COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL;
	default:

		return COMPLETE_DEFERRED_PATHS_KERNEL;
	}
}

std::map<std::string, std::shared_ptr<GPUKernel>> WavefrontRenderPass::get_all_kernels()
{
	if (!is_render_pass_used(*m_compiler_options))

		return {};

	unsigned int material_family_count = get_material_family_count(*m_compiler_options);

	std::map<std::string, std::shared_ptr<GPUKernel>> active_kernels;
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		active_kernels[get_primary_shading_kernel_name(family_index)]	= m_kernels.at(get_primary_shading_kernel_name(family_index));
		active_kernels[get_secondary_shading_kernel_name(family_index)] = m_kernels.at(get_secondary_shading_kernel_name(family_index));
	}

	// Without family-routed deferred completion, TracePaths classifies the secondary hits itself
	if (uses_deferred_material_family_routing(*m_compiler_options))
		active_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL] = m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL);

	active_kernels[COMPLETE_DEFERRED_PATHS_KERNEL] = m_kernels.at(COMPLETE_DEFERRED_PATHS_KERNEL);
	if (uses_deferred_material_family_routing(*m_compiler_options))
	{
		for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
			active_kernels[get_deferred_completion_kernel_name(family_index)] = m_kernels.at(get_deferred_completion_kernel_name(family_index));
	}

	active_kernels[TRACE_PATHS_KERNEL]					 = m_kernels.at(TRACE_PATHS_KERNEL);
	active_kernels[TRACE_SHADOW_RAYS_KERNEL]			 = m_kernels.at(TRACE_SHADOW_RAYS_KERNEL);
	active_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL] = m_kernels.at(NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL);

	return active_kernels;
}

std::map<std::string, std::shared_ptr<GPUKernel>> WavefrontRenderPass::get_tracing_kernels()
{
	if (!is_render_pass_used(*m_compiler_options))

		return {};

	unsigned int material_family_count = get_material_family_count(*m_compiler_options);

	std::map<std::string, std::shared_ptr<GPUKernel>> tracing_kernels;
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		tracing_kernels[get_primary_shading_kernel_name(family_index)]	 = m_kernels.at(get_primary_shading_kernel_name(family_index));
		tracing_kernels[get_secondary_shading_kernel_name(family_index)] = m_kernels.at(get_secondary_shading_kernel_name(family_index));
	}

	tracing_kernels[TRACE_PATHS_KERNEL]		  = m_kernels.at(TRACE_PATHS_KERNEL);
	tracing_kernels[TRACE_SHADOW_RAYS_KERNEL] = m_kernels.at(TRACE_SHADOW_RAYS_KERNEL);

	return tracing_kernels;
}

std::size_t WavefrontRenderPass::get_nee_deferred_mis_context_byte_size()
{
	if (!m_nee_deferred_mis_context_byte_size_dirty)

		return m_nee_deferred_mis_context_byte_size;

	std::shared_ptr<GPUKernel> context_size_kernel = m_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL];
	if (!context_size_kernel->has_been_compiled())

		return 0;

	OrochiBuffer<std::size_t> out_size_buffer(1);
	std::size_t* out_size_buffer_pointer = out_size_buffer.get_device_pointer();
	void* launch_args[]					 = { &out_size_buffer_pointer };
	context_size_kernel->launch_synchronous(1, 1, 1, 1, launch_args, 0);

	m_nee_deferred_mis_context_byte_size	   = out_size_buffer.download_data()[0];
	m_nee_deferred_mis_context_byte_size_dirty = false;

	return m_nee_deferred_mis_context_byte_size;
}

unsigned int WavefrontRenderPass::get_shadow_work_capacity(unsigned int path_capacity) const
{
	if (m_compiler_options->get_macro_value(GPUKernelCompilerOptions::WAVEFRONT_DEFER_NEE_SHADOW_TRAVERSAL) != KERNEL_OPTION_TRUE)
		return 0;

	int nee_estimator					   = m_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	int sampling_strategy				   = m_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY);
	unsigned int direct_light_sample_count = 1;

	if (sampling_strategy == LSS_BASE_LIGHT_TREE_ATS &&
		m_compiler_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_DO_SPLITTING) == KERNEL_OPTION_TRUE)
		direct_light_sample_count =
			static_cast<unsigned int>(m_compiler_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_SPLITTING_MAX_LIGHT_SAMPLES));
	else if (sampling_strategy == LSS_BASE_LIGHT_TREE_SG &&
			 m_compiler_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_DO_SPLITTING) == KERNEL_OPTION_TRUE)
		direct_light_sample_count =
			static_cast<unsigned int>(m_compiler_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_SPLITTING_MAX_LIGHT_SAMPLES));

	unsigned int emissive_shadow_count = 0;
	if (nee_estimator == LSS_ONE_LIGHT || nee_estimator == LSS_MIS_LIGHT_BSDF)
		emissive_shadow_count = direct_light_sample_count;
	else if (nee_estimator == LSS_RIS_BSDF_AND_LIGHT || nee_estimator == LSS_RISLTC || nee_estimator == LSS_NEURAL_MANY_LIGHTS ||
			 DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(nee_estimator))
		emissive_shadow_count = 1;

	unsigned int envmap_shadow_count = 0;
	if (nee_estimator != LSS_NO_DIRECT_LIGHT_SAMPLING &&
		m_compiler_options->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) != ESS_NO_SAMPLING)
	{
		envmap_shadow_count = 1;
		if (m_compiler_options->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS) == KERNEL_OPTION_TRUE)
			envmap_shadow_count++;
	}

	return path_capacity * (emissive_shadow_count + envmap_shadow_count);
}

bool WavefrontRenderPass::kernels_ready() const
{
	if (!m_kernels.at(TRACE_PATHS_KERNEL)->has_been_compiled() || !m_kernels.at(COMPLETE_DEFERRED_PATHS_KERNEL)->has_been_compiled() ||
		!m_kernels.at(TRACE_SHADOW_RAYS_KERNEL)->has_been_compiled() || !m_kernels.at(NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL)->has_been_compiled() ||
		m_nee_deferred_mis_context_byte_size == 0)

		return false;

	unsigned int material_family_count = get_material_family_count(*m_compiler_options);
	if (uses_deferred_material_family_routing(*m_compiler_options))
	{
		for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
			if (!m_kernels.at(get_deferred_completion_kernel_name(family_index))->has_been_compiled())

				return false;
	}

	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
		if (!m_kernels.at(get_primary_shading_kernel_name(family_index))->has_been_compiled() ||
			!m_kernels.at(get_secondary_shading_kernel_name(family_index))->has_been_compiled())

			return false;

	if (uses_deferred_material_family_routing(*m_compiler_options))
		return m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL)->has_been_compiled();

	return true;
}

bool WavefrontRenderPass::resize_staging_buffers()
{
	unsigned int path_capacity					  = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	unsigned int shadow_work_capacity			  = get_shadow_work_capacity(path_capacity);
	std::size_t ray_volume_state_byte_size		  = m_renderer->get_render_data().ray_volume_state_byte_size;
	bool allocate_resolved_material_control_cache = m_compiler_options->get_macro_value(GPUKernelCompilerOptions::USE_MATERIAL_TEXTURES) == KERNEL_OPTION_TRUE;
	// Only the principled BSDF classifies the secondary hits into material families: the classification is cached for the shading pass
	bool allocate_current_material_classifications =
		allocate_resolved_material_control_cache && m_compiler_options->get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL) == BSDF_PRINCIPLED;
	int nee_estimator								  = m_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	bool allocate_learning_to_cluster_shadow_sidecars = shadow_work_capacity > 0 && DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(nee_estimator);
	bool allocate_nisml_shadow_sidecars				  = shadow_work_capacity > 0 && nee_estimator == LSS_NEURAL_MANY_LIGHTS;
	if (ray_volume_state_byte_size == 0)
		Debug::debugbreak();
	std::size_t nee_deferred_mis_context_byte_size = get_nee_deferred_mis_context_byte_size();
	if (nee_deferred_mis_context_byte_size == 0)
	{
		bool had_buffers = m_staging_buffers_allocated;
		free_staging_buffers();

		return had_buffers;
	}

	bool needs_resize = !m_staging_buffers_allocated || m_wavefront_data.path_capacity() != path_capacity ||
						m_wavefront_data.has_resolved_material_control_cache() != allocate_resolved_material_control_cache ||
						m_wavefront_data.has_current_material_classifications() != allocate_current_material_classifications ||
						m_wavefront_data.shadow_work_capacity() != shadow_work_capacity ||
						m_wavefront_data.has_learning_to_cluster_shadow_sidecars() != allocate_learning_to_cluster_shadow_sidecars ||
						m_wavefront_data.has_nisml_shadow_sidecars() != allocate_nisml_shadow_sidecars ||
						m_allocated_ray_volume_state_byte_size != ray_volume_state_byte_size ||
						m_allocated_nee_deferred_mis_context_byte_size != nee_deferred_mis_context_byte_size ||
						m_render_data_host_pinned.size() != 1 + KernelMaterialSpecializationCount ||
						m_zero_host_pinned.size() != KernelMaterialSpecializationCount ||
						m_material_family_counts_host_pinned.size() != 2 * KernelMaterialSpecializationCount;
	if (!needs_resize)

		return false;

	free_staging_buffers();

	m_wavefront_data.resize(path_capacity, shadow_work_capacity, ray_volume_state_byte_size, nee_deferred_mis_context_byte_size,
							allocate_resolved_material_control_cache, allocate_current_material_classifications, allocate_learning_to_cluster_shadow_sidecars,
							allocate_nisml_shadow_sidecars);
	// One render data for the main stream kernels and one per material family
	m_render_data_host_pinned.resize_host_pinned_mem(1 + KernelMaterialSpecializationCount);
	// Large enough to reset all the material family counters with a single upload
	m_zero_host_pinned.resize_host_pinned_mem(KernelMaterialSpecializationCount);
	std::fill_n(m_zero_host_pinned.get_host_pinned_pointer(), KernelMaterialSpecializationCount, 0u);
	// The primary and secondary shading counts read back to find the dominant material family of each pass
	m_material_family_counts_host_pinned.resize_host_pinned_mem(2 * KernelMaterialSpecializationCount);
	std::fill_n(m_material_family_counts_host_pinned.get_host_pinned_pointer(), 2 * KernelMaterialSpecializationCount, 0u);
	m_material_family_count_readback_frame_counter = 0;

	m_allocated_ray_volume_state_byte_size		   = ray_volume_state_byte_size;
	m_allocated_nee_deferred_mis_context_byte_size = nee_deferred_mis_context_byte_size;
	m_staging_buffers_allocated					   = true;

	return true;
}

void WavefrontRenderPass::free_staging_buffers()
{
	m_wavefront_data.free();
	m_render_data_host_pinned.free_no_error();
	m_zero_host_pinned.free_no_error();
	m_material_family_counts_host_pinned.free_no_error();

	m_allocated_ray_volume_state_byte_size		   = 0;
	m_allocated_nee_deferred_mis_context_byte_size = 0;
	m_staging_buffers_allocated					   = false;
}
