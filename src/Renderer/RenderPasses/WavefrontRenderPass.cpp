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
}

void WavefrontRenderPass::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
									const std::vector<hiprtFuncNameSet>& func_name_sets,
									bool silent,
									bool use_cache)
{
	// Kernel options can change the NEE deferred MIS context layout. Query its newly compiled size
	// before deciding whether the wavefront staging buffers need to be resized.
	m_wavefront_common.set_nee_deferred_mis_context_byte_size_dirty();

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
				m_wavefront_common.set_nee_deferred_mis_context_byte_size_dirty();
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
		bool had_buffers = m_wavefront_common.are_staging_buffers_allocated();
		m_wavefront_common.free_staging_buffers();

		return had_buffers;
	}

	unsigned int path_capacity = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	bool resized			   = m_wavefront_common.resize_staging_buffers(path_capacity, render_data.ray_volume_state_byte_size, *m_compiler_options,
																		   m_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL]);

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
	if (!kernels_ready() || !m_wavefront_common.are_staging_buffers_allocated())
	{
		if (m_wavefront_common.are_staging_buffers_allocated())
			// The camera rays pass may already have appended this frame's primary hits to the material family queues
			m_wavefront_common.reset_material_family_counts_async(m_renderer->get_main_stream());

		return false;
	}

	render_data.wavefront_data				= m_wavefront_common.get_wavefront_data_device(*m_compiler_options);
	bool trace_classifies_material_families = render_data.wavefront_data.trace_classifies_material_families != 0;

	unsigned int path_capacity = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	unsigned int bounce_count  = static_cast<unsigned int>(render_data.render_settings.nb_bounces);
	if (render_data.render_settings.do_render_low_resolution())
		bounce_count = std::min(3u, bounce_count);

	oroStream_t main_stream = m_renderer->get_main_stream();

	unsigned int queue_block_size	   = KernelBlockWidthHeight * KernelBlockWidthHeight;
	unsigned int wavefront_block_count = WavefrontRenderPassCommon::get_wavefront_block_count(path_capacity);
	unsigned int shadow_block_count	   = WavefrontRenderPassCommon::get_shadow_block_count(path_capacity);

	int nee_estimator = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	WavefrontRoutingConfiguration routing_configuration =
		get_wavefront_routing_configuration(compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL), nee_estimator);
	unsigned int material_family_count	   = routing_configuration.material_family_count;
	bool has_deferred_bsdf_sampling		   = routing_configuration.requires_terminal_trace_completion;
	bool route_deferred_by_material_family = routing_configuration.route_deferred_completion_by_material_family;
	bool defer_nee_shadow_traversal = compiler_options.get_macro_value(GPUKernelCompilerOptions::WAVEFRONT_DEFER_NEE_SHADOW_TRAVERSAL) == KERNEL_OPTION_TRUE;
	bool run_shadow_pass			= defer_nee_shadow_traversal && m_wavefront_common.get_wavefront_data().shadow_work_capacity() > 0;

	unsigned int family_block_count;
	bool overlap_family_kernels =
		m_wavefront_common.prepare_material_family_render_data(render_data, material_family_count, wavefront_block_count, family_block_count);

	m_wavefront_common.upload_main_stream_render_data(*m_kernels[TRACE_PATHS_KERNEL], "WAVEFRONT_TRACE_RENDER_DATA", main_stream);
	m_wavefront_common.upload_main_stream_render_data(*m_kernels[TRACE_SHADOW_RAYS_KERNEL], "WAVEFRONT_TRACE_SHADOW_RENDER_DATA", main_stream);
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		m_wavefront_common.upload_material_family_render_data(*m_kernels[get_primary_shading_kernel_name(family_index)], "WAVEFRONT_SHADE_PRIMARY_RENDER_DATA",
															  family_index, main_stream);
		m_wavefront_common.upload_material_family_render_data(*m_kernels[get_secondary_shading_kernel_name(family_index)], "WAVEFRONT_SHADE_RENDER_DATA",
															  family_index, main_stream);
	}

	if (!trace_classifies_material_families)
		m_wavefront_common.upload_main_stream_render_data(*m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL],
														  "WAVEFRONT_CLASSIFY_SECONDARY_MATERIAL_FAMILIES_RENDER_DATA", main_stream);
	if (route_deferred_by_material_family)
	{
		for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
			m_wavefront_common.upload_material_family_render_data(*m_kernels[get_deferred_completion_kernel_name(family_index)],
																  "WAVEFRONT_COMPLETE_DEFERRED_RENDER_DATA", family_index, main_stream);
	}
	else
		m_wavefront_common.upload_main_stream_render_data(*m_kernels[COMPLETE_DEFERRED_PATHS_KERNEL], "WAVEFRONT_COMPLETE_DEFERRED_RENDER_DATA", main_stream);

	bool read_back_material_family_counts = m_wavefront_common.update_material_family_count_readback(overlap_family_kernels);

	for (unsigned int bounce_index = 0; bounce_index <= bounce_count; bounce_index++)
	{
		m_wavefront_common.reset_queue_count_async(1, main_stream);
		if (run_shadow_pass)
		{
			m_wavefront_common.reset_shadow_work_count_async(main_stream);
			m_wavefront_common.reset_shadow_queue_overflow_count_async(main_stream);
		}

		void* shade_launch_args[] = { &bounce_count };
		if (bounce_index == 0 && path_capacity > 0)
		{
			// The camera rays pass has already appended the primary hits to the material family queues
			launch_material_family_kernels_async(MaterialFamilyKernelType::PRIMARY_SHADING, shade_launch_args, material_family_count, family_block_count,
												 overlap_family_kernels);
			if (read_back_material_family_counts)
				m_wavefront_common.read_back_material_family_counts_async(get_material_family_count_readback_slot(MaterialFamilyKernelType::PRIMARY_SHADING),
																		  main_stream);
		}
		else if (bounce_index > 0 && path_capacity > 0)
		{
			if (!trace_classifies_material_families)
			{
				// The family-routed deferred completion of the previous bounce has consumed the family queues, they can be rebuilt from queue 0
				m_wavefront_common.reset_material_family_counts_async(main_stream);
				m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, nullptr,
																												 main_stream);
			}

			launch_material_family_kernels_async(MaterialFamilyKernelType::SECONDARY_SHADING, shade_launch_args, material_family_count, family_block_count,
												 overlap_family_kernels);
			// The first secondary bounce is representative of the following ones
			if (read_back_material_family_counts && bounce_index == 1)
				m_wavefront_common.read_back_material_family_counts_async(get_material_family_count_readback_slot(MaterialFamilyKernelType::SECONDARY_SHADING),
																		  main_stream);
		}

		if (run_shadow_pass)
			m_kernels[TRACE_SHADOW_RAYS_KERNEL]->launch_asynchronous_3D_block_count(shadow_block_count, 1, 1, 64, 1, 1, nullptr, main_stream);

		if (bounce_index >= bounce_count && !has_deferred_bsdf_sampling)
			continue;

		if (trace_classifies_material_families)
			// TracePaths appends the new hits to the material family queues, which the shading passes of this bounce have already consumed.
			// Queue 0 is not used in that case
			m_wavefront_common.reset_material_family_counts_async(main_stream);
		else
			m_wavefront_common.reset_queue_count_async(0, main_stream);
		m_wavefront_common.reset_queue_count_async(WAVEFRONT_COMPLETION_QUEUE_INDEX, main_stream);
		m_kernels[TRACE_PATHS_KERNEL]->launch_asynchronous(queue_block_size, 1, path_capacity, 1, nullptr, main_stream);

		bool deferred_completion_can_emit_shadow = run_shadow_pass && nee_estimator == LSS_RIS_BSDF_AND_LIGHT;
		if (deferred_completion_can_emit_shadow)
			m_wavefront_common.reset_shadow_work_count_async(main_stream);

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
	m_wavefront_common.reset_material_family_counts_async(main_stream);

	return true;
}

void WavefrontRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	// Every pass of the render graph that runs wavefront kernels updates the render data every frame but only the pass in use
	// may write render_data.wavefront_data: an unused pass leaves it to the pass in use.
	// The ReSTIR PT wavefront path sampling strategy has its own wavefront pass. With any other strategy, this pass writes the
	// wavefront data, including when it is not used so that the wavefront data is the default one when no wavefront pass is used
	bool pass_used				  = is_render_pass_used(*m_compiler_options);
	bool restir_pt_wavefront_used = m_compiler_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_PT_WAVEFRONT;
	if (!pass_used && restir_pt_wavefront_used)
		return;

	m_wavefront_common.update_render_data_wavefront_data(render_data, pass_used, *m_compiler_options);
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

WavefrontRenderPassCommon::MaterialFamilyKernels WavefrontRenderPass::get_material_family_kernels(MaterialFamilyKernelType kernel_type) const
{
	WavefrontRenderPassCommon::MaterialFamilyKernels family_kernels;
	for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
		family_kernels[family_index] = m_kernels.at(get_material_family_kernel_name(kernel_type, family_index));

	return family_kernels;
}

WavefrontRenderPassCommon::MaterialFamilyCountReadbackSlot WavefrontRenderPass::get_material_family_count_readback_slot(
	MaterialFamilyKernelType kernel_type) const
{
	// The deferred completion consumes the family queues of the secondary hits
	return kernel_type == MaterialFamilyKernelType::PRIMARY_SHADING ? WavefrontRenderPassCommon::MaterialFamilyCountReadbackSlot::PRIMARY_HITS
																	: WavefrontRenderPassCommon::MaterialFamilyCountReadbackSlot::SECONDARY_HITS;
}

void WavefrontRenderPass::launch_material_family_kernels_async(
	MaterialFamilyKernelType kernel_type, void** launch_args, unsigned int material_family_count, unsigned int block_count, bool overlap)
{
	m_wavefront_common.launch_material_family_kernels_async(get_material_family_kernels(kernel_type), get_material_family_count_readback_slot(kernel_type),
															launch_args, material_family_count, block_count, overlap, m_renderer->get_main_stream());
}

bool WavefrontRenderPass::get_overlap_material_family_kernels() const
{

	return m_wavefront_common.get_overlap_material_family_kernels();
}

void WavefrontRenderPass::set_overlap_material_family_kernels(bool overlap_material_family_kernels)
{
	m_wavefront_common.set_overlap_material_family_kernels(overlap_material_family_kernels);
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

	unsigned int material_family_count = WavefrontRenderPassCommon::get_material_family_count(*m_compiler_options);

	std::map<std::string, std::shared_ptr<GPUKernel>> active_kernels;
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		active_kernels[get_primary_shading_kernel_name(family_index)]	= m_kernels.at(get_primary_shading_kernel_name(family_index));
		active_kernels[get_secondary_shading_kernel_name(family_index)] = m_kernels.at(get_secondary_shading_kernel_name(family_index));
	}

	// Without family-routed deferred completion, TracePaths classifies the secondary hits itself
	if (WavefrontRenderPassCommon::uses_deferred_material_family_routing(*m_compiler_options))
		active_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL] = m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL);

	active_kernels[COMPLETE_DEFERRED_PATHS_KERNEL] = m_kernels.at(COMPLETE_DEFERRED_PATHS_KERNEL);
	if (WavefrontRenderPassCommon::uses_deferred_material_family_routing(*m_compiler_options))
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

	unsigned int material_family_count = WavefrontRenderPassCommon::get_material_family_count(*m_compiler_options);

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

bool WavefrontRenderPass::kernels_ready() const
{
	if (!m_kernels.at(TRACE_PATHS_KERNEL)->has_been_compiled() || !m_kernels.at(COMPLETE_DEFERRED_PATHS_KERNEL)->has_been_compiled() ||
		!m_kernels.at(TRACE_SHADOW_RAYS_KERNEL)->has_been_compiled() || !m_kernels.at(NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL)->has_been_compiled() ||
		m_wavefront_common.get_nee_deferred_mis_context_byte_size() == 0)

		return false;

	unsigned int material_family_count = WavefrontRenderPassCommon::get_material_family_count(*m_compiler_options);
	if (WavefrontRenderPassCommon::uses_deferred_material_family_routing(*m_compiler_options))
	{
		for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
			if (!m_kernels.at(get_deferred_completion_kernel_name(family_index))->has_been_compiled())

				return false;
	}

	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
		if (!m_kernels.at(get_primary_shading_kernel_name(family_index))->has_been_compiled() ||
			!m_kernels.at(get_secondary_shading_kernel_name(family_index))->has_been_compiled())

			return false;

	if (WavefrontRenderPassCommon::uses_deferred_material_family_routing(*m_compiler_options))
		return m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL)->has_been_compiled();

	return true;
}
