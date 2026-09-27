/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
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
const std::string WavefrontRenderPass::NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL				 = "Wavefront - NEE Deferred MIS Context Size";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_RESET_KERNEL				 = "Wavefront - Reset Material Family Routing";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL			 = "Wavefront - Classify Primary Material Families";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL			 = "Wavefront - Build Material Family Offsets";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL			 = "Wavefront - Scatter Primary Material Families";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL = "Wavefront - Classify Secondary Material Families";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL	 = "Wavefront - Scatter Secondary Material Families";

const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_KERNEL					= "Wavefront - Complete Terminated Paths";
const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL			= "Wavefront - Complete Terminated Diffuse Paths";
const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_GLASS_KERNEL				= "Wavefront - Complete Terminated Glass Paths";
const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL	= "Wavefront - Complete Terminated Single Metallic Paths";
const std::string WavefrontRenderPass::COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL	= "Wavefront - Complete Terminated Specular Diffuse Paths";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL = "Wavefront - Classify Deferred Material Families";
const std::string WavefrontRenderPass::MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL	= "Wavefront - Scatter Deferred Material Families";

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

	m_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL);
	m_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Utils/NEEDeferredMISContextSize.h");
	m_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL]->set_kernel_function_name("NEEDeferredMISContextSize");

	m_kernels[MATERIAL_FAMILY_ROUTING_RESET_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_RESET_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_RESET_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ResetMaterialFamilyRouting.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_RESET_KERNEL]->set_kernel_function_name("ResetMaterialFamilyRouting");

	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ClassifyPrimaryMaterialFamilies.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL]->set_kernel_function_name("ClassifyPrimaryMaterialFamilies");

	m_kernels[MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/BuildMaterialFamilyOffsets.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL]->set_kernel_function_name("BuildMaterialFamilyOffsets");

	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ScatterPrimaryMaterialFamilies.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL]->set_kernel_function_name("ScatterPrimaryMaterialFamilies");

	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																					   "/Wavefront/ClassifySecondaryMaterialFamilies.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->set_kernel_function_name("ClassifySecondaryMaterialFamilies");

	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ScatterSecondaryMaterialFamilies.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL]->set_kernel_function_name("ScatterSecondaryMaterialFamilies");

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

	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ClassifyDeferredMaterialFamilies.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL]->set_kernel_function_name("ClassifyDeferredMaterialFamilies");

	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL);
	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/ScatterDeferredMaterialFamilies.h");
	m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL]->set_kernel_function_name("ScatterDeferredMaterialFamilies");

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

		return false;

	render_data.wavefront_data = m_wavefront_data.to_device();

	unsigned int path_capacity = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	unsigned int bounce_count  = static_cast<unsigned int>(render_data.render_settings.nb_bounces);
	if (render_data.render_settings.do_render_low_resolution())
		bounce_count = std::min(3u, bounce_count);

	oroStream_t main_stream = m_renderer->get_main_stream();

	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	*host_pinned_render_data				 = render_data;

	m_kernels[TRACE_PATHS_KERNEL]->upload_to_module_global("WAVEFRONT_TRACE_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);

	int nee_estimator									= compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	WavefrontRoutingConfiguration routing_configuration = get_wavefront_routing_configuration(
		compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL),
		compiler_options.get_macro_value(GPUKernelCompilerOptions::WAVEFRONT_MATERIAL_SPECIALIZATION) == KERNEL_OPTION_TRUE, nee_estimator);
	bool route_primary_by_material_family  = routing_configuration.route_shading_by_material_family;
	bool has_deferred_bsdf_sampling		   = routing_configuration.requires_terminal_trace_completion;
	bool route_deferred_by_material_family = routing_configuration.route_deferred_completion_by_material_family;
	if (route_primary_by_material_family)
	{
		for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
		{
			m_kernels[get_primary_shading_kernel_name(family_index)]->upload_to_module_global("WAVEFRONT_SHADE_PRIMARY_RENDER_DATA", host_pinned_render_data,
																							  sizeof(HIPRTRenderData), main_stream);
			m_kernels[get_secondary_shading_kernel_name(family_index)]->upload_to_module_global("WAVEFRONT_SHADE_RENDER_DATA", host_pinned_render_data,
																								sizeof(HIPRTRenderData), main_stream);
		}

		m_kernels[MATERIAL_FAMILY_ROUTING_RESET_KERNEL]->upload_to_module_global("WAVEFRONT_RESET_MATERIAL_FAMILY_ROUTING_RENDER_DATA", host_pinned_render_data,
																				 sizeof(HIPRTRenderData), main_stream);
		m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL]->upload_to_module_global("WAVEFRONT_CLASSIFY_PRIMARY_MATERIAL_FAMILIES_RENDER_DATA",
																					host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
		m_kernels[MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL]->upload_to_module_global("WAVEFRONT_BUILD_MATERIAL_FAMILY_OFFSETS_RENDER_DATA",
																				   host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
		m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL]->upload_to_module_global("WAVEFRONT_SCATTER_PRIMARY_MATERIAL_FAMILIES_RENDER_DATA",
																				   host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
		m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->upload_to_module_global("WAVEFRONT_CLASSIFY_SECONDARY_MATERIAL_FAMILIES_RENDER_DATA",
																							  host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
		m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL]->upload_to_module_global("WAVEFRONT_SCATTER_SECONDARY_MATERIAL_FAMILIES_RENDER_DATA",
																							 host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
		if (route_deferred_by_material_family)
		{
			m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL]->upload_to_module_global("WAVEFRONT_CLASSIFY_DEFERRED_MATERIAL_FAMILIES_RENDER_DATA",
																								 host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL]->upload_to_module_global("WAVEFRONT_SCATTER_DEFERRED_MATERIAL_FAMILIES_RENDER_DATA",
																								host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
		}
	}
	else
	{
		m_kernels[SHADE_PRIMARY_PATHS_KERNEL]->upload_to_module_global("WAVEFRONT_SHADE_PRIMARY_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData),
																	   main_stream);
		m_kernels[SHADE_PATHS_KERNEL]->upload_to_module_global("WAVEFRONT_SHADE_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
	}
	if (route_deferred_by_material_family)
	{
		for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
			m_kernels[get_deferred_completion_kernel_name(family_index)]->upload_to_module_global(
				"WAVEFRONT_COMPLETE_DEFERRED_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);
	}
	else
		m_kernels[COMPLETE_DEFERRED_PATHS_KERNEL]->upload_to_module_global("WAVEFRONT_COMPLETE_DEFERRED_RENDER_DATA", host_pinned_render_data,
																		   sizeof(HIPRTRenderData), main_stream);

	unsigned int* zero = m_zero_host_pinned.get_host_pinned_pointer();
	zero[0]			   = 0;

	unsigned int queue_block_size	   = KernelBlockWidthHeight * KernelBlockWidthHeight;
	unsigned int wavefront_block_count = std::max(1u, std::min((path_capacity + 63u) / 64u, 2048u));

	for (unsigned int bounce_index = 0; bounce_index <= bounce_count; bounce_index++)
	{
		m_wavefront_data.get_queue_count_buffer(1).upload_data_async(zero, main_stream);

		bool route_by_family	  = false;
		void* shade_launch_args[] = { &bounce_count, &route_by_family };
		if (bounce_index == 0 && route_primary_by_material_family && path_capacity > 0)
		{
			m_kernels[MATERIAL_FAMILY_ROUTING_RESET_KERNEL]->launch_asynchronous_3D_block_count(1, 1, 1, 64, 1, 1, nullptr, main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, nullptr, main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL]->launch_asynchronous_3D_block_count(1, 1, 1, 64, 1, 1, nullptr, main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, nullptr, main_stream);

			route_by_family = true;
			for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
				m_kernels[get_primary_shading_kernel_name(family_index)]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1,
																											 shade_launch_args, main_stream);
		}

		if (bounce_index == 0 && !route_primary_by_material_family && path_capacity > 0)
		{
			m_kernels[SHADE_PRIMARY_PATHS_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, shade_launch_args, main_stream);
		}
		else if (bounce_index > 0 && route_primary_by_material_family && path_capacity > 0)
		{
			m_kernels[MATERIAL_FAMILY_ROUTING_RESET_KERNEL]->launch_asynchronous_3D_block_count(1, 1, 1, 64, 1, 1, nullptr, main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, nullptr,
																											 main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL]->launch_asynchronous_3D_block_count(1, 1, 1, 64, 1, 1, nullptr, main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, nullptr,
																											main_stream);

			route_by_family = true;
			for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
				m_kernels[get_secondary_shading_kernel_name(family_index)]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1,
																											   shade_launch_args, main_stream);
		}
		else if (bounce_index > 0)
			m_kernels[SHADE_PATHS_KERNEL]->launch_asynchronous(queue_block_size, 1, path_capacity, 1, shade_launch_args, main_stream);

		if (bounce_index >= bounce_count && !has_deferred_bsdf_sampling)
			continue;

		m_wavefront_data.get_queue_count_buffer(0).upload_data_async(zero, main_stream);
		m_wavefront_data.get_queue_count_buffer(WAVEFRONT_COMPLETION_QUEUE_INDEX).upload_data_async(zero, main_stream);
		m_kernels[TRACE_PATHS_KERNEL]->launch_asynchronous(queue_block_size, 1, path_capacity, 1, nullptr, main_stream);

		if (route_deferred_by_material_family && path_capacity > 0)
		{
			m_kernels[MATERIAL_FAMILY_ROUTING_RESET_KERNEL]->launch_asynchronous_3D_block_count(1, 1, 1, 64, 1, 1, nullptr, main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, nullptr,
																											main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL]->launch_asynchronous_3D_block_count(1, 1, 1, 64, 1, 1, nullptr, main_stream);
			m_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, nullptr,
																										   main_stream);
		}

		bool complete_by_family		 = route_deferred_by_material_family;
		void* complete_launch_args[] = { &complete_by_family };
		if (route_deferred_by_material_family)
		{
			for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
				m_kernels[get_deferred_completion_kernel_name(family_index)]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1,
																												 complete_launch_args, main_stream);
		}
		else
			m_kernels[COMPLETE_DEFERRED_PATHS_KERNEL]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, complete_launch_args,
																						  main_stream);
	}

	return true;
}

void WavefrontRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (is_render_pass_used(*m_compiler_options) && m_staging_buffers_allocated)
		render_data.wavefront_data = m_wavefront_data.to_device();
	else
		render_data.wavefront_data = WavefrontDataDevice();
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

bool WavefrontRenderPass::uses_material_family_routing(const GPUKernelCompilerOptions& compiler_options) const
{
	WavefrontRoutingConfiguration configuration =
		get_wavefront_routing_configuration(compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL),
											compiler_options.get_macro_value(GPUKernelCompilerOptions::WAVEFRONT_MATERIAL_SPECIALIZATION) == KERNEL_OPTION_TRUE,
											compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR));

	return configuration.route_shading_by_material_family;
}

bool WavefrontRenderPass::uses_deferred_material_family_routing(const GPUKernelCompilerOptions& compiler_options) const
{
	WavefrontRoutingConfiguration configuration =
		get_wavefront_routing_configuration(compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL),
											compiler_options.get_macro_value(GPUKernelCompilerOptions::WAVEFRONT_MATERIAL_SPECIALIZATION) == KERNEL_OPTION_TRUE,
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

	std::map<std::string, std::shared_ptr<GPUKernel>> active_kernels;
	if (uses_material_family_routing(*m_compiler_options))
	{
		for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
		{
			active_kernels[get_primary_shading_kernel_name(family_index)]	= m_kernels.at(get_primary_shading_kernel_name(family_index));
			active_kernels[get_secondary_shading_kernel_name(family_index)] = m_kernels.at(get_secondary_shading_kernel_name(family_index));
		}

		active_kernels[MATERIAL_FAMILY_ROUTING_RESET_KERNEL]			  = m_kernels.at(MATERIAL_FAMILY_ROUTING_RESET_KERNEL);
		active_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL]			  = m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL);
		active_kernels[MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL]			  = m_kernels.at(MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL);
		active_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL]			  = m_kernels.at(MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL);
		active_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL] = m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL);
		active_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL]  = m_kernels.at(MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL);
	}
	else
	{
		active_kernels[SHADE_PRIMARY_PATHS_KERNEL] = m_kernels.at(SHADE_PRIMARY_PATHS_KERNEL);
		active_kernels[SHADE_PATHS_KERNEL]		   = m_kernels.at(SHADE_PATHS_KERNEL);
	}

	active_kernels[COMPLETE_DEFERRED_PATHS_KERNEL] = m_kernels.at(COMPLETE_DEFERRED_PATHS_KERNEL);
	if (uses_deferred_material_family_routing(*m_compiler_options))
	{
		for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
			active_kernels[get_deferred_completion_kernel_name(family_index)] = m_kernels.at(get_deferred_completion_kernel_name(family_index));
		active_kernels[MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL] = m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL);
		active_kernels[MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL]	 = m_kernels.at(MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL);
	}

	active_kernels[TRACE_PATHS_KERNEL]					 = m_kernels.at(TRACE_PATHS_KERNEL);
	active_kernels[NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL] = m_kernels.at(NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL);

	return active_kernels;
}

std::map<std::string, std::shared_ptr<GPUKernel>> WavefrontRenderPass::get_tracing_kernels()
{
	if (!is_render_pass_used(*m_compiler_options))

		return {};

	std::map<std::string, std::shared_ptr<GPUKernel>> tracing_kernels;
	if (uses_material_family_routing(*m_compiler_options))
	{
		for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
		{
			tracing_kernels[get_primary_shading_kernel_name(family_index)]	 = m_kernels.at(get_primary_shading_kernel_name(family_index));
			tracing_kernels[get_secondary_shading_kernel_name(family_index)] = m_kernels.at(get_secondary_shading_kernel_name(family_index));
		}
	}
	else
	{
		tracing_kernels[SHADE_PRIMARY_PATHS_KERNEL] = m_kernels.at(SHADE_PRIMARY_PATHS_KERNEL);
		tracing_kernels[SHADE_PATHS_KERNEL]			= m_kernels.at(SHADE_PATHS_KERNEL);
	}

	tracing_kernels[TRACE_PATHS_KERNEL] = m_kernels.at(TRACE_PATHS_KERNEL);

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

bool WavefrontRenderPass::kernels_ready() const
{
	if (!m_kernels.at(TRACE_PATHS_KERNEL)->has_been_compiled() || !m_kernels.at(COMPLETE_DEFERRED_PATHS_KERNEL)->has_been_compiled() ||
		!m_kernels.at(NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL)->has_been_compiled() || m_nee_deferred_mis_context_byte_size == 0)

		return false;
	if (uses_deferred_material_family_routing(*m_compiler_options))
	{
		for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
			if (!m_kernels.at(get_deferred_completion_kernel_name(family_index))->has_been_compiled())

				return false;
		if (!m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_DEFERRED_KERNEL)->has_been_compiled() ||
			!m_kernels.at(MATERIAL_FAMILY_ROUTING_SCATTER_DEFERRED_KERNEL)->has_been_compiled())

			return false;
	}

	if (uses_material_family_routing(*m_compiler_options))
	{
		for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
			if (!m_kernels.at(get_primary_shading_kernel_name(family_index))->has_been_compiled() ||
				!m_kernels.at(get_secondary_shading_kernel_name(family_index))->has_been_compiled())

				return false;

		return m_kernels.at(MATERIAL_FAMILY_ROUTING_RESET_KERNEL)->has_been_compiled() &&
			   m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_KERNEL)->has_been_compiled() &&
			   m_kernels.at(MATERIAL_FAMILY_ROUTING_OFFSETS_KERNEL)->has_been_compiled() &&
			   m_kernels.at(MATERIAL_FAMILY_ROUTING_SCATTER_KERNEL)->has_been_compiled() &&
			   m_kernels.at(MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL)->has_been_compiled() &&
			   m_kernels.at(MATERIAL_FAMILY_ROUTING_SCATTER_SECONDARY_KERNEL)->has_been_compiled();
	}

	return m_kernels.at(SHADE_PRIMARY_PATHS_KERNEL)->has_been_compiled() && m_kernels.at(SHADE_PATHS_KERNEL)->has_been_compiled();
}

bool WavefrontRenderPass::resize_staging_buffers()
{
	unsigned int path_capacity					  = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	std::size_t ray_volume_state_byte_size		  = m_renderer->get_render_data().ray_volume_state_byte_size;
	bool allocate_resolved_material_control_cache = m_compiler_options->get_macro_value(GPUKernelCompilerOptions::USE_MATERIAL_TEXTURES) == KERNEL_OPTION_TRUE;
	bool allocate_material_family_routing		  = uses_material_family_routing(*m_compiler_options);
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
						m_wavefront_data.has_material_family_routing() != allocate_material_family_routing ||
						m_allocated_ray_volume_state_byte_size != ray_volume_state_byte_size ||
						m_allocated_nee_deferred_mis_context_byte_size != nee_deferred_mis_context_byte_size || m_render_data_host_pinned.size() != 1 ||
						m_zero_host_pinned.size() != 1;
	if (!needs_resize)

		return false;

	free_staging_buffers();

	m_wavefront_data.resize(path_capacity, ray_volume_state_byte_size, nee_deferred_mis_context_byte_size, allocate_resolved_material_control_cache,
							allocate_material_family_routing);
	m_render_data_host_pinned.resize_host_pinned_mem(1);
	m_zero_host_pinned.resize_host_pinned_mem(1);

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

	m_allocated_ray_volume_state_byte_size		   = 0;
	m_allocated_nee_deferred_mis_context_byte_size = 0;
	m_staging_buffers_allocated					   = false;
}
