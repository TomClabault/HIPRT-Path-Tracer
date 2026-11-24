/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/kernels/ReSTIR/SPMIS/BuildCDFs.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReSTIRPTRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

#include <algorithm>
#include <cstdlib>
#include <ranges>
const std::string ReSTIRPTRenderPass::RESTIR_PT_RENDER_PASS_NAME								  = "ReSTIR PT Render pass";
const std::string ReSTIRPTRenderPass::RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_KERNEL_ID				  = "ReSTIR PT - Wavefront Initial candidates - Shade Primary";
const std::string ReSTIRPTRenderPass::RESTIR_PT_WAVEFRONT_SHADE_KERNEL_ID						  = "ReSTIR PT - Wavefront Initial candidates - Shade";
const std::string ReSTIRPTRenderPass::RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID						  = "ReSTIR PT - Wavefront Initial candidates - Trace";
const std::string ReSTIRPTRenderPass::RESTIR_PT_WAVEFRONT_COMPLETE_KERNEL_ID					  = "ReSTIR PT - Wavefront Initial candidates - Complete";
const std::string ReSTIRPTRenderPass::RESTIR_PT_WAVEFRONT_NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL_ID = "ReSTIR PT - Wavefront NEE Deferred MIS Context Size";
const std::string ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID						  = "ReSTIR PT - Temporal reuse";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID							  = "ReSTIR PT - Spatial reuse";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID					  = "ReSTIR PT - Spatial reuse SPMIS";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID =
	"ReSTIR PT - Spatial reuse SPMIS Union Domain Shadow Split Prepare";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID =
	"ReSTIR PT - Spatial reuse SPMIS Union Domain Shadow Split Trace Selection";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID =
	"ReSTIR PT - Spatial reuse SPMIS Union Domain Shadow Split Weights";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID =
	"ReSTIR PT - Spatial reuse SPMIS Union Domain Shadow Split Resolve";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID					   = "ReSTIR PT - Shading";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID		   = "ReSTIR PT - SPMIS Reset Buffers";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID	   = "ReSTIR PT - SPMIS Reset Cells Data";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID			   = "ReSTIR PT - SPMIS Count Cells";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID		   = "ReSTIR PT - SPMIS Compute Offsets";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID				   = "ReSTIR PT - SPMIS Sort";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID			   = "ReSTIR PT - SPMIS Build CDFs";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID = "ReSTIR PT - SPMIS Compute Per Cell Noise";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID = "ReSTIR PT - SPMIS Build Neighbor Records";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID	   = "ReSTIR PT - SPMIS Build Alias Tables";

const std::unordered_map<std::string, std::string> ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES = {
	{ RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID, "ReSTIR_PT_TemporalReuse" },
	{ RESTIR_PT_SPATIAL_REUSE_KERNEL_ID, "ReSTIR_PT_SpatialReuse" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID, "ReSTIR_PT_SpatialReuseSPMIS" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID, "ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitPrepare" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID, "ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitTraceSelection" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID, "ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitWeights" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID, "ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitResolve" },
	{ RESTIR_PT_SHADING_KERNEL_ID, "ReSTIR_PT_Shading" },
	{ RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID, "ReSTIR_PT_SPMIS_ResetBuffers" },
	{ RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID, "ReSTIR_PT_SPMIS_ResetCellsData" },
	{ RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID, "ReSTIR_PT_SPMIS_CountCells" },
	{ RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID, "ReSTIR_PT_SPMIS_ComputeOffsets" },
	{ RESTIR_PT_SPMIS_SORT_KERNEL_ID, "ReSTIR_PT_SPMIS_Sort" },
	{ RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID, "ReSTIR_PT_SPMIS_BuildCDFs" },
	{ RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID, "ReSTIR_PT_SPMIS_ComputeCellsVariance" },
	{ RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID, "ReSTIR_PT_SPMIS_BuildNeighborRecords" },
	{ RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID, "ReSTIR_PT_SPMIS_BuildAliasTables" }
};

const std::unordered_map<std::string, std::string> ReSTIRPTRenderPass::KERNEL_FILES = {
	{ RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/TemporalReuse.h" },
	{ RESTIR_PT_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/SpatialReuse.h" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/SpatialReuseSPMIS.h" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID,
	  DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/SpatialReuseSPMISUnionDomainShadowSplitPrepare.h" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID,
	  DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/SpatialReuseSPMISUnionDomainShadowSplitTraceSelection.h" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID,
	  DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/SpatialReuseSPMISUnionDomainShadowSplitWeights.h" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID,
	  DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/SpatialReuseSPMISUnionDomainShadowSplitResolve.h" },
	{ RESTIR_PT_SHADING_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/Shading.h" },
	{ RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/ResetBuffers.h" },
	{ RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/ResetCellsData.h" },
	{ RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/CountCells.h" },
	{ RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/ComputeOffsets.h" },
	{ RESTIR_PT_SPMIS_SORT_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/Sort.h" },
	{ RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/BuildCDFs.h" },
	{ RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/ComputeCellsVariance.h" },
	{ RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/BuildNeighborRecords.h" },
	{ RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/BuildAliasTables.h" }
};

ReSTIRPTRenderPass::ReSTIRPTRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(ReSTIRPTRenderPass::RESTIR_PT_RENDER_PASS_NAME, renderer, options)
{
	m_render_data_host_pinned.resize_host_pinned_mem(ReSTIRPTRenderPass::RENDER_DATA_UPLOAD_SLOT_COUNT);

	for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
	{
		std::string shade_primary_kernel_id = get_material_family_kernel_id(RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_KERNEL_ID, family_index);
		m_kernels[shade_primary_kernel_id]	= std::make_shared<GPUKernel>(this->get_name() + "::" + shade_primary_kernel_id);
		m_kernels[shade_primary_kernel_id]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/Wavefront/InitialCandidatesShadePrimary.h");
		m_kernels[shade_primary_kernel_id]->set_kernel_function_name("ReSTIR_PT_WavefrontInitialCandidatesShadePrimary");

		std::string shade_kernel_id = get_material_family_kernel_id(RESTIR_PT_WAVEFRONT_SHADE_KERNEL_ID, family_index);
		m_kernels[shade_kernel_id]	= std::make_shared<GPUKernel>(this->get_name() + "::" + shade_kernel_id);
		m_kernels[shade_kernel_id]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/Wavefront/InitialCandidatesShade.h");
		m_kernels[shade_kernel_id]->set_kernel_function_name("ReSTIR_PT_WavefrontInitialCandidatesShade");
	}

	m_kernels[RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID] = std::make_shared<GPUKernel>(this->get_name() + "::" + RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID);
	m_kernels[RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/Wavefront/InitialCandidatesTrace.h");
	m_kernels[RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID]->set_kernel_function_name("ReSTIR_PT_WavefrontInitialCandidatesTrace");

	m_kernels[RESTIR_PT_WAVEFRONT_COMPLETE_KERNEL_ID] = std::make_shared<GPUKernel>(this->get_name() + "::" + RESTIR_PT_WAVEFRONT_COMPLETE_KERNEL_ID);
	m_kernels[RESTIR_PT_WAVEFRONT_COMPLETE_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/Wavefront/InitialCandidatesComplete.h");
	m_kernels[RESTIR_PT_WAVEFRONT_COMPLETE_KERNEL_ID]->set_kernel_function_name("ReSTIR_PT_WavefrontInitialCandidatesComplete");

	m_kernels[RESTIR_PT_WAVEFRONT_NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + RESTIR_PT_WAVEFRONT_NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL_ID);
	m_kernels[RESTIR_PT_WAVEFRONT_NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Utils/NEEDeferredMISContextSize.h");
	m_kernels[RESTIR_PT_WAVEFRONT_NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL_ID]->set_kernel_function_name("NEEDeferredMISContextSize");

	for (const std::string& kernel_id : get_wavefront_kernel_ids(KernelMaterialSpecializationCount))
	{
		m_kernels[kernel_id]->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
		m_kernels[kernel_id]->get_kernel_options().set_macro_value_independently(GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION,
																				 KernelMaterialSpecializationAll);
	}

	// The shading kernels trace the shadow rays of their light candidates
	for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
	{
		std::string shade_primary_kernel_id = get_material_family_kernel_id(RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_KERNEL_ID, family_index);
		std::string shade_kernel_id			= get_material_family_kernel_id(RESTIR_PT_WAVEFRONT_SHADE_KERNEL_ID, family_index);
		for (const std::string& kernel_id : { shade_primary_kernel_id, shade_kernel_id })
		{
			GPUKernelCompilerOptions& kernel_options = m_kernels[kernel_id]->get_kernel_options();

			kernel_options.set_macro_value_independently(GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION, family_index);
			kernel_options.set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
			kernel_options.set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
		}
	}

	m_kernels[RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
																						 KERNEL_OPTION_TRUE);
	m_kernels[RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																								GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																							   GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																									 GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	// The kernels of the union domain spatial reuse use the shared stack BVH traversal, as the legacy SPMIS spatial reuse kernel
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID]->synchronize_options_with(
		m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID] = std::make_shared<GPUKernel>(
		this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID]->synchronize_options_with(
		m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID]->synchronize_options_with(
		m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID]->synchronize_options_with(
		m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
																									 KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE,
																									 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID]->synchronize_options_with(m_compiler_options);
}

std::string ReSTIRPTRenderPass::get_material_family_kernel_id(const std::string& kernel_id, unsigned int family_index)
{
	switch (family_index)
	{
	case KernelMaterialSpecializationDiffuse:

		return kernel_id + " Diffuse";
	case KernelMaterialSpecializationGlass:

		return kernel_id + " Glass";
	case KernelMaterialSpecializationSingleMetallic:

		return kernel_id + " Single Metallic";
	case KernelMaterialSpecializationSpecularDiffuse:

		return kernel_id + " Specular Diffuse";
	default:

		return kernel_id;
	}
}

std::vector<std::string> ReSTIRPTRenderPass::get_wavefront_kernel_ids(unsigned int material_family_count)
{
	std::vector<std::string> kernel_ids;

	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		kernel_ids.push_back(get_material_family_kernel_id(RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_KERNEL_ID, family_index));
		kernel_ids.push_back(get_material_family_kernel_id(RESTIR_PT_WAVEFRONT_SHADE_KERNEL_ID, family_index));
	}

	kernel_ids.push_back(RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID);
	kernel_ids.push_back(RESTIR_PT_WAVEFRONT_COMPLETE_KERNEL_ID);
	kernel_ids.push_back(RESTIR_PT_WAVEFRONT_NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL_ID);

	return kernel_ids;
}

void ReSTIRPTRenderPass::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
								   const std::vector<hiprtFuncNameSet>& func_name_sets,
								   bool silent,
								   bool use_cache)
{
	// Kernel options can change the NEE deferred MIS context layout. Query its newly compiled size
	// before deciding whether the wavefront staging buffers need to be resized.
	m_wavefront_common.set_nee_deferred_mis_context_byte_size_dirty();

	RenderPass::recompile(hiprt_orochi_ctx, func_name_sets, silent, use_cache);
}

void ReSTIRPTRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used(*m_compiler_options))

		return;

	m_initial_candidates_buffer.resize(new_width * new_height);
	m_initial_candidates_working_samples_buffer.resize(new_width * new_height);
	m_temporal_buffer.resize(new_width * new_height);
	m_spatial_buffer.resize(new_width * new_height);

	m_spmis_data.resize(new_width, new_height);

	if (m_spmis_union_domain_shadow_split_buffer.size() > 0)
		// Keeping the state of the union domain spatial reuse kernels allocated, at the new resolution
		resize_spmis_union_domain_shadow_split_buffer(new_width, new_height);
}

bool ReSTIRPTRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
													  const std::vector<hiprtFuncNameSet>& func_name_sets,
													  bool silent,
													  bool use_cache)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))

		return false;

	bool recompiled = false;

	bool need_temporal = render_data.render_settings.restir_pt_settings.temporal_pass.do_temporal_reuse_pass &&
						 !m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_temporal;

	if (need_temporal)
		// Temporal needed
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_spatial = render_data.render_settings.restir_pt_settings.spatial_pass.do_spatial_reuse_pass &&
						!m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_spatial;

	if (need_spatial)
		// Spatial needed
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool using_spmis = (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) ==
							RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
						m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) ==
							RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE);
	if (using_spmis)
	{
		if (!m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->has_been_compiled())
		{
			// SPMIS has never been compiled but we need it now
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets,
																															  use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID]->compile(
				hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets,
																															  use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets,
																															  use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

			recompiled = true;
		}
	}

	unsigned int material_family_count = WavefrontRenderPassCommon::get_material_family_count(*m_compiler_options);
	for (const std::string& kernel_id : get_wavefront_kernel_ids(material_family_count))
	{
		if (m_kernels[kernel_id]->has_been_compiled())
			continue;

		recompiled = true;
		m_kernels[kernel_id]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

		if (kernel_id == RESTIR_PT_WAVEFRONT_NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL_ID)
			m_wavefront_common.set_nee_deferred_mis_context_byte_size_dirty();
	}

	return recompiled;
}

bool ReSTIRPTRenderPass::pre_frame_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (is_render_pass_used(*m_compiler_options))
	{
		// Resetting this flag as this is a new frame
		render_data.render_settings.do_update_status_buffers = false;

		if (!render_data.render_settings.accumulate)
			render_data.render_settings.sample_number = 0;
	}

	bool render_data_invalidated = false;

	int2_t render_resolution = m_renderer->m_render_resolution;

	if (is_render_pass_used(*m_compiler_options))
	{
		// ReSTIR PT enabled
		bool initial_candidates_reservoir_needs_resize	= m_initial_candidates_buffer.size() == 0;
		bool temporal_candidates_reservoir_needs_resize = m_temporal_buffer.size() == 0;
		bool spatial_candidates_reservoir_needs_resize	= m_spatial_buffer.size() == 0;

		if (initial_candidates_reservoir_needs_resize || temporal_candidates_reservoir_needs_resize || spatial_candidates_reservoir_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			render_data_invalidated = true;

		if (initial_candidates_reservoir_needs_resize)
		{
			m_initial_candidates_buffer.resize(render_resolution.x * render_resolution.y);
			m_initial_candidates_working_samples_buffer.resize(render_resolution.x * render_resolution.y);
		}

		if (temporal_candidates_reservoir_needs_resize)
			m_temporal_buffer.resize(render_resolution.x * render_resolution.y);

		if (spatial_candidates_reservoir_needs_resize)
			m_spatial_buffer.resize(render_resolution.x * render_resolution.y);

		render_data_invalidated |= pre_render_update_spmis_buffers(render_data, *m_renderer->get_global_compiler_options());
		render_data_invalidated |= pre_render_update_wavefront_buffers();

		// Arbitrary setting this one so that we're sure it's pointing to a valid buffer when all the buffers are resized
		m_last_temporal_output_reservoirs = m_initial_candidates_buffer.get_device_pointer();
	}
	else
	{
		// ReSTIR PT disabled, we're going to free the buffers if that's not already done
		if (m_initial_candidates_buffer.size() > 0)
		{
			m_initial_candidates_buffer.free();
			m_initial_candidates_working_samples_buffer.free();

			render_data_invalidated = true;
		}

		if (m_temporal_buffer.size() > 0)
		{
			m_temporal_buffer.free();

			render_data_invalidated = true;
		}

		if (m_spatial_buffer.size() > 0)
		{
			m_spatial_buffer.free();

			render_data_invalidated = true;
		}

		render_data_invalidated |= m_spmis_data.free();
		render_data_invalidated |= free_spmis_union_domain_shadow_split_buffer();
		render_data_invalidated |= free_wavefront_buffers();
	}

	if (render_data.render_settings.restir_pt_settings.spatial_pass.auto_reuse_radius)
		// A percentage of the maximum render resolution extent for automatic spatial reuse radius
		render_data.render_settings.restir_pt_settings.spatial_pass.reuse_radius =
			hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) * ReSTIRPTRenderPass::AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE;

	return render_data_invalidated;
}

bool ReSTIRPTRenderPass::pre_render_update_wavefront_buffers()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	unsigned int path_capacity	 = static_cast<unsigned int>(m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);
	bool render_data_invalidated = m_wavefront_common.resize_staging_buffers(path_capacity, render_data.ray_volume_state_byte_size, *m_compiler_options,
																			 m_kernels[RESTIR_PT_WAVEFRONT_NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL_ID]);

	if (m_restir_pt_wavefront_data.path_capacity() != path_capacity)
	{
		m_restir_pt_wavefront_data.resize(path_capacity);

		render_data_invalidated = true;
	}

	m_path_tree_count =
		static_cast<unsigned int>(std::max(1, m_compiler_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_INITIAL_PATH_TREES_COUNT)));

	// Not part of the render data: only the render pass copies the queues to and from these buffers
	std::size_t backup_family_index_count = m_path_tree_count > 1 ? static_cast<std::size_t>(path_capacity) * KernelMaterialSpecializationCount : 0;
	if (m_primary_material_family_indices_backup.size() != backup_family_index_count)
	{
		if (backup_family_index_count > 0)
		{
			m_primary_material_family_indices_backup.resize(static_cast<int>(backup_family_index_count));
			m_primary_material_family_counts_backup.resize(KernelMaterialSpecializationCount);
		}
		else
		{
			m_primary_material_family_indices_backup.free_no_error();
			m_primary_material_family_counts_backup.free_no_error();
		}
	}

	return render_data_invalidated;
}

bool ReSTIRPTRenderPass::free_wavefront_buffers()
{
	bool had_buffers = m_wavefront_common.are_staging_buffers_allocated();
	m_wavefront_common.free_staging_buffers();

	had_buffers |= m_restir_pt_wavefront_data.free();

	m_primary_material_family_indices_backup.free_no_error();
	m_primary_material_family_counts_backup.free_no_error();

	return had_buffers;
}

void ReSTIRPTRenderPass::copy_primary_material_family_queues_async(bool backup, oroStream_t stream)
{
	WavefrontDataDevice wavefront_data = m_wavefront_common.get_wavefront_data_device(*m_compiler_options);

	void* family_indices		= wavefront_data.material_family_indices;
	void* family_counts			= wavefront_data.material_family_counts;
	void* family_indices_backup = m_primary_material_family_indices_backup.get_device_pointer();
	void* family_counts_backup	= m_primary_material_family_counts_backup.get_device_pointer();

	std::size_t family_indices_byte_size = m_primary_material_family_indices_backup.get_byte_size();
	std::size_t family_counts_byte_size	 = m_primary_material_family_counts_backup.get_byte_size();

	if (backup)
	{
		OROCHI_CHECK_ERROR(oroMemcpyAsync(family_indices_backup, family_indices, family_indices_byte_size, oroMemcpyDeviceToDevice, stream));
		OROCHI_CHECK_ERROR(oroMemcpyAsync(family_counts_backup, family_counts, family_counts_byte_size, oroMemcpyDeviceToDevice, stream));
	}
	else
	{
		OROCHI_CHECK_ERROR(oroMemcpyAsync(family_indices, family_indices_backup, family_indices_byte_size, oroMemcpyDeviceToDevice, stream));
		OROCHI_CHECK_ERROR(oroMemcpyAsync(family_counts, family_counts_backup, family_counts_byte_size, oroMemcpyDeviceToDevice, stream));
	}
}

bool ReSTIRPTRenderPass::pre_render_update_spmis_buffers(const HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	int mis_weight_type = compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE);

	bool render_data_invalidated = false;

	if (mis_weight_type == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
		mis_weight_type == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
	{
		if (m_spmis_data.maximum_size() == 0)
		{
			m_spmis_data.resize(render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y);

			render_data_invalidated = true;
		}

		// The buffers of the union domain reuse are only allocated while the spatial reuse kernel uses them
		if (render_data.render_settings.restir_pt_settings.do_spmis_union_domain_reuse(mis_weight_type))
		{
			if (!m_spmis_data.union_domain_buffers_allocated())
			{
				m_spmis_data.resize_union_domain_buffers(render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y);

				render_data_invalidated = true;
			}
		}
		else
			render_data_invalidated |= m_spmis_data.free_union_domain_buffers();
	}
	else
	{
		if (m_spmis_data.maximum_size() > 0)
		{
			m_spmis_data.free();

			render_data_invalidated = true;
		}
	}

	render_data_invalidated |= pre_render_update_spmis_union_domain_shadow_split_buffer(render_data, compiler_options);

	return render_data_invalidated;
}

bool ReSTIRPTRenderPass::pre_render_update_spmis_union_domain_shadow_split_buffer(const HIPRTRenderData& render_data,
																				  GPUKernelCompilerOptions& compiler_options)
{
	int mis_weight_type = compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE);

	// Only allocated while the union domain reuse runs: freed when the settings make the legacy SPMIS spatial reuse run
	bool union_domain_reuse = render_data.render_settings.restir_pt_settings.do_spmis_union_domain_reuse(mis_weight_type) &&
							  render_data.render_settings.restir_pt_settings.spatial_pass.do_spatial_reuse_pass;
	if (!union_domain_reuse)
		return free_spmis_union_domain_shadow_split_buffer();

	if (m_spmis_union_domain_shadow_split_buffer.size() > 0)
		return false;

	resize_spmis_union_domain_shadow_split_buffer(render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y);

	return true;
}

void ReSTIRPTRenderPass::resize_spmis_union_domain_shadow_split_buffer(unsigned int width, unsigned int height)
{
	unsigned int pixel_count = width * height;

	ReSTIRPTSPMISUnionDomainShadowSplitBuffers unused_shadow_split_buffers;
	size_t buffer_byte_size = ReSTIRPTSPMISUnionDomainShadowSplitBuffers::layout(nullptr, pixel_count, unused_shadow_split_buffers);

	m_spmis_union_domain_shadow_split_buffer.resize(static_cast<int>((buffer_byte_size + sizeof(float4_t) - 1) / sizeof(float4_t)));
	m_spmis_union_domain_shadow_split_pixel_count = pixel_count;
}

bool ReSTIRPTRenderPass::free_spmis_union_domain_shadow_split_buffer()
{
	if (m_spmis_union_domain_shadow_split_buffer.size() == 0)
		return false;

	m_spmis_union_domain_shadow_split_buffer.free();
	m_spmis_union_domain_shadow_split_pixel_count = 0;

	return true;
}

void ReSTIRPTRenderPass::configure_initial_candidates_pass(HIPRTRenderData& render_data)
{
	render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer = m_initial_candidates_buffer.get_device_pointer();
	render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_working_samples_buffer =
		m_initial_candidates_working_samples_buffer.get_device_pointer();
}

void ReSTIRPTRenderPass::launch_initial_candidates_pass(HIPRTRenderData& render_data)
{
	oroStream_t main_stream = m_renderer->get_main_stream();

	// With several path trees, the primary hits material family queues must be kept for the path trees after the first one
	bool primary_material_family_queues_kept = m_path_tree_count == 1 || m_primary_material_family_indices_backup.size() > 0;

	if (!wavefront_kernels_ready() || !m_wavefront_common.are_staging_buffers_allocated() || m_restir_pt_wavefront_data.path_capacity() == 0 ||
		!primary_material_family_queues_kept)
	{
		if (m_wavefront_common.are_staging_buffers_allocated())
			// The camera rays pass may already have appended this frame's primary hits to the material family queues
			m_wavefront_common.reset_material_family_counts_async(main_stream);

		return;
	}

	render_data.wavefront_data			 = m_wavefront_common.get_wavefront_data_device(*m_compiler_options);
	render_data.restir_pt_wavefront_data = m_restir_pt_wavefront_data.to_device();

	unsigned int path_capacity = static_cast<unsigned int>(m_wavefront_common.get_wavefront_data().path_capacity());

	unsigned int bounce_count = static_cast<unsigned int>(render_data.render_settings.nb_bounces);
	if (render_data.render_settings.do_render_low_resolution())
		// Reducing the number of bounces to 3 if rendering at low resolution
		// for better interactivity
		bounce_count = std::min(3u, bounce_count);

	unsigned int queue_block_size	   = KernelBlockWidthHeight * KernelBlockWidthHeight;
	unsigned int wavefront_block_count = WavefrontRenderPassCommon::get_wavefront_block_count(path_capacity);
	unsigned int material_family_count = WavefrontRenderPassCommon::get_material_family_count(*m_compiler_options);

	unsigned int family_block_count;
	bool overlap_family_kernels =
		m_wavefront_common.prepare_material_family_render_data(render_data, material_family_count, wavefront_block_count, family_block_count);

	m_wavefront_common.upload_main_stream_render_data(*m_kernels[RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID], "RESTIR_PT_WAVEFRONT_TRACE_RENDER_DATA", main_stream);
	m_wavefront_common.upload_main_stream_render_data(*m_kernels[RESTIR_PT_WAVEFRONT_COMPLETE_KERNEL_ID], "RESTIR_PT_WAVEFRONT_COMPLETE_RENDER_DATA",
													  main_stream);

	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		m_wavefront_common.upload_material_family_render_data(
			*m_kernels[get_material_family_kernel_id(RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_KERNEL_ID, family_index)],
			"RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_RENDER_DATA", family_index, main_stream);
		m_wavefront_common.upload_material_family_render_data(*m_kernels[get_material_family_kernel_id(RESTIR_PT_WAVEFRONT_SHADE_KERNEL_ID, family_index)],
															  "RESTIR_PT_WAVEFRONT_SHADE_RENDER_DATA", family_index, main_stream);
	}

	bool read_back_material_family_counts = m_wavefront_common.update_material_family_count_readback(overlap_family_kernels);

	WavefrontRenderPassCommon::MaterialFamilyKernels shade_primary_kernels = get_material_family_kernels(RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_KERNEL_ID);
	WavefrontRenderPassCommon::MaterialFamilyKernels shade_kernels		   = get_material_family_kernels(RESTIR_PT_WAVEFRONT_SHADE_KERNEL_ID);

	if (m_path_tree_count > 1)
		copy_primary_material_family_queues_async(true, main_stream);

	for (unsigned int path_tree_index = 0; path_tree_index < m_path_tree_count; path_tree_index++)
	{
		if (path_tree_index > 0)
			copy_primary_material_family_queues_async(false, main_stream);

		for (unsigned int bounce_index = 0; bounce_index <= bounce_count; bounce_index++)
		{
			m_wavefront_common.reset_queue_count_async(1, main_stream);
			m_wavefront_common.reset_queue_count_async(WAVEFRONT_COMPLETION_QUEUE_INDEX, main_stream);

			void* shade_launch_args[] = { &bounce_count, &path_tree_index };

			if (bounce_index == 0)
			{
				// The camera rays pass has appended the primary hits to the material family queues
				m_wavefront_common.launch_material_family_kernels_async(
					shade_primary_kernels, WavefrontRenderPassCommon::MaterialFamilyCountReadbackSlot::PRIMARY_HITS, shade_launch_args, material_family_count,
					family_block_count, overlap_family_kernels, main_stream);

				if (read_back_material_family_counts && path_tree_index == 0)
					m_wavefront_common.read_back_material_family_counts_async(WavefrontRenderPassCommon::MaterialFamilyCountReadbackSlot::PRIMARY_HITS,
																			  main_stream);
			}
			else
			{
				m_wavefront_common.launch_material_family_kernels_async(
					shade_kernels, WavefrontRenderPassCommon::MaterialFamilyCountReadbackSlot::SECONDARY_HITS, shade_launch_args, material_family_count,
					family_block_count, overlap_family_kernels, main_stream);

				// The first secondary bounce is representative of the following ones
				if (read_back_material_family_counts && path_tree_index == 0 && bounce_index == 1)
					m_wavefront_common.read_back_material_family_counts_async(WavefrontRenderPassCommon::MaterialFamilyCountReadbackSlot::SECONDARY_HITS,
																			  main_stream);
			}

			// The trace kernel appends the new hits to the material family queues, which the shading kernels of this bounce have already consumed
			m_wavefront_common.reset_material_family_counts_async(main_stream);
			m_kernels[RESTIR_PT_WAVEFRONT_TRACE_KERNEL_ID]->launch_asynchronous(queue_block_size, 1, path_capacity, 1, nullptr, main_stream);

			void* complete_launch_args[] = { &path_tree_index };
			m_kernels[RESTIR_PT_WAVEFRONT_COMPLETE_KERNEL_ID]->launch_asynchronous_3D_block_count(wavefront_block_count, 1, 1, 64, 1, 1, complete_launch_args,
																								  main_stream);
		}
	}

	// The camera rays pass of the next frame appends the primary hits to the material family queues
	m_wavefront_common.reset_material_family_counts_async(main_stream);
}

void ReSTIRPTRenderPass::launch_spmis_create_reuse_cells_pass(HIPRTRenderData& render_data,
															  GPUKernelCompilerOptions& compiler_options,
															  ReSTIRPTReservoir* input_reservoirs)
{
	if (compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) != RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS &&
		compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) != RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)

		return;

	const ReSTIRPTSPMISSettings& spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;

	unsigned int num_cells = spmis_settings.pixel_hashes_count;
	upload_render_data(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID, render_data);
	void* reset_counters_launch_args[] = { &num_cells };
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1,
																								   reset_counters_launch_args, m_renderer->get_main_stream());

	unsigned short int* cell_counters =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_COUNTERS>().get_device_pointer();
	unsigned short int* cell_non_zero_reservoir_counters =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_NON_ZERO_RESERVOIR_COUNTERS>().get_device_pointer();
	unsigned int* cell_confidence_sums =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_CONFIDENCE_SUMS>().get_device_pointer();
	unsigned int* all_pixels_hashes =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_ALL_PIXEL_HASHES>().get_device_pointer();
	unsigned int* all_pixels_index_in_cell =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_ALL_PIXEL_INDEX_IN_CELL>().get_device_pointer();
	bool count_important			= true; // We want to count the important pixels first so that they are sorted at the beginning of their cell
	void* count_cells_launch_args[] = { &all_pixels_hashes,	   &all_pixels_index_in_cell, &cell_counters, &cell_non_zero_reservoir_counters,
										&cell_confidence_sums, &input_reservoirs,		  &num_cells,	  &count_important };
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1, count_cells_launch_args,
																							  m_renderer->get_main_stream());

	// And then a second call counting the non-important pixels so that they are sorted after the important ones in their cell
	count_important = false;
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1, count_cells_launch_args,
																							  m_renderer->get_main_stream());

	unsigned int* cell_global_offset_counter =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_GLOBAL_OFFSET_COUNTER>().get_device_pointer();
	unsigned int* cell_offsets = m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_OFFSETS>().get_device_pointer();
	void* compute_offsets_launch_args[] = { &cell_counters, &cell_global_offset_counter, &cell_offsets, &num_cells };
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1,
																								  compute_offsets_launch_args, m_renderer->get_main_stream());

	unsigned int* pixel_indices_sorted =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_PIXEL_INDICES_SORTED>().get_device_pointer();
	void* sort_launch_args[] = { &all_pixels_hashes, &all_pixels_index_in_cell, &cell_offsets, &pixel_indices_sorted, &num_cells };
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1, sort_launch_args,
																					   m_renderer->get_main_stream());

	float* cell_cdfs			  = m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_CDFS>().get_device_pointer();
	unsigned int* cell_alive_list = m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_ALIVE_LIST>().get_device_pointer();
	unsigned int* cell_alive_count =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_TOTAL_COUNT_COUNTER>().get_device_pointer();

	bool union_domain_reuse = render_data.render_settings.restir_pt_settings.do_spmis_union_domain_reuse(
		compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE));

	// Only the legacy SPMIS spatial reuse samples the CDFs through their LUTs, the union domain reuse samples the alias tables of the cells
	bool build_cdf_luts = !union_domain_reuse;
	if (build_cdf_luts)
	{
		// The LUT buffer is sized for the alive cells: the count is read back on the host, which waits for the kernels above to complete.
		// The cell kernels below read the count on the GPU, the union domain reuse does not need it on the host
		unsigned int cell_alive_count_host =
			m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_TOTAL_COUNT_COUNTER>().download_data()[0];

		resize_cdf_luts_buffer(render_data, cell_alive_count_host);
	}

	// The LUT buffer is only allocated for the legacy SPMIS spatial reuse
	unsigned short int* cell_cdf_luts =
		build_cdf_luts ? m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_CDF_LUTS>().get_device_pointer() : nullptr;
	unsigned int* cell_cdf_lut_offsets =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_CDF_LUT_OFFSETS>().get_device_pointer();
	void* build_cdfs_launch_args[] = { &cell_non_zero_reservoir_counters,
									   &cell_offsets,
									   &cell_alive_list,
									   &pixel_indices_sorted,
									   &input_reservoirs,
									   &cell_cdfs,
									   &cell_cdf_luts,
									   &cell_cdf_lut_offsets,
									   &build_cdf_luts,
									   &cell_alive_count,
									   &num_cells };

	// The cell kernels run one block per alive cell in a grid sized without the number of alive cells: their blocks loop over the cells.
	// Always dispatching 1024 sized blocks for the build cdfs kernel, since the kernel is designed to handle that many threads per cell
	unsigned int build_cdfs_block_count = get_spmis_cell_kernel_block_count(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID, 1024, num_cells);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->launch_asynchronous(1024, 1, build_cdfs_block_count * 1024, 1, build_cdfs_launch_args,
																							 m_renderer->get_main_stream());

	// Variance is only needed for the first frame to seed the cached cell search radius;
	// subsequent frames reuse the cached cell index and never read cell_variance.
	if (render_data.render_settings.sample_number == 0)
	{
		float* cell_variance = m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_VARIANCE>().get_device_pointer();
		void* compute_noise_launch_args[] = { &cell_counters,	 &cell_offsets,	 &cell_alive_list,	&pixel_indices_sorted,
											  &input_reservoirs, &cell_variance, &cell_alive_count, &num_cells };

		// Same dispatch as BuildCDFs: blocks of 1024 threads looping over the alive cells
		unsigned int compute_variance_block_count =
			get_spmis_cell_kernel_block_count(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID, 1024, num_cells);
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->launch_asynchronous(
			1024, 1, compute_variance_block_count * 1024, 1, compute_noise_launch_args, m_renderer->get_main_stream());
	}

	if (union_domain_reuse)
	{
		// Alias tables of the cells, same distributions as their CDFs built above. Blocks looping over the alive cells
		float* cell_alias_tables =
			m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_UNION_DOMAIN_CELL_ALIAS_TABLES>().get_device_pointer();
		void* build_alias_tables_launch_args[]		= { &cell_non_zero_reservoir_counters,
														&cell_offsets,
														&cell_alive_list,
														&pixel_indices_sorted,
														&input_reservoirs,
														&cell_cdfs,
														&cell_alias_tables,
														&cell_alive_count,
														&num_cells };
		unsigned int build_alias_tables_block_count = get_spmis_cell_kernel_block_count(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID,
																						RESTIR_PT_SPMIS_ALIAS_TABLE_BLOCK_SIZE, num_cells);
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID]->launch_asynchronous(
			RESTIR_PT_SPMIS_ALIAS_TABLE_BLOCK_SIZE, 1, build_alias_tables_block_count * RESTIR_PT_SPMIS_ALIAS_TABLE_BLOCK_SIZE, 1,
			build_alias_tables_launch_args, m_renderer->get_main_stream());

		// Compact records of the non-zero reservoirs of the cells, read by the neighbor selection of the union domain reuse. One thread per pixel
		unsigned int pixel_count = render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y;

		upload_render_data(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID, render_data);
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID]->launch_asynchronous(64, 1, pixel_count, 1, nullptr,
																											 m_renderer->get_main_stream());
	}
}

unsigned int ReSTIRPTRenderPass::get_spmis_cell_kernel_block_count(const std::string& kernel_id, unsigned int block_size, unsigned int num_cells)
{
	// Enough blocks to fill the GPU once: the blocks then loop over the alive cells. There are never more alive cells than cells
	unsigned int maximum_active_blocks_per_multiprocessor =
		static_cast<unsigned int>(m_kernels[kernel_id]->get_max_active_blocks_per_multiprocessor(static_cast<int>(block_size)));
	unsigned int resident_block_count =
		hippt::max(1u, maximum_active_blocks_per_multiprocessor) * static_cast<unsigned int>(m_renderer->get_device_properties().multiProcessorCount * 2);

	return hippt::max(1u, hippt::min(resident_block_count, num_cells));
}

void ReSTIRPTRenderPass::resize_cdf_luts_buffer(HIPRTRenderData& render_data, unsigned int cell_alive_count)
{
	if (m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_CDF_LUTS>().size() <
		cell_alive_count * ReSTIR_PT_SPMISCDFLUTSize)
	{
		// Resizing the buffer to get enough space for the CDF LUTs of all cells
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_CDF_LUTS>().resize(cell_alive_count *
																												   ReSTIR_PT_SPMISCDFLUTSize * 1.05f);

		// Settings the pointer in both the render data and the renderer render data to the new buffer pointer
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_luts =
			m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_CDF_LUTS>().get_device_pointer();
		m_renderer->get_render_data().render_settings.restir_pt_settings.spmis_settings.cell_cdf_luts =
			m_spmis_data.m_spmis_data.get_buffer<ReSTIRPTSPMISDataHostBuffers::RESTIR_PT_SPMIS_CELL_CDF_LUTS>().get_device_pointer();
	}
}

void ReSTIRPTRenderPass::configure_temporal_reuse_pass(HIPRTRenderData& render_data)
{
	render_data.render_settings.restir_pt_settings.temporal_pass.temporal_buffer_clear_requested = m_temporal_buffer_clear_requested;

	ReSTIRPTReservoir* temporal_input_reservoirs;
	ReSTIRPTReservoir* temporal_output_reservoirs;

	if ((render_data.render_settings.sample_number == 0 && render_data.render_settings.accumulate) || render_data.render_settings.need_to_reset)
		// First frame, using the initial candidates as the input
		temporal_input_reservoirs = render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer;
	else
		// Not the first frame, the input to the temporal pass is the output of the last frame ReSTIR
		temporal_input_reservoirs = m_last_restir_pt_output_reservoirs;

	// For the output, using whatever buffer isn't the one we're reading from (the input buffer)
	if (temporal_input_reservoirs == m_spatial_buffer.get_device_pointer())
		temporal_output_reservoirs = m_temporal_buffer.get_device_pointer();
	else
		temporal_output_reservoirs = m_spatial_buffer.get_device_pointer();

	render_data.render_settings.restir_pt_settings.temporal_pass.input_reservoirs  = temporal_input_reservoirs;
	render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs = temporal_output_reservoirs;

	m_last_temporal_output_reservoirs = temporal_output_reservoirs;
}

void ReSTIRPTRenderPass::launch_temporal_reuse_pass(HIPRTRenderData& render_data)
{
	if (render_data.render_settings.restir_pt_settings.temporal_pass.do_temporal_reuse_pass)
	{
		upload_render_data(ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID, render_data);
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, nullptr,
			m_renderer->get_main_stream());
	}
}

void ReSTIRPTRenderPass::configure_spatial_reuse_pass(HIPRTRenderData& render_data, int spatial_pass_index)
{
	render_data.render_settings.restir_pt_settings.spatial_pass.spatial_pass_index = spatial_pass_index;

	// The spatial reuse pass spatially reuse on the output of the temporal pass in the 'temporal buffer' and
	// stores in the 'spatial buffer'

	ReSTIRPTReservoir* input_reservoirs;
	ReSTIRPTReservoir* output_reservoirs;

	if (spatial_pass_index > 0)
		// If this is the second spatial reuse pass or more, reading from the output of the previous pass
		input_reservoirs = render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs;
	else
	{
		// This is the first spatial reuse pass, reading from the output of the temporal pass
		// or the initial candidates depending on whether or not we have a temporal reuse pass at all

		if (render_data.render_settings.restir_pt_settings.temporal_pass.do_temporal_reuse_pass)
			// and we have a temporal reuse pass so we're going to read from the temporal reservoirs
			input_reservoirs = render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs;
		else
			// and we do not have a temporal reuse pass so we're just going to read from the initial candidates
			input_reservoirs = m_initial_candidates_buffer.get_device_pointer();
	}

	// Outputting to whichever reservoir we're not reading from to avoid race conditions
	if (input_reservoirs == m_temporal_buffer.get_device_pointer())
		output_reservoirs = m_spatial_buffer.get_device_pointer();
	else
		output_reservoirs = m_temporal_buffer.get_device_pointer();

	render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs  = input_reservoirs;
	render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs = output_reservoirs;
}

bool ReSTIRPTRenderPass::use_spmis_union_domain_kernels(const HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) const
{
	const ReSTIRPTSPMISUnionDomainSettings& union_domain_settings		   = render_data.render_settings.restir_pt_settings.spmis_settings.union_domain;
	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers = union_domain_settings.shadow_split_buffers;

	unsigned int pixel_count = render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y;

	// The buffers of the union domain reuse are allocated by pre_render_update_spmis_buffers() and pre_render_update_spmis_union_domain_shadow_split_buffer()
	// whenever do_spmis_union_domain_reuse() is true, checking them anyway so that a frame that doesn't have them falls back to the legacy kernel
	return render_data.render_settings.restir_pt_settings.do_spmis_union_domain_reuse(
			   compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE)) &&
		   union_domain_settings.neighbor_records != nullptr && union_domain_settings.domain_cache != nullptr &&
		   shadow_split_buffers.prepare_flags != nullptr && shadow_split_buffers.pixel_count == pixel_count;
}

void ReSTIRPTRenderPass::launch_spmis_union_domain_shadow_split_kernels(HIPRTRenderData& render_data)
{
	// In this order. Every kernel is launched with one thread per pixel over 8 x 8 blocks: the global BVH traversal stack
	// has one slot per thread of such a launch
	const std::string shadow_split_kernel_ids[] = { ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID,
													ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID,
													ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID,
													ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID };
	for (const std::string& shadow_split_kernel_id : shadow_split_kernel_ids)
	{
		upload_render_data(shadow_split_kernel_id, render_data);

		m_kernels[shadow_split_kernel_id]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x,
															   m_renderer->m_render_resolution.y, nullptr, m_renderer->get_main_stream());
	}
}

void ReSTIRPTRenderPass::launch_spatial_reuse_pass(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!render_data.render_settings.restir_pt_settings.spatial_pass.do_spatial_reuse_pass)

		return;

	for (int pass_index = 0; pass_index < render_data.render_settings.restir_pt_settings.spatial_pass.number_of_passes; pass_index++)
	{
		configure_spatial_reuse_pass(render_data, pass_index);

		if (compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
			compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) ==
				RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
		{
			launch_spmis_create_reuse_cells_pass(render_data, compiler_options, render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs);

			// The union domain reuse and the legacy single reuse cell path are separate kernels: each one only carries the code of its own path
			if (use_spmis_union_domain_kernels(render_data, compiler_options))
				launch_spmis_union_domain_shadow_split_kernels(render_data);
			else
			{
				upload_render_data(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID, render_data);

				m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->launch_asynchronous(
					KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, nullptr,
					m_renderer->get_main_stream());
			}
		}
		else
		{
			upload_render_data(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID, render_data);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(
				KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, nullptr,
				m_renderer->get_main_stream());
		}
	}
}

void ReSTIRPTRenderPass::configure_shading_pass(HIPRTRenderData& render_data)
{
	if (render_data.render_settings.restir_pt_settings.spatial_pass.do_spatial_reuse_pass)
		render_data.render_settings.restir_pt_settings.restir_pt_output_reservoirs =
			render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs;
	else if (render_data.render_settings.restir_pt_settings.temporal_pass.do_temporal_reuse_pass)
		render_data.render_settings.restir_pt_settings.restir_pt_output_reservoirs =
			render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs;
	else
		render_data.render_settings.restir_pt_settings.restir_pt_output_reservoirs =
			render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer;

	m_last_restir_pt_output_reservoirs = render_data.render_settings.restir_pt_settings.restir_pt_output_reservoirs;
}

void ReSTIRPTRenderPass::launch_shading_pass(HIPRTRenderData& render_data)
{
	upload_render_data(ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID, render_data);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight,
																					m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y,
																					nullptr, m_renderer->get_main_stream());
}

bool ReSTIRPTRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))

		return false;

	// GPUKernel records every kernel launch automatically and accumulates its execution time for the frame.

	configure_initial_candidates_pass(render_data);
	launch_initial_candidates_pass(render_data);

	configure_temporal_reuse_pass(render_data);
	launch_temporal_reuse_pass(render_data);

	launch_spatial_reuse_pass(render_data, compiler_options);

	configure_shading_pass(render_data);
	launch_shading_pass(render_data);

	return true;
}

void ReSTIRPTRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))

		return;

	// If we had requested a temporal buffers clear, this has be done by this frame so we can
	// now reset the flag
	m_temporal_buffer_clear_requested = false;

	if (compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
		compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
	{
		upload_render_data(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID, render_data);
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->launch_asynchronous(
			64, 1, render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y, 1, nullptr,
			m_renderer->get_main_stream());
	}
}

void ReSTIRPTRenderPass::upload_render_data(const std::string& kernel_id, HIPRTRenderData& render_data)
{
	// Each upload uses the next host pinned slot of the ring. The asynchronous copy reads the slot when it executes on the GPU: the slot is only
	// rewritten once the copy that last read it has completed, otherwise a kernel could receive the render data of a later upload of the frame
	// (the temporal and spatial passes change their input/output reservoirs and the spatial pass index between uploads)
	unsigned int slot_index		   = m_next_render_data_upload_slot;
	m_next_render_data_upload_slot = (m_next_render_data_upload_slot + 1) % ReSTIRPTRenderPass::RENDER_DATA_UPLOAD_SLOT_COUNT;

	if (m_render_data_upload_events[slot_index] == nullptr)
		OROCHI_CHECK_ERROR(oroEventCreate(&m_render_data_upload_events[slot_index]));
	else
		OROCHI_CHECK_ERROR(oroEventSynchronize(m_render_data_upload_events[slot_index]));

	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer() + slot_index;
	*host_pinned_render_data				 = render_data;

	std::string render_data_global_name = kernel_id == ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID ||
												  kernel_id == ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID ||
												  kernel_id == ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID
											  ? "RESTIR_PT_SPMIS_RENDER_DATA"
											  : "RESTIR_PT_RENDER_DATA";
	m_kernels[kernel_id]->upload_to_module_global(render_data_global_name.c_str(), host_pinned_render_data, sizeof(HIPRTRenderData),
												  m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroEventRecord(m_render_data_upload_events[slot_index], m_renderer->get_main_stream()));
}

void ReSTIRPTRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	// Setting the pointers for use in reset_render() in the camera rays kernel
	if (is_render_pass_used(*m_compiler_options))
	{
		render_data.aux_buffers.restir_pt_reservoir_buffer_1 = m_initial_candidates_buffer.get_device_pointer();
		render_data.aux_buffers.restir_pt_reservoir_buffer_2 = m_spatial_buffer.get_device_pointer();
		render_data.aux_buffers.restir_pt_reservoir_buffer_3 = m_temporal_buffer.get_device_pointer();

		m_spmis_data.to_device(render_data);

		// The pointers of the union domain spatial reuse kernels are published here, before the frame is launched, and are never changed by launch_async():
		// every kernel of the frame sees the same pointers
		ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers =
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.shadow_split_buffers;

		if (m_spmis_union_domain_shadow_split_buffer.size() > 0)
			ReSTIRPTSPMISUnionDomainShadowSplitBuffers::layout(reinterpret_cast<unsigned char*>(m_spmis_union_domain_shadow_split_buffer.get_device_pointer()),
															   m_spmis_union_domain_shadow_split_pixel_count, shadow_split_buffers);
		else
			// Union domain reuse not used, its buffer is not allocated
			shadow_split_buffers = ReSTIRPTSPMISUnionDomainShadowSplitBuffers();

		m_wavefront_common.update_render_data_wavefront_data(render_data, true, *m_compiler_options);
		render_data.restir_pt_wavefront_data = m_restir_pt_wavefront_data.to_device();
	}
	else
	{
		// If ReSTIR PT is disabled, setting the pointers to nullptr so that the camera rays kernel
		// for example can detect that the buffers are freed and doesn't try to reset them or do
		// anything with them (which would be invalid since we would be accessing nullptr buffers)

		render_data.aux_buffers.restir_pt_reservoir_buffer_1 = nullptr;
		render_data.aux_buffers.restir_pt_reservoir_buffer_2 = nullptr;
		render_data.aux_buffers.restir_pt_reservoir_buffer_3 = nullptr;

		render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.shadow_split_buffers = ReSTIRPTSPMISUnionDomainShadowSplitBuffers();

		// The wavefront data is written by the wavefront render pass when this render pass is not used, see WavefrontRenderPass::update_render_data()
		render_data.restir_pt_wavefront_data = ReSTIRPTWavefrontDataDevice();
	}
}

void ReSTIRPTRenderPass::reset(bool reset_by_camera_movement)
{
	m_spmis_data.reset();

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))

		return;

	if (render_data.render_settings.accumulate)
		if (m_renderer->get_application_settings()->auto_sample_per_frame)
			render_data.render_settings.samples_per_frame = 1;

	render_data.render_settings.denoiser_AOV_accumulation_counter = 0;
	render_data.render_settings.sample_number					  = 0;
}

std::map<std::string, std::shared_ptr<GPUKernel>> ReSTIRPTRenderPass::get_all_kernels()
{
	if (!is_render_pass_used(*m_compiler_options))

		return {};

	// The kernels of the material families that the BSDF model does not have are not compiled
	std::map<std::string, std::shared_ptr<GPUKernel>> active_kernels = m_kernels;

	for (const std::string& kernel_id : get_wavefront_kernel_ids(KernelMaterialSpecializationCount))
		active_kernels.erase(kernel_id);

	unsigned int material_family_count = WavefrontRenderPassCommon::get_material_family_count(*m_compiler_options);
	for (const std::string& kernel_id : get_wavefront_kernel_ids(material_family_count))
		active_kernels[kernel_id] = m_kernels.at(kernel_id);

	return active_kernels;
}

bool ReSTIRPTRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return path_sampling_strategy_is_restir_pt(compiler_options.get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY));
}

WavefrontRenderPassCommon::MaterialFamilyKernels ReSTIRPTRenderPass::get_material_family_kernels(const std::string& kernel_id) const
{
	WavefrontRenderPassCommon::MaterialFamilyKernels family_kernels;
	for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
		family_kernels[family_index] = m_kernels.at(get_material_family_kernel_id(kernel_id, family_index));

	return family_kernels;
}

bool ReSTIRPTRenderPass::wavefront_kernels_ready() const
{
	if (m_wavefront_common.get_nee_deferred_mis_context_byte_size() == 0)

		return false;

	unsigned int material_family_count = WavefrontRenderPassCommon::get_material_family_count(*m_compiler_options);
	for (const std::string& kernel_id : get_wavefront_kernel_ids(material_family_count))
		if (!m_kernels.at(kernel_id)->has_been_compiled())

			return false;

	return true;
}

void ReSTIRPTRenderPass::request_temporal_bufffers_clear()
{
	m_temporal_buffer_clear_requested = true;
}

float ReSTIRPTRenderPass::get_VRAM_usage() const
{
	return get_VRAM_usage_breakdown().get_total_bytes() / 1000000.0f;
}

ReSTIRPTVRAMUsage ReSTIRPTRenderPass::get_VRAM_usage_breakdown() const
{
	ReSTIRPTVRAMUsage vram_usage;

	vram_usage.initial_candidates_reservoirs	  = m_initial_candidates_buffer.get_byte_size();
	vram_usage.initial_candidates_working_samples = m_initial_candidates_working_samples_buffer.get_byte_size();
	vram_usage.temporal_reservoirs				  = m_temporal_buffer.get_byte_size();
	vram_usage.spatial_reservoirs				  = m_spatial_buffer.get_byte_size();

	const ReSTIRPTSPMISDataHostInternal<OrochiBuffer>& spmis_data = m_spmis_data.m_spmis_data;
	vram_usage.spmis_pixel_hashes								  = spmis_data.get_buffer<RESTIR_PT_SPMIS_ALL_PIXEL_HASHES>().get_byte_size() +
									spmis_data.get_buffer<RESTIR_PT_SPMIS_ALL_PIXEL_HASHES_CHECKSUMS>().get_byte_size();
	vram_usage.spmis_pixel_cell_indices = spmis_data.get_buffer<RESTIR_PT_SPMIS_ALL_PIXEL_INDEX_IN_CELL>().get_byte_size() +
										  spmis_data.get_buffer<RESTIR_PT_SPMIS_ALL_PIXEL_REUSE_CELL_PIXEL_INDEX>().get_byte_size() +
										  spmis_data.get_buffer<RESTIR_PT_SPMIS_PIXEL_INDICES_SORTED>().get_byte_size();
	vram_usage.spmis_cell_counters_and_offsets =
		spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_COUNTERS>().get_byte_size() +
		spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_NON_ZERO_RESERVOIR_COUNTERS>().get_byte_size() +
		spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_GLOBAL_OFFSET_COUNTER>().get_byte_size() +
		spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_TOTAL_COUNT_COUNTER>().get_byte_size() +
		spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_OCCUPIED>().get_byte_size() + spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_ALIVE_LIST>().get_byte_size() +
		spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_OFFSETS>().get_byte_size() + spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_CONFIDENCE_SUMS>().get_byte_size();
	vram_usage.spmis_cell_cdfs = spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_CDFS>().get_byte_size() +
								 spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_CDF_LUTS>().get_byte_size() +
								 spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_CDF_LUT_OFFSETS>().get_byte_size();
	vram_usage.spmis_cell_variance					= spmis_data.get_buffer<RESTIR_PT_SPMIS_CELL_VARIANCE>().get_byte_size();
	vram_usage.spmis_union_domain_neighbor_records	= spmis_data.get_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORDS>().get_byte_size();
	vram_usage.spmis_union_domain_cell_alias_tables = spmis_data.get_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_CELL_ALIAS_TABLES>().get_byte_size();
	vram_usage.spmis_union_domain_cache				= spmis_data.get_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE>().get_byte_size();
	vram_usage.spmis_union_domain_shadow_split		= m_spmis_union_domain_shadow_split_buffer.get_byte_size();

	vram_usage.wavefront_staging_buffers = m_wavefront_common.get_staging_buffers_byte_size();
	vram_usage.wavefront_path_data		 = m_restir_pt_wavefront_data.get_byte_size();
	vram_usage.wavefront_primary_material_family_queues_backup =
		m_primary_material_family_indices_backup.get_byte_size() + m_primary_material_family_counts_backup.get_byte_size();

	return vram_usage;
}
