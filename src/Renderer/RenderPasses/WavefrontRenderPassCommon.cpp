/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "Renderer/RenderPasses/WavefrontRenderPassCommon.h"
#include "Renderer/RenderPasses/WavefrontRoutingConfiguration.h"
#include "Utils/Debug.h"

#include <algorithm>

WavefrontRenderPassCommon::WavefrontRenderPassCommon()
{
	// The events only order the streams, they are never used for timing
	OROCHI_CHECK_ERROR(oroEventCreateWithFlags(&m_material_family_fork_event, oroEventDisableTiming));

	for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
	{
		OROCHI_CHECK_ERROR(oroStreamCreate(&m_material_family_streams[family_index]));
		OROCHI_CHECK_ERROR(oroEventCreateWithFlags(&m_material_family_join_events[family_index], oroEventDisableTiming));
	}
}

unsigned int WavefrontRenderPassCommon::get_material_family_count(const GPUKernelCompilerOptions& compiler_options)
{
	WavefrontRoutingConfiguration configuration =
		get_wavefront_routing_configuration(compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL),
											compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR));

	return configuration.material_family_count;
}

bool WavefrontRenderPassCommon::uses_deferred_material_family_routing(const GPUKernelCompilerOptions& compiler_options)
{
	WavefrontRoutingConfiguration configuration =
		get_wavefront_routing_configuration(compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL),
											compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR));

	return configuration.route_deferred_completion_by_material_family;
}

unsigned int WavefrontRenderPassCommon::get_shadow_work_capacity(unsigned int path_capacity, const GPUKernelCompilerOptions& compiler_options)
{
	if (compiler_options.get_macro_value(GPUKernelCompilerOptions::WAVEFRONT_DEFER_NEE_SHADOW_TRAVERSAL) != KERNEL_OPTION_TRUE)
		return 0;

	int nee_estimator = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);

	int sampling_strategy				   = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY);
	unsigned int direct_light_sample_count = 1;
	if (sampling_strategy == LSS_BASE_LIGHT_TREE_ATS &&
		compiler_options.get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_DO_SPLITTING) == KERNEL_OPTION_TRUE)
		direct_light_sample_count =
			static_cast<unsigned int>(compiler_options.get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_SPLITTING_MAX_LIGHT_SAMPLES));
	else if (sampling_strategy == LSS_BASE_LIGHT_TREE_SG &&
			 compiler_options.get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_DO_SPLITTING) == KERNEL_OPTION_TRUE)
		direct_light_sample_count =
			static_cast<unsigned int>(compiler_options.get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_SPLITTING_MAX_LIGHT_SAMPLES));

	unsigned int emissive_shadow_count = 0;
	if (nee_estimator == LSS_ONE_LIGHT || nee_estimator == LSS_MIS_LIGHT_BSDF)
		emissive_shadow_count = direct_light_sample_count;
	else if (nee_estimator == LSS_RIS_BSDF_AND_LIGHT || DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(nee_estimator))
		emissive_shadow_count = 1;

	unsigned int envmap_shadow_count = 0;

	if (nee_estimator != LSS_NO_DIRECT_LIGHT_SAMPLING &&
		compiler_options.get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) != ESS_NO_SAMPLING)
	{
		envmap_shadow_count = 1;

		if (compiler_options.get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS) == KERNEL_OPTION_TRUE)
			envmap_shadow_count++;
	}

	return path_capacity * (emissive_shadow_count + envmap_shadow_count);
}

unsigned int WavefrontRenderPassCommon::get_wavefront_block_count(unsigned int path_capacity)
{
	return std::max(1u, std::min((path_capacity + 63u) / 64u, 2048u));
}

unsigned int WavefrontRenderPassCommon::get_shadow_block_count(unsigned int path_capacity)
{
	return (path_capacity + 63u) / 64u;
}

void WavefrontRenderPassCommon::set_nee_deferred_mis_context_byte_size_dirty()
{
	m_nee_deferred_mis_context_byte_size_dirty = true;
}

std::size_t WavefrontRenderPassCommon::compute_nee_deferred_mis_context_byte_size(const std::shared_ptr<GPUKernel>& context_size_kernel)
{
	if (!m_nee_deferred_mis_context_byte_size_dirty)

		return m_nee_deferred_mis_context_byte_size;

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

std::size_t WavefrontRenderPassCommon::get_nee_deferred_mis_context_byte_size() const
{
	return m_nee_deferred_mis_context_byte_size;
}

bool WavefrontRenderPassCommon::resize_staging_buffers(unsigned int path_capacity,
													   std::size_t ray_volume_state_byte_size,
													   const GPUKernelCompilerOptions& compiler_options,
													   const std::shared_ptr<GPUKernel>& nee_deferred_mis_context_size_kernel)
{
	unsigned int shadow_work_capacity			  = get_shadow_work_capacity(path_capacity, compiler_options);
	bool allocate_resolved_material_control_cache = compiler_options.get_macro_value(GPUKernelCompilerOptions::USE_MATERIAL_TEXTURES) == KERNEL_OPTION_TRUE;
	// Only the principled BSDF classifies the secondary hits into material families: the classification is cached for the shading pass
	bool allocate_current_material_classifications =
		allocate_resolved_material_control_cache && compiler_options.get_macro_value(GPUKernelCompilerOptions::BSDF_MODEL) == BSDF_PRINCIPLED;
	int nee_estimator								  = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	bool allocate_learning_to_cluster_shadow_sidecars = shadow_work_capacity > 0 && DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(nee_estimator);

	if (ray_volume_state_byte_size == 0)
		Debug::debugbreak();

	std::size_t nee_deferred_mis_context_byte_size = compute_nee_deferred_mis_context_byte_size(nee_deferred_mis_context_size_kernel);
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
						m_allocated_ray_volume_state_byte_size != ray_volume_state_byte_size ||
						m_allocated_nee_deferred_mis_context_byte_size != nee_deferred_mis_context_byte_size ||
						m_render_data_host_pinned.size() != 1 + KernelMaterialSpecializationCount ||
						m_zero_host_pinned.size() != KernelMaterialSpecializationCount ||
						m_material_family_counts_host_pinned.size() != 2 * KernelMaterialSpecializationCount;
	if (!needs_resize)

		return false;

	free_staging_buffers();

	m_wavefront_data.resize(path_capacity, shadow_work_capacity, ray_volume_state_byte_size, nee_deferred_mis_context_byte_size,
							allocate_resolved_material_control_cache, allocate_current_material_classifications, allocate_learning_to_cluster_shadow_sidecars);
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

void WavefrontRenderPassCommon::free_staging_buffers()
{
	m_wavefront_data.free();
	m_render_data_host_pinned.free_no_error();
	m_zero_host_pinned.free_no_error();
	m_material_family_counts_host_pinned.free_no_error();

	m_allocated_ray_volume_state_byte_size		   = 0;
	m_allocated_nee_deferred_mis_context_byte_size = 0;
	m_staging_buffers_allocated					   = false;
}

bool WavefrontRenderPassCommon::are_staging_buffers_allocated() const
{
	return m_staging_buffers_allocated;
}

std::size_t WavefrontRenderPassCommon::get_staging_buffers_byte_size() const
{
	return m_wavefront_data.get_byte_size();
}

WavefrontDataHost<OrochiBuffer>& WavefrontRenderPassCommon::get_wavefront_data()
{
	return m_wavefront_data;
}

WavefrontDataDevice WavefrontRenderPassCommon::get_wavefront_data_device(const GPUKernelCompilerOptions& compiler_options)
{
	WavefrontDataDevice wavefront_data_device = m_wavefront_data.to_device();

	// The family-routed deferred completion runs after TracePaths and still needs the family queues of the shaded vertex.
	// TracePaths cannot overwrite them with the families of the new hits in that case so a separate classification pass runs instead
	if (wavefront_data_device.material_family_routing_enabled && !uses_deferred_material_family_routing(compiler_options))
		wavefront_data_device.trace_classifies_material_families = 1;

	return wavefront_data_device;
}

void WavefrontRenderPassCommon::update_render_data_wavefront_data(HIPRTRenderData& render_data,
																  bool pass_used,
																  const GPUKernelCompilerOptions& compiler_options)
{
	if (pass_used && m_staging_buffers_allocated)
		render_data.wavefront_data = get_wavefront_data_device(compiler_options);
	else
		render_data.wavefront_data = WavefrontDataDevice();
}

void WavefrontRenderPassCommon::reset_material_family_counts_async(oroStream_t stream)
{
	if (!m_staging_buffers_allocated)
		return;

	m_wavefront_data.get_material_family_count_buffer().upload_data_async(m_zero_host_pinned.get_host_pinned_pointer(), stream);
}

void WavefrontRenderPassCommon::reset_queue_count_async(unsigned int queue_index, oroStream_t stream)
{
	m_wavefront_data.get_queue_count_buffer(queue_index).upload_data_async(m_zero_host_pinned.get_host_pinned_pointer(), stream);
}

void WavefrontRenderPassCommon::reset_shadow_work_count_async(oroStream_t stream)
{
	m_wavefront_data.get_shadow_work_count_buffer().upload_data_async(m_zero_host_pinned.get_host_pinned_pointer(), stream);
}

void WavefrontRenderPassCommon::reset_shadow_queue_overflow_count_async(oroStream_t stream)
{
	m_wavefront_data.get_shadow_queue_overflow_count_buffer().upload_data_async(m_zero_host_pinned.get_host_pinned_pointer(), stream);
}

bool WavefrontRenderPassCommon::prepare_material_family_render_data(const HIPRTRenderData& render_data,
																	unsigned int material_family_count,
																	unsigned int wavefront_block_count,
																	unsigned int& family_block_count)
{
	unsigned int wavefront_block_size = 64;

	// HIPRT assigns the global BVH traversal stacks statically from the block index: two family kernels running
	// concurrently would share the stacks of their blocks with the same index. Each family is given its own slice
	// of the global stack buffer and the grid of the family kernels is limited to the size of that slice
	hiprtGlobalStackBuffer global_stack_buffer = render_data.global_traversal_stack_buffer;
	unsigned int family_stack_count			   = global_stack_buffer.stackCount / material_family_count;
	family_block_count						   = std::min(wavefront_block_count, family_stack_count / wavefront_block_size);

	// The families fall back to running one after the other on the main stream with the whole stack buffer if
	// it is too small to be split between them. A single family has nothing to overlap with.
	// They also run one after the other if overlapping was turned off to measure the cost of each family kernel
	bool overlap_family_kernels =
		m_overlap_material_family_kernels && material_family_count > 1 && global_stack_buffer.stackData != nullptr && family_block_count > 0;
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

	return overlap_family_kernels;
}

void WavefrontRenderPassCommon::upload_main_stream_render_data(GPUKernel& kernel, const char* global_name, oroStream_t stream)
{
	kernel.upload_to_module_global(global_name, m_render_data_host_pinned.get_host_pinned_pointer(), sizeof(HIPRTRenderData), stream);
}

void WavefrontRenderPassCommon::upload_material_family_render_data(GPUKernel& kernel, const char* global_name, unsigned int family_index, oroStream_t stream)
{
	HIPRTRenderData* family_render_data = m_render_data_host_pinned.get_host_pinned_pointer() + 1 + family_index;

	kernel.upload_to_module_global(global_name, family_render_data, sizeof(HIPRTRenderData), stream);
}

bool WavefrontRenderPassCommon::update_material_family_count_readback(bool overlap_family_kernels)
{
	// The dominant material family only matters when the families are overlapped: it is the one that runs on the main stream
	bool read_back_material_family_counts		   = overlap_family_kernels && m_material_family_count_readback_frame_counter == 0;
	m_material_family_count_readback_frame_counter = (m_material_family_count_readback_frame_counter + 1) % MATERIAL_FAMILY_COUNT_READBACK_PERIOD;

	return read_back_material_family_counts;
}

void WavefrontRenderPassCommon::read_back_material_family_counts_async(MaterialFamilyCountReadbackSlot readback_slot, oroStream_t main_stream)
{
	unsigned int readback_slot_index = static_cast<unsigned int>(readback_slot);
	unsigned int* destination		 = m_material_family_counts_host_pinned.get_host_pinned_pointer() + readback_slot_index * KernelMaterialSpecializationCount;

	// Enqueued on the main stream after the join of the family kernels: the counters still hold the counts that the
	// family kernels have just consumed, they are only reset before the next pass that appends to the family queues
	m_wavefront_data.get_material_family_count_buffer().download_data_async(destination, main_stream);
}

unsigned int WavefrontRenderPassCommon::get_dominant_material_family(MaterialFamilyCountReadbackSlot readback_slot, unsigned int material_family_count) const
{
	unsigned int readback_slot_index = static_cast<unsigned int>(readback_slot);
	const unsigned int* counts		 = m_material_family_counts_host_pinned.get_host_pinned_pointer() + readback_slot_index * KernelMaterialSpecializationCount;
	unsigned int dominant_family	 = 0;

	// Before the first read back has completed, all the counts are 0 and the first family is picked
	for (unsigned int family_index = 1; family_index < material_family_count; family_index++)
		if (counts[family_index] > counts[dominant_family])
			dominant_family = family_index;

	return dominant_family;
}

void WavefrontRenderPassCommon::launch_material_family_kernels_async(const MaterialFamilyKernels& family_kernels,
																	 MaterialFamilyCountReadbackSlot readback_slot,
																	 void** launch_args,
																	 unsigned int material_family_count,
																	 unsigned int block_count,
																	 bool overlap,
																	 oroStream_t main_stream)
{
	if (!overlap)
	{
		for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
			family_kernels[family_index]->launch_asynchronous_3D_block_count(block_count, 1, 1, 64, 1, 1, launch_args, main_stream);

		return;
	}

	unsigned int dominant_family_index = get_dominant_material_family(readback_slot, material_family_count);

	// The family kernels must not start before the work already queued on the main stream: render data uploads,
	// queue counter resets and the kernels that produced the family queues
	OROCHI_CHECK_ERROR(oroEventRecord(m_material_family_fork_event, main_stream));

	// The dominant family is launched on the main stream: it has no cross-stream wait so the worker thread of the main stream
	// submits it ahead of time and it starts right after the kernel that produced its queue. The worker threads of the
	// other streams wait on the CPU for the fork event before they can submit their kernels (see the comment on the join below).
	// It still uses its own render data and its slice of the global stack buffer since the other families run concurrently
	family_kernels[dominant_family_index]->launch_asynchronous_3D_block_count(block_count, 1, 1, 64, 1, 1, launch_args, main_stream);

	// The families shade disjoint sets of paths and only append to the shared queues with atomics so they can run concurrently
	for (unsigned int family_index = 0; family_index < material_family_count; family_index++)
	{
		if (family_index == dominant_family_index)
			continue;

		oroStream_t family_stream = m_material_family_streams[family_index];

		OROCHI_CHECK_ERROR(oroStreamWaitEvent(family_stream, m_material_family_fork_event, 0));
		family_kernels[family_index]->launch_asynchronous_3D_block_count(block_count, 1, 1, 64, 1, 1, launch_args, family_stream);
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

bool WavefrontRenderPassCommon::get_overlap_material_family_kernels() const
{
	return m_overlap_material_family_kernels;
}

void WavefrontRenderPassCommon::set_overlap_material_family_kernels(bool overlap_material_family_kernels)
{
	m_overlap_material_family_kernels = overlap_material_family_kernels;
}
