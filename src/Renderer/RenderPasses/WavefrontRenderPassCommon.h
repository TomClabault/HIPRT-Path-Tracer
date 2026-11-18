/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_WAVEFRONT_RENDER_PASS_COMMON_H
#define RENDERER_WAVEFRONT_RENDER_PASS_COMMON_H

#include "Compiler/GPUKernel.h"
#include "Compiler/GPUKernelCompilerOptions.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/RenderData.h"
#include "Renderer/CPUGPUCommonDataStructures/WavefrontDataHost.h"

#include <array>
#include <cstddef>
#include <memory>

/**
 * Host machinery shared by the render passes that run wavefront kernels on the wavefront data:
 *
 * - the wavefront staging buffers and their sizes,
 * - the material family queues filled by the camera rays pass and their counters,
 * - the streams and events that overlap the kernels of the different material families,
 * - the per-family render data with their slice of the global BVH traversal stack buffer.
 *
 * The kernels themselves are owned by the render pass and are given to the functions of this class as parameters
 */
class WavefrontRenderPassCommon
{
public:
	// Which part of the read back material family counts is used to choose the dominant family of a pass
	enum class MaterialFamilyCountReadbackSlot
	{
		// The counts of the family queues filled with the primary hits
		PRIMARY_HITS = 0,
		// The counts of the family queues filled with the hits of the first secondary bounce
		SECONDARY_HITS = 1
	};

	// One kernel per material family, indexed by the KernelMaterialSpecialization of the family
	using MaterialFamilyKernels = std::array<std::shared_ptr<GPUKernel>, KernelMaterialSpecializationCount>;

	WavefrontRenderPassCommon();

	static unsigned int get_material_family_count(const GPUKernelCompilerOptions& compiler_options);
	static bool uses_deferred_material_family_routing(const GPUKernelCompilerOptions& compiler_options);
	static unsigned int get_shadow_work_capacity(unsigned int path_capacity, const GPUKernelCompilerOptions& compiler_options);

	/**
	 * Number of blocks of 64 threads of the wavefront kernels that loop over the queues of 'path_capacity' paths
	 */
	static unsigned int get_wavefront_block_count(unsigned int path_capacity);
	/**
	 * Number of blocks of 64 threads of the kernels that process one shadow work item per thread
	 */
	static unsigned int get_shadow_block_count(unsigned int path_capacity);

	/**
	 * Must be called whenever the options of the kernels change or when the kernel that computes
	 * the size of the NEE deferred MIS context is compiled: the size depends on the compile options
	 */
	void set_nee_deferred_mis_context_byte_size_dirty();
	/**
	 * Returns the size of the NEE deferred MIS context of one path, computed on the GPU by 'context_size_kernel'
	 * if the options changed since the last call. Returns 0 if 'context_size_kernel' has not been compiled yet
	 */
	std::size_t compute_nee_deferred_mis_context_byte_size(const std::shared_ptr<GPUKernel>& context_size_kernel);
	/**
	 * Returns the size last computed by compute_nee_deferred_mis_context_byte_size(), without launching anything
	 */
	std::size_t get_nee_deferred_mis_context_byte_size() const;

	/**
	 * Allocates or resizes the wavefront staging buffers for 'path_capacity' paths.
	 * Frees them if the size of the NEE deferred MIS context is not known yet.
	 *
	 * Returns true if the buffers were allocated, resized or freed: the render data has to be updated in that case
	 */
	bool resize_staging_buffers(unsigned int path_capacity,
								std::size_t ray_volume_state_byte_size,
								const GPUKernelCompilerOptions& compiler_options,
								const std::shared_ptr<GPUKernel>& nee_deferred_mis_context_size_kernel);
	void free_staging_buffers();
	bool are_staging_buffers_allocated() const;
	std::size_t get_staging_buffers_byte_size() const;

	WavefrontDataHost<OrochiBuffer>& get_wavefront_data();
	WavefrontDataDevice get_wavefront_data_device(const GPUKernelCompilerOptions& compiler_options);
	/**
	 * Writes the wavefront data of this object to 'render_data' if 'pass_used' and the staging buffers are allocated,
	 * the default wavefront data otherwise
	 */
	void update_render_data_wavefront_data(HIPRTRenderData& render_data, bool pass_used, const GPUKernelCompilerOptions& compiler_options);

	void reset_material_family_counts_async(oroStream_t stream);
	void reset_queue_count_async(unsigned int queue_index, oroStream_t stream);
	void reset_shadow_work_count_async(oroStream_t stream);
	void reset_shadow_queue_overflow_count_async(oroStream_t stream);

	/**
	 * Copies 'render_data' to the pinned render data of the main stream kernels and of each material family.
	 *
	 * If the material family kernels can overlap, each family gets its own slice of the global BVH traversal stack buffer.
	 * 'family_block_count' is set to the grid size of the family kernels.
	 *
	 * Returns whether the material family kernels overlap
	 */
	bool prepare_material_family_render_data(const HIPRTRenderData& render_data,
											 unsigned int material_family_count,
											 unsigned int wavefront_block_count,
											 unsigned int& family_block_count);
	/**
	 * Uploads the render data of the main stream kernels prepared by prepare_material_family_render_data() to the
	 * '__constant__' 'global_name' of 'kernel'
	 */
	void upload_main_stream_render_data(GPUKernel& kernel, const char* global_name, oroStream_t stream);
	/**
	 * Uploads the render data of the material family 'family_index' prepared by prepare_material_family_render_data()
	 * to the '__constant__' 'global_name' of 'kernel'
	 */
	void upload_material_family_render_data(GPUKernel& kernel, const char* global_name, unsigned int family_index, oroStream_t stream);

	/**
	 * Returns whether the material family counts must be read back this frame and advances the read back frame counter
	 */
	bool update_material_family_count_readback(bool overlap_family_kernels);
	void read_back_material_family_counts_async(MaterialFamilyCountReadbackSlot readback_slot, oroStream_t main_stream);
	unsigned int get_dominant_material_family(MaterialFamilyCountReadbackSlot readback_slot, unsigned int material_family_count) const;

	/**
	 * Launches the kernel of each of the 'material_family_count' material families with 'block_count' blocks of 64 threads.
	 *
	 * If 'overlap', the dominant family of 'readback_slot' runs on 'main_stream' and the other families on their own
	 * streams, joined back on 'main_stream'. Otherwise, the families run one after the other on 'main_stream'
	 */
	void launch_material_family_kernels_async(const MaterialFamilyKernels& family_kernels,
											  MaterialFamilyCountReadbackSlot readback_slot,
											  void** launch_args,
											  unsigned int material_family_count,
											  unsigned int block_count,
											  bool overlap,
											  oroStream_t main_stream);

	bool get_overlap_material_family_kernels() const;
	void set_overlap_material_family_kernels(bool overlap_material_family_kernels);

private:
	bool m_staging_buffers_allocated						   = false;
	std::size_t m_allocated_ray_volume_state_byte_size		   = 0;
	std::size_t m_allocated_nee_deferred_mis_context_byte_size = 0;
	std::size_t m_nee_deferred_mis_context_byte_size		   = 0;
	bool m_nee_deferred_mis_context_byte_size_dirty			   = true;

	WavefrontDataHost<OrochiBuffer> m_wavefront_data;
	// Entry 0 is the render data of the kernels that run on the main stream.
	// Entry 1 + family_index is the render data of the kernels of that material family: each family gets
	// its own slice of the global BVH traversal stack buffer so that the families can run concurrently
	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;
	OrochiBuffer<unsigned int> m_zero_host_pinned;

	// The family counters read back from the GPU to find the dominant material family of each shading pass.
	// Entries [0, KernelMaterialSpecializationCount) are the counts of the primary shading pass, the following entries
	// are the counts of the first secondary shading pass.
	// The copies are asynchronous and never waited on: the counts are only used to choose which family kernel runs
	// on the main stream so values that are stale or still being written cannot affect the image
	OrochiBuffer<unsigned int> m_material_family_counts_host_pinned;
	// The counts are read back once every that many frames: the dominant family rarely changes from one sample to the next
	// and each read back adds a copy command on the main stream
	static constexpr unsigned int MATERIAL_FAMILY_COUNT_READBACK_PERIOD = 16;
	unsigned int m_material_family_count_readback_frame_counter			= 0;

	// The shading and deferred completion kernels of the different material families run concurrently on their own streams.
	// On a RX 7900 XTX (white room, 1280x720) this went from 73 to 80 samples/s compared to launching them one after the other
	// on the main stream, with one HIP hardware queue per stream (see HIPRTOrochiCtx::set_HIP_hardware_queue_count())
	//
	// The dominant family (the one with the most paths) runs on the main stream so that it starts right after the kernel
	// that produced its queue: a kernel behind a cross-stream wait is only submitted once the worker thread of its stream
	// has seen the wait complete on the CPU. Only the other families are forked to their own streams.
	//
	// If false, the families run one after the other on the main stream instead. This is slower but the execution time of each
	// family kernel is then its real cost: when overlapped, it also includes the time spent waiting for the other families
	bool m_overlap_material_family_kernels = true;

	// One stream per material family for the overlapped family kernels
	oroStream_t m_material_family_streams[KernelMaterialSpecializationCount] = {};
	// Recorded on the main stream before the family kernels so that the family streams wait for the previous work of the main stream
	oroEvent_t m_material_family_fork_event = nullptr;
	// Recorded on each forked family stream after its kernel so that the main stream waits for all the families before continuing
	oroEvent_t m_material_family_join_events[KernelMaterialSpecializationCount] = {};
};

#endif // #ifndef RENDERER_WAVEFRONT_RENDER_PASS_COMMON_H
