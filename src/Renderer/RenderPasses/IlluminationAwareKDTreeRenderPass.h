/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include <cstdint>
#include <vector>

struct IlluminationAwareKDTreeVRAMUsage
{
	std::size_t nodes		= 0;
	std::size_t node_bounds = 0;
	std::size_t node_count	= 0;

	std::size_t active_guiding_nodes		= 0;
	std::size_t active_guiding_node_count	= 0;
	std::size_t needs_split					= 0;
	std::size_t light_clustering_count		= 0;
	std::size_t normal_clustering_set_count = 0;

	std::size_t current_frontier	   = 0;
	std::size_t current_frontier_count = 0;
	std::size_t next_frontier		   = 0;
	std::size_t next_frontier_count	   = 0;

	std::size_t training_samples						  = 0;
	std::size_t training_sample_count					  = 0;
	std::size_t learning_to_cluster_training_samples	  = 0;
	std::size_t learning_to_cluster_training_sample_soa	  = 0;
	std::size_t learning_to_cluster_training_sample_count = 0;

	std::size_t batch_signatures		= 0;
	std::size_t history_signatures		= 0;
	std::size_t batch_spatial_moments	= 0;
	std::size_t history_spatial_moments = 0;

	std::size_t initial_light_cut_node_indices		  = 0;
	std::size_t normal_clustering_sets				  = 0;
	std::size_t normal_face_observation_counts		  = 0;
	std::size_t light_cluster_node_indices			  = 0;
	std::size_t light_cluster_statistics			  = 0;
	std::size_t light_cluster_cdfs					  = 0;
	std::size_t light_cluster_sample_counts			  = 0;
	std::size_t light_clustering_data				  = 0;
	std::size_t representative_shading_contexts		  = 0;
	std::size_t representative_shading_context_states = 0;

	std::size_t nisml_cache							 = 0;
	std::size_t nisml_hash_keys						 = 0;
	std::size_t nisml_hash_entry_states				 = 0;
	std::size_t nisml_hash_occupied_entry_count		 = 0;
	std::size_t nisml_representative_sample_counts	 = 0;
	std::size_t nisml_representative_occupied_counts = 0;
	std::size_t nisml_representative_valid			 = 0;
	std::size_t nisml_representative_write_locks	 = 0;
	std::size_t nisml_representative_dirty			 = 0;
	std::size_t nisml_cache_ready					 = 0;
	std::size_t nisml_pending_cell_count			 = 0;

	std::size_t get_total_bytes() const
	{
		return nodes + node_bounds + node_count + active_guiding_nodes + active_guiding_node_count + needs_split + light_clustering_count +
			   normal_clustering_set_count + current_frontier + current_frontier_count + next_frontier + next_frontier_count + training_samples +
			   training_sample_count + learning_to_cluster_training_samples + learning_to_cluster_training_sample_soa +
			   learning_to_cluster_training_sample_count + batch_signatures + history_signatures + batch_spatial_moments + history_spatial_moments +
			   initial_light_cut_node_indices + normal_clustering_sets + normal_face_observation_counts + light_cluster_node_indices +
			   light_cluster_statistics + light_cluster_cdfs + light_cluster_sample_counts + light_clustering_data + representative_shading_contexts +
			   representative_shading_context_states + nisml_cache + nisml_hash_keys + nisml_hash_entry_states + nisml_hash_occupied_entry_count +
			   nisml_representative_sample_counts + nisml_representative_occupied_counts + nisml_representative_valid + nisml_representative_write_locks +
			   nisml_representative_dirty + nisml_cache_ready + nisml_pending_cell_count;
	}
};

class IlluminationAwareKDTreeRenderPass : public RenderPass
{
public:
	static const std::string ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME;
	static const std::string RESET_TREE_KERNEL_ID;
	static const std::string INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID;
	static const std::string ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID;
	static const std::string ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID;
	static const std::string ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string INITIALIZE_LIGHT_CLUSTER_Q0_KERNEL_ID;
	static const std::string REFINE_LIGHT_CLUSTERINGS_KERNEL_ID;
	static const std::string REPLAY_LIGHT_CLUSTER_STATISTICS_KERNEL_ID;
	static const std::string REPLAY_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID;
	static const std::string BUILD_LIGHT_CLUSTER_SAMPLING_CDFS_KERNEL_ID;
	static const std::string ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID;
	static const std::string RESET_BATCH_KD_TREE_STATISTICS_KERNEL_ID;
	static const std::string RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID;
	static const std::string EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID;
	static const std::string REPLAY_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID;
	static const std::string MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID;
	static const std::string PROMOTE_GUIDING_CELLS_KERNEL_ID;
	static const std::string REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string BUILD_NISML_CACHES_KERNEL_ID;

public:
	IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_all_kernels() override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;
	virtual bool pre_frame_render_update(float delta_time) override;
	virtual void pre_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	void ensure_all_lookahead_cell_levels(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options);

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;
	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

	int& get_split_iterations_per_SPP();
	bool& get_auto_split_iterations_per_SPP();
	int& get_training_sample_buffer_capacity();

	int& get_learning_to_cluster_learning_spp();
	int& get_learning_to_cluster_learning_seconds();

	int& get_nisml_representative_capacity();
	int& get_nisml_hash_table_size_mb();
	int& get_nisml_hash_normal_precision();
	unsigned int get_nisml_hash_occupied_entry_count() const;
	unsigned int get_nisml_hash_table_capacity() const;

	int& get_current_node_buffer_capacity();
	std::size_t get_current_node_count() const;
	std::size_t get_current_guiding_node_count() const;

	void mark_buffers_need_reallocation();

	bool& get_frozen_tree();

	/**
	 * If this function is updated, IlluminationAwareKDTreeVRAMUsage::get_total_bytes() should also be updated and the ImGui UI should also be updated to keep
	 * the tooltip up to date
	 */
	IlluminationAwareKDTreeVRAMUsage get_vram_usage_breakdown() const;
	std::size_t get_vram_usage_bytes() const;

private:
	void upload_render_data(const std::string& kernel_id, HIPRTRenderData& render_data);
	bool ensure_buffers_match_configuration();

	bool is_using_nisml(const GPUKernelCompilerOptions& compiler_options) const;
	bool is_using_learning_to_cluster(const GPUKernelCompilerOptions& compiler_options) const;
	void build_nisml(HIPRTRenderData& render_data);

	bool m_frozen_tree				 = false;
	bool m_buffers_need_reallocation = true;

	IlluminationAwareKDTreeUserSettings m_user_settings;
	IlluminationAwareKDTreeDataHost<OrochiBuffer> m_illumination_aware_kd_tree;
	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;

	// How many times to:
	//	for (int split; split < m_split_iterations; split++)
	//	{
	//		- Create lookahead nodes below guiding cells
	//		- Replay training samples to the newly created lookahead nodes
	//		- Split guiding cells that have been marked for splitting
	//		- Accumulate the statistics of the newly created lookahead nodes into their history
	//	}
	// per each SPP
	//
	// Higher number subdivide faster but is more expensive
	int m_split_iterations_per_SPP = 3;
	// If true, the number of split iterations per SPP will be automatically adjusted based on the current SPP for efficiency
	bool m_auto_split_iterations_per_SPP  = true;
	int m_nodes_buffer_capacity			  = IlluminationAwareKDTreeCoreDataHost<OrochiBuffer>::MAXIMUM_NUMBER_OF_NODES;
	int m_training_sample_buffer_capacity = IlluminationAwareKDTreeCoreDataHost<OrochiBuffer>::INITIAL_TRAINING_SAMPLE_BUFFER_CAPACITY;

	int m_learning_to_cluster_learning_spp		= 4096;
	int m_learning_to_cluster_learning_seconds	= 0;
	float m_learning_to_cluster_elapsed_seconds = 0.0f;

	int m_nisml_representative_capacity			   = 2;
	int m_nisml_hash_table_size_mb				   = 275;
	int m_nisml_hash_normal_precision			   = 3;
	unsigned int m_nisml_hash_occupied_entry_count = 0;

	std::size_t m_cached_current_node_count			= 1;
	std::size_t m_cached_current_guiding_node_count = 1;

	bool m_lookahead_frontier_initialized	  = false;
	bool m_current_frontier_uses_first_buffer = true;

	unsigned int m_next_creation_tag = 0;

	bool m_mark_guiding_cells_debug_check_done	  = false;
	bool m_promote_guiding_cells_debug_check_done = false;
};

#endif
