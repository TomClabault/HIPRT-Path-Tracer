/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEE_PLUS_PLUS
#define DEVICE_INCLUDES_NEE_PLUS_PLUS

#include "Device/includes/HashGrid.h"
#include "Device/includes/HashGridHash.h"

#include "HostDeviceCommon/Math.h"

/**
 * Context passed when tracing shadow rays 
 */
struct NEEPlusPlusContext
{
	float3 shaded_point;
	float3 point_on_light;

	// After passing this context to a call to 'evaluate_shadow_ray_nee_plus_plus',
	// this member will be filled with the probability that the points 'shaded_point'
	// and 'point_on_light' are mutually visible.
	//
	// If the call to 'evaluate_shadow_ray_nee_plus_plus' returns 'false' i.e. that the
	// points are mutually visibile, you will need to account this
	// 'unoccluded_probability' in the PDF of the light you sampled i.e. multiply
	// your PDF by this 'unoccluded_probability' to guarantee unbiasedness
	float unoccluded_probability = 1.0f;
	// Set this flag to true if this context should be used
	// for testing visibility probability between 'shaded_point' and the
	// envmap.
	//
	// ----- WARNING:
	// 'point_on_light' should be the normalized direction towards the envmap if this is set to true
	bool envmap = false;
};

struct NEEPlusPlusEntry
{
	AtomicType<unsigned int>* total_unoccluded_rays = nullptr;
	AtomicType<unsigned int>* total_num_rays = nullptr;

	AtomicType<unsigned int>* checksum_buffer = nullptr;
};

/**
 * Structure that contains the data for the implementation of NEE++.
 * 
 * Reference:
 * [1] [Next Event Estimation++: Visibility Mapping for Efficient Light Transport Simulation]
 */
struct NEEPlusPlusDevice
{
	// If true, the next camera rays kernel call will reset the visibility map
	bool m_reset_visibility_map = false;
	// If true, the grid visibility will be updated this frame (new visibility values will be accumulated)
	bool m_update_visibility_map = true;
	// Whether or not to do russian roulette with NEE++ on emissive lights
	bool m_enable_nee_plus_plus_RR_for_emissives = true;
	// Whether or not to do russian roulette with NEE++ on envmap samples
	bool m_enable_nee_plus_plus_RR_for_envmap = false;

	unsigned int m_total_number_of_cells = 0;
	float m_grid_cell_min_size = 0.25f;
	float m_grid_cell_target_projected_size = 25.0f;

	// After how many samples to stop updating the visibility map
	// (because it's probably converged enough)
	int m_stop_update_samples = 256;

	enum BufferNames : unsigned int
	{
		VISIBILITY_MAP_UNOCCLUDED_COUNT = 0,
		VISIBILITY_MAP_TOTAL_COUNT = 1,
	};

	// Linear buffer that is a packing of 4 buffers:
	// 
	// - 1 buffer that stores the number of rays that were
	//		computed as non-occluded from voxel to voxel in the scene.
	//
	//		For example, if 16 rays were shot from one voxel to another
	//		and 7 of these rays were found to be unoccluded, then the corresponding
	//		entry in the map will contain the value 7
	//
	//		Because the visibility map is symmetrical, this is a linear buffer that contains
	//		only half of the visibility matrix
	//
	//		For the indexing logic, (0, 0) is in the top left corner of the matrix
	//
	// - 1 buffer that is the same the same as the previous one but stores how many rays
	//		in total were traced in total from one voxel to another, not just the unoccluded ones. 
	//		In the example from above, this would contain the value 16.
	//
	//		For the indexing logic, (0, 0) is in the top left corner of the matrix
	//
	// - 2 buffers used for accumulation during the rendering process
	//		These two buffers are used for accumulation of the visibility information during the rendering
	//		For example, if we trace a shadow ray between voxel A and voxel B and that this shadow ray is
	//		occluded, we're going to have to update the visibility map with information.
	// 
	//		However, we cannot just simply update the visibility map (i.e. the 2 first buffers)
	//		during the rendering because this would	lead to concurrency issues where the map is 
	//		being updated while also being read by other threads.
	// 
	//		The race condition is fine, what's not fine is that this will vary the estimate of the occlusion probability
	//		from voxel A to voxel B and I found that this resulted in bias / non-determinism because the order in which
	//		the threads update the map now influences how the other threads are going to read the map
	//
	//		So instead we have some additional buffers here to accumulate separately and then this buffers are copied
	//		every N frames (or N seconds) to the 'true' visibility map used during rendering
	//
	// Each one these 4 buffers are of type unsigned chars, packed into 1 unsigned ints.
	// 
	// The data is stored such that the first unsigned int contains the 4 buffers at index 0 of the matrix
	// The second unsigned int contains the 4 buffers at index 1
	// ...
	NEEPlusPlusEntry m_entries_buffer;

	// If a voxel-to-voxel unocclusion probability is higher than that, the voxel will be considered unoccluded
	// and so a shadow ray will be traced. This is to avoid trusting voxel that have a low probability of
	// being unoccluded
	//
	// 0.0f basically disables NEE++ as any entry of the visibility map will require a shadow ray
	float m_confidence_threshold = 0.025f;
	float m_minimum_unoccluded_proba = 0.0f;

	// Whether or not to count the number of shadow rays actually traced vs. the number of shadow
	// queries made. This is used in 'evaluate_shadow_ray_nee_plus_plus()'
	bool do_update_shadow_rays_traced_statistics = true;

	AtomicType<unsigned long long int>* total_shadow_ray_queries = nullptr;
	AtomicType<unsigned long long int>* shadow_rays_actually_traced = nullptr;

	HIPRT_HOST_DEVICE void accumulate_visibility(bool visible, unsigned int hash_grid_index)
	{
		if (hash_grid_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			// One of the two points was outside the scene, cannot cache this
			return;
		
		if (read_buffer<BufferNames::VISIBILITY_MAP_TOTAL_COUNT>(hash_grid_index) >= 255)
			return;

		if (visible)
			increment_buffer<BufferNames::VISIBILITY_MAP_UNOCCLUDED_COUNT>(hash_grid_index, 1);
		increment_buffer<BufferNames::VISIBILITY_MAP_TOTAL_COUNT>(hash_grid_index, 1);
	}

	/**
	 * Updates the visibility map with one additional entry: whether or not the two given world points are visible
	 */
	HIPRT_HOST_DEVICE void accumulate_visibility(const NEEPlusPlusContext& context, HIPRTCamera& current_camera, bool visible)
	{
		return accumulate_visibility(visible, get_visibility_map_index<true>(context, current_camera));
	}

	/**
	 * Returns the estimated probability that a ray between the two given world points 
	 * is going to be unoccluded (i.e. the two points are mutually visible)
	 * 
	 * Returns the index in the visibility matrix of the voxel-to-voxel correspondance of the
	 * two given points. This value can then be passed as argument to 'accumulate_visibility'
	 * to save a little bit of computations (otherwise, 'accumulate_visibility' would have recomputed
	 * that value on its own even though the world points given may be the same and thus, the matrix
	 * index is the same)
	 */
	HIPRT_HOST_DEVICE float estimate_visibility_probability(const NEEPlusPlusContext& context, const HIPRTCamera& current_camera, unsigned int& out_hash_grid_index) const
	{
		out_hash_grid_index = get_visibility_map_index<true>(context, current_camera);
		if (out_hash_grid_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			// One of the two points was outside the scene, cannot read the cache for this
			// 
	 		// Returning 1.0f indicating that the two points are not occluded such that the caller
			// tests for a shadow ray
			return 1.0f;

		unsigned int total_map_count = read_buffer<BufferNames::VISIBILITY_MAP_TOTAL_COUNT>(out_hash_grid_index);
		if (total_map_count == 0)
			// No information for these two points
			// 
			// Returning 1.0f indicating that the two points are not occluded such that the caller
			// tests for a shadow ray
			return 1.0f;
		else
		{
			unsigned int unoccluded_count = read_buffer<BufferNames::VISIBILITY_MAP_UNOCCLUDED_COUNT>(out_hash_grid_index);
			
			float unoccluded_proba = unoccluded_count / static_cast<float>(total_map_count);
			if (unoccluded_proba >= m_confidence_threshold)
				return 1.0f;
			else
				return hippt::max(m_minimum_unoccluded_proba, unoccluded_proba);
		}
	}

	/**
	 * Returns the estimated probability that a ray between the two given world points
	 * is going to be unoccluded (i.e. the two points are mutually visible)
	 */
	HIPRT_HOST_DEVICE float estimate_visibility_probability(const NEEPlusPlusContext& context, const HIPRTCamera& current_camera) const
	{
		unsigned int trash_matrix_index;

		return estimate_visibility_probability(context, current_camera, trash_matrix_index);
	}

	///**
	// * Copies the accumulation buffers to the visibility map (all in the packed buffers)
	// */
	//HIPRT_HOST_DEVICE void copy_accumulation_buffers(unsigned int hash_grid_index)
	//{
	//	unsigned int accumulation_buffer = read_buffer<BufferNames::ACCUMULATION_BUFFER_UNOCCLUDED_COUNT>(hash_grid_index);
	//	unsigned int accumulation_buffer_count = read_buffer<BufferNames::ACCUMULATION_BUFFER_TOTAL_COUNT>(hash_grid_index);

	//	set_buffer<BufferNames::VISIBILITY_MAP_UNOCCLUDED_COUNT>(hash_grid_index, accumulation_buffer);
	//	set_buffer<BufferNames::VISIBILITY_MAP_TOTAL_COUNT>(hash_grid_index, accumulation_buffer_count);

	//	return;
	//}

	HIPRT_HOST_DEVICE unsigned int hash_context(const NEEPlusPlusContext& context, const HIPRTCamera& current_camera, unsigned int& out_checksum) const
	{
		float3 second_point = context.envmap ? (context.shaded_point + context.point_on_light * 1.0e20f) : context.point_on_light;

		return hash_double_position_camera(m_total_number_of_cells, context.shaded_point, second_point, current_camera, m_grid_cell_target_projected_size, m_grid_cell_min_size, out_checksum);
	}

	template <bool isInsertion = false>
	HIPRT_HOST_DEVICE unsigned int get_visibility_map_index(const NEEPlusPlusContext& context, const HIPRTCamera& current_camera) const
	{
		unsigned int checksum;
		unsigned int hash_grid_index = hash_context(context, current_camera, checksum);
		if (!HashGrid::resolve_collision<NEEPlusPlus_LinearProbingSteps, isInsertion>(m_entries_buffer.checksum_buffer, m_total_number_of_cells, hash_grid_index, checksum))
		{
			return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
		}

		return hash_grid_index;
	}

	// TODO compare with the alpha learning rate and the ground truth to see the behavior of a single float buffer
	// TODO see if capping at 255 / 65535 is enough
private:
	/**
	 * Returns the value packed in the buffer at the given visibility matrix index and with the given
	 * buffer name from the BufferNames enum
	 */
	template <unsigned int bufferName>
	HIPRT_HOST_DEVICE unsigned int read_buffer(unsigned int hash_grid_index) const
	{
		if constexpr (bufferName == 0)
			return m_entries_buffer.total_unoccluded_rays[hash_grid_index];
		else if constexpr (bufferName == 1)
			return m_entries_buffer.total_num_rays[hash_grid_index];
	}

	/**
	 * Increments the packed value in the packed buffer 'bufferName' at the given matrix index
	 * 
	 * There is no protection against overflows in this function
	 */
	template <unsigned int bufferName>
	HIPRT_HOST_DEVICE void increment_buffer(unsigned int hash_grid_index, unsigned int value)
	{
		if constexpr (bufferName == 0)
			hippt::atomic_fetch_add(&m_entries_buffer.total_unoccluded_rays[hash_grid_index], value);
		if constexpr (bufferName == 1)
			hippt::atomic_fetch_add(&m_entries_buffer.total_num_rays[hash_grid_index], value);
	}

	/**
	 * Sets the value in one of the packed buffer
	 * 
	 * WARNING:
	 * This function is non-atomic
	 */
	template <unsigned int bufferName>
	HIPRT_HOST_DEVICE void set_buffer(unsigned int hash_grid_index, unsigned int value)
	{
		if constexpr (bufferName == 0)
			m_entries_buffer.total_unoccluded_rays[hash_grid_index] = value;
		if constexpr (bufferName == 1)
			m_entries_buffer.total_num_rays[hash_grid_index] = value;
	}
};

#endif
