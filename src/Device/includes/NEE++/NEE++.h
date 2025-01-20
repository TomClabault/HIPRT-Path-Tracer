/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEE_PLUS_PLUS
#define DEVICE_INCLUDES_NEE_PLUS_PLUS

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

/**
 * Structure that contains the data for the implementation of NEE++.
 * 
 * Reference:
 * [1] [Next Event Estimation++: Visibility Mapping for Efficient Light Transport Simulation]
 */
struct NEEPlusPlusDevice
{
	static constexpr int NEE_PLUS_PLUS_DEFAULT_GRID_SIZE = 16;

	// If true, the next camera rays kernel call will reset the visibility map
	bool reset_visibility_map = false;
	// If true, the grid visibility will be updated this frame (new visibility values will be accumulated)
	bool update_visibility_map = true;
	// Whether or not to do russian roulette with NEE++ on emissive lights
	bool enable_nee_plus_plus_RR_for_emissives = true;
	// Whether or not to do russian roulette with NEE++ on envmap samples
	bool enable_nee_plus_plus_RR_for_envmap = true;

	int3 grid_dimensions = make_int3(NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEE_PLUS_PLUS_DEFAULT_GRID_SIZE);

	// Bottom left corner of the 3D grid in world space
	float3 grid_min_point = make_float3(0.0f, 0.0f, 0.0f);
	//  corner of the 3D grid in world space
	float3 grid_max_point = make_float3(0.0f, 0.0f, 0.0f);

	enum BufferNames : unsigned int
	{
		VISIBILITY_MAP = 0,
		VISIBILITY_MAP_COUNT = 1,
		ACCUMULATION_BUFFER = 2,
		ACCUMULATION_BUFFER_COUNT = 3,
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
	AtomicType<unsigned int>* packed_buffers = nullptr;

	// TODO deallocate accumulation buffers if not updating the vis map anymore

	// If a voxel-to-voxel unocclusion probability is higher than that, the voxel will be considered unoccluded
	// and so a shadow ray will be traced. This is to avoid trusting voxel that have a low probability of
	// being unoccluded
	//
	// 0.0f basically disables NEE++ as any entry of the visibility map will require a shadow ray
	float confidence_threshold = 0.025f;

	// Whether or not to count the number of shadow rays actually traced vs. the number of shadow
	// queries made. This is used in 'evaluate_shadow_ray_nee_plus_plus()'
	bool do_update_shadow_rays_traced_statistics = true;

	AtomicType<unsigned int>* total_shadow_ray_queries = nullptr;
	AtomicType<unsigned int>* shadow_rays_actually_traced = nullptr;

	HIPRT_HOST_DEVICE void accumulate_visibility(bool visible, int matrix_index)
	{
		if (matrix_index == -1)
			// One of the two points was outside the scene, cannot cache this
			return;

		if (read_buffer<BufferNames::ACCUMULATION_BUFFER_COUNT>(matrix_index) > 220)
			// We're at the limit of unsigned chars, cannot accumulate anymore
			return;

		if (visible)
			increment_buffer<BufferNames::ACCUMULATION_BUFFER>(matrix_index, 1);
		increment_buffer<BufferNames::ACCUMULATION_BUFFER_COUNT>(matrix_index, 1);
	}

	/**
	 * Updates the visibility map with one additional entry: whether or not the two given world points are visible
	 */
	HIPRT_HOST_DEVICE void accumulate_visibility(const NEEPlusPlusContext& context, bool visible)
	{
		return accumulate_visibility(visible, get_visibility_map_index(context));
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
	HIPRT_HOST_DEVICE float estimate_visibility_probability(const NEEPlusPlusContext& context, int& out_matrix_index) const
	{
		out_matrix_index = get_visibility_map_index(context);
		if (out_matrix_index == -1)
			// One of the two points was outside the scene, cannot read the cache for this
			// 
	 		// Returning 1.0f indicating that the two points are not occluded such that the caller
			// tests for a shadow ray
			return 1.0f;

		unsigned char map_count = read_buffer<BufferNames::VISIBILITY_MAP_COUNT>(out_matrix_index);
		if (map_count == 0)
			// No information for these two points
			// 
			// Returning 1.0f indicating that the two points are not occluded such that the caller
			// tests for a shadow ray
			return 1.0f;
		else
		{
			float unoccluded_proba = read_buffer<BufferNames::VISIBILITY_MAP>(out_matrix_index) / static_cast<float>(map_count);
			if (unoccluded_proba >= confidence_threshold)
				return 1.0f;
			else
				return unoccluded_proba;
		}
	}

	/**
	 * Returns the estimated probability that a ray between the two given world points
	 * is going to be unoccluded (i.e. the two points are mutually visible)
	 */
	HIPRT_HOST_DEVICE float estimate_visibility_probability(const NEEPlusPlusContext& context) const
	{
		int trash_matrix_index;
		return estimate_visibility_probability(context, trash_matrix_index);
	}

	HIPRT_HOST_DEVICE unsigned int get_visibility_matrix_element_count() const
	{
		unsigned int grid_elements_count = grid_dimensions.x * grid_dimensions.y * grid_dimensions.z;
		unsigned half_matrix_size = grid_elements_count * (grid_elements_count + 1) / 2.0f;

		return half_matrix_size;
	}

	/**
	 * Copies the accumulation buffers to the visibility map (all in the packed buffers)
	 */
	HIPRT_HOST_DEVICE void copy_accumulation_buffers(unsigned int matrix_index)
	{
		unsigned char accumulation_buffer = read_buffer<BufferNames::ACCUMULATION_BUFFER>(matrix_index);
		unsigned char accumulation_buffer_count = read_buffer<BufferNames::ACCUMULATION_BUFFER_COUNT>(matrix_index);

		set_buffer<BufferNames::VISIBILITY_MAP>(matrix_index, accumulation_buffer);
		set_buffer<BufferNames::VISIBILITY_MAP_COUNT>(matrix_index, accumulation_buffer_count);
		return;
	}

	// TODO compare with the alpha learning rate and the ground truth to see the behavior of a single float buffer
	// TODO see if capping at 255 / 65535 is enough
private:
	/**
	 * Returns the value packed in the buffer at the given visibility matrix index and with the given
	 * buffer name from the BufferNames enum
	 */
	template <unsigned int bufferName>
	HIPRT_HOST_DEVICE unsigned char read_buffer(int matrix_index) const
	{
		return (packed_buffers[matrix_index] >> (bufferName * 8)) & 0xFF;
	}

	/**
	 * Increments the packed value in the packed buffer 'bufferName' at the given matrix index
	 * There is no protection against overflows in this function
	 */
	template <unsigned int bufferName>
	HIPRT_HOST_DEVICE void increment_buffer(int matrix_index, unsigned char value)
	{
		hippt::atomic_fetch_add(&packed_buffers[matrix_index], static_cast<unsigned int>(value) << (8 * bufferName));
	}

	/**
	 * Sets the value in one of the packed buffer
	 * 
	 * WARNING:
	 * This function is non-atomic
	 */
	template <unsigned int bufferName>
	HIPRT_HOST_DEVICE void set_buffer(int matrix_index, unsigned char value)
	{
		// Clearing
		packed_buffers[matrix_index] &= ~(0x000000FF << (bufferName * 8));

		// Setting
		packed_buffers[matrix_index] |= value << (bufferName * 8);
	}

	/**
	 * Returns the index of the voxel of the given position in [grid_dimensions.x - 1, grid_dimensions.y - 1, grid_dimensions.z - 1]
	 */
	HIPRT_HOST_DEVICE int3 get_voxel_3D_index(float3 position) const
	{
		float3 position_grid_space = position - grid_min_point;
		float3 voxel_index_float = position_grid_space / (grid_max_point - grid_min_point);

		if (voxel_index_float.x > 1.0f || voxel_index_float.y > 1.0f || voxel_index_float.z > 1.0f)
			// The point is outside the grid
			return make_int3(-1, -1, -1);

		voxel_index_float = hippt::min(0.99999f, voxel_index_float);

		int3 voxel_index_int = make_int3(static_cast<int>(voxel_index_float.x * grid_dimensions.x),
			static_cast<int>(voxel_index_float.y * grid_dimensions.y),
			static_cast<int>(voxel_index_float.z * grid_dimensions.z));

		return voxel_index_int;
	}

	/**
	 * This function returns the point at the very boundary of the voxel grid (i.e. a point in the envmap layer)
	 * given a point and a direction.
	 * 
	 * This is basically an intersection test between a ray(shaded_point, direction) and the envmap boundary planes
	 */
	HIPRT_HOST_DEVICE float3 compute_voxel_grid_exit_point_from_direction(float3 shaded_point, float3 direction) const
	{
		// Finding the times in X/Y/Z for the ray to escape the voxel grid (the voxel grid accounting
		// for the envmap layer, hence why we add 'one_voxel_size' because that extra voxel layer is out
		// of all the geometry of the scene so it doesn't contain the envmap layer)
		float tx = 1.0e35f, ty = 1.0e35f, tz = 1.0e35f;
		if (hippt::abs(direction.x) > 1.0e-5f)
			tx = (direction.x < 0.0f ? (grid_min_point.x - shaded_point.x) : (grid_max_point.x - shaded_point.x)) / direction.x;
		if (hippt::abs(direction.y) > 1.0e-5f)
			ty = (direction.y < 0.0f ? (grid_min_point.y - shaded_point.y) : (grid_max_point.y - shaded_point.y)) / direction.y;
		if (hippt::abs(direction.z) > 1.0e-5f)
			tz = (direction.z < 0.0f ? (grid_min_point.z - shaded_point.z) : (grid_max_point.z - shaded_point.z)) / direction.z;

		float min_t = hippt::min(tx, hippt::min(ty, tz));
		
		// Subtracting 1.0e-3f to be sure to avoid precision issues 
		float3 exit_point = shaded_point + direction * (min_t - 1.0e-3f);

		return exit_point;
	}

	HIPRT_HOST_DEVICE int3 get_voxel_3D_index_envmap(float3 shaded_point, float3 direction) const
	{
		// The visibility for the envmap is cached in the very outer layer of the voxel grid
		// 
		// This is a layer that is outside of any geometry of the scene (there is no geometry in
		// this "layer" of the voxel grid)
		//
		// The goal here is to find which voxel of that outer layer the direction is pointing to
		// and this will give us the envmap voxel for that direction.
		//
		// This is the same as looking for the voxel at the boundaries of the grid when a ray
		// with direction 'direction' tries to escape the grid
		
		float3 exit_point = compute_voxel_grid_exit_point_from_direction(shaded_point, direction);

		// Once that we have the point in the envmap layer of the voxel grid corresponding to
		// the direction, we just need its index
		return get_voxel_3D_index(exit_point);
	}

	HIPRT_HOST_DEVICE int get_visibility_map_index(const NEEPlusPlusContext& context) const
	{
		int3 shaded_point_voxel_3D_index = get_voxel_3D_index(context.shaded_point);
		int3 second_pos_voxel_3D_index;

		if (context.envmap)
			second_pos_voxel_3D_index = get_voxel_3D_index_envmap(context.shaded_point, context.point_on_light);
		else
			second_pos_voxel_3D_index = get_voxel_3D_index(context.point_on_light);

		if (shaded_point_voxel_3D_index.x == -1 || second_pos_voxel_3D_index.x == -1)
			// One of the two points is outside the scene, cannot cache this
			return -1;

		int first_voxel_index = shaded_point_voxel_3D_index.x + shaded_point_voxel_3D_index.y * grid_dimensions.x + shaded_point_voxel_3D_index.z * grid_dimensions.y * grid_dimensions.x;
		int second_voxel_index = second_pos_voxel_3D_index.x + second_pos_voxel_3D_index.y * grid_dimensions.x + second_pos_voxel_3D_index.z * grid_dimensions.y * grid_dimensions.x;

		// Just renaming
		int column = first_voxel_index;
		int row = second_voxel_index;

		if (column > row)
		{
			// We're past the diagonal, flipping to bring it back in the lower left triangle of the matrix
			int temp = column;
			column = row;
			row = temp;
		}

		// Because the visibility matrix is symmetrical, we're only using half of it to save half the VRAM
		// Thus we need to find the index in the lower half of the visiblity matrix
		//
		// We know that our matrix is N*N
		// Because we only consider the lower half (lower triangle) of the matrix:
		// 
		// - The first row contains  1 element   : (0, 0)
		// - The second row contains 2 elements : (1, 0), (1, 1)
		// - The third row contains  3 elements  : (2, 0), (2, 1), (2, 2)
		//
		// The starting index of a row R is then given by the sum of the number of elements
		// of all rows coming before row N. So for row 3, that would be 1 + 2 + 3 = 6
		//
		// In general, this gives us starting_index = N * (N + 1) / 2
	
		int starting_index = row * (row + 1) / 2.0f;
		// We then just need to index our item inside that row
		int final_index = starting_index + column;

		return final_index;
	}
};

#endif
