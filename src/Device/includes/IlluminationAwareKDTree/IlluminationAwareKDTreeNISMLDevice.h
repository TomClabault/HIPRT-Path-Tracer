/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NISML_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NISML_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNISMLCache.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"

#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "HostDeviceCommon/Xorshift.h"

struct IlluminationAwareKDTreeNISMLDevice
{
	HIPRT_DEVICE void append_nisml_representative(unsigned int node_index,
												  unsigned int node_capacity,
												  float3_t position,
												  float3_t view_direction,
												  float3_t normal,
												  float sg_specular_weight,
												  float alpha_x,
												  float alpha_y,
												  Xorshift32Generator& random_number_generator)
	{
		if (node_index >= node_capacity)
			return;

		unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(normal);
		unsigned int cache_index = get_nisml_cache_index(node_index, normal_face);

		unsigned int sample_index	   = hippt::atomic_fetch_add(&nisml_representative_sample_counts[cache_index], 1u);
		unsigned int replacement_index = sample_index;
		bool is_replacement			   = false;

		if (sample_index >= nisml_representative_capacity)
		{
#if IlluminationAwareKDTreeNISMLNormalDiversityHeuristic == ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_DIVERSITY_HEURISTIC_LOCAL
			// A single representative cannot provide normal diversity. Keep its first sample stable instead of doing unnecessary work.
			if (nisml_representative_capacity <= 1u)
				return;

			float candidate_closest_similarity					= -1.0f;
			unsigned int candidate_closest_representative_index = nisml_representative_capacity;
			for (unsigned int representative_index = 0; representative_index < nisml_representative_capacity; representative_index++)
			{
				unsigned int flat_representative_index = get_nisml_representative_index(cache_index, representative_index);
				if (nisml_representative_valid[flat_representative_index] == 0)
					continue;

				float normal_similarity = hippt::dot(normal, nisml_cache[flat_representative_index].representative_normal);
				if (normal_similarity > candidate_closest_similarity)
				{
					candidate_closest_similarity		   = normal_similarity;
					candidate_closest_representative_index = representative_index;
				}
			}

			if (candidate_closest_representative_index >= nisml_representative_capacity)
				return;

			unsigned int selected_representative_flat_index	 = get_nisml_representative_index(cache_index, candidate_closest_representative_index);
			float selected_representative_closest_similarity = -1.0f;
			for (unsigned int representative_index = 0; representative_index < nisml_representative_capacity; representative_index++)
			{
				if (representative_index == candidate_closest_representative_index)
					continue;

				unsigned int flat_representative_index = get_nisml_representative_index(cache_index, representative_index);
				if (nisml_representative_valid[flat_representative_index] == 0)
					continue;

				float normal_similarity					   = hippt::dot(nisml_cache[selected_representative_flat_index].representative_normal,
																		nisml_cache[flat_representative_index].representative_normal);
				selected_representative_closest_similarity = hippt::max(selected_representative_closest_similarity, normal_similarity);
			}

			// Replace a representative only when the candidate is more separated from the selected representative's closest neighbor.
			if (candidate_closest_similarity >= selected_representative_closest_similarity)
				return;

			replacement_index = candidate_closest_representative_index;
#elif IlluminationAwareKDTreeNISMLNormalDiversityHeuristic == ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_DIVERSITY_HEURISTIC_FULL
			// A single representative cannot provide normal diversity. Keep its first sample stable instead of doing unnecessary work.
			if (nisml_representative_capacity <= 1u)
				return;

			unsigned int valid_representative_count = 0;
			for (unsigned int representative_index = 0; representative_index < nisml_representative_capacity; representative_index++)
			{
				unsigned int flat_representative_index = get_nisml_representative_index(cache_index, representative_index);
				valid_representative_count += nisml_representative_valid[flat_representative_index] != 0 ? 1u : 0u;
			}
			if (valid_representative_count < nisml_representative_capacity)
				return;

			// Evaluate every possible replacement and minimize the worst pairwise normal similarity in the resulting set.
			float best_resulting_maximum_similarity = 2.0f;
			unsigned int best_replacement_index		= nisml_representative_capacity;
			for (unsigned int candidate_replacement_index = 0; candidate_replacement_index < nisml_representative_capacity; candidate_replacement_index++)
			{
				unsigned int candidate_replacement_flat_index = get_nisml_representative_index(cache_index, candidate_replacement_index);
				if (nisml_representative_valid[candidate_replacement_flat_index] == 0)
					continue;

				float resulting_maximum_similarity = -1.0f;
				for (unsigned int first_representative_index = 0; first_representative_index < nisml_representative_capacity; first_representative_index++)
				{
					if (first_representative_index == candidate_replacement_index)
						continue;

					unsigned int first_representative_flat_index = get_nisml_representative_index(cache_index, first_representative_index);
					if (nisml_representative_valid[first_representative_flat_index] == 0)
						continue;

					float candidate_similarity	 = hippt::dot(normal, nisml_cache[first_representative_flat_index].representative_normal);
					resulting_maximum_similarity = hippt::max(resulting_maximum_similarity, candidate_similarity);

					for (unsigned int second_representative_index = first_representative_index + 1; second_representative_index < nisml_representative_capacity;
						 second_representative_index++)
					{
						if (second_representative_index == candidate_replacement_index)
							continue;

						unsigned int second_representative_flat_index = get_nisml_representative_index(cache_index, second_representative_index);
						if (nisml_representative_valid[second_representative_flat_index] == 0)
							continue;

						float representative_similarity = hippt::dot(nisml_cache[first_representative_flat_index].representative_normal,
																	 nisml_cache[second_representative_flat_index].representative_normal);
						resulting_maximum_similarity	= hippt::max(resulting_maximum_similarity, representative_similarity);
					}
				}

				if (resulting_maximum_similarity < best_resulting_maximum_similarity)
				{
					best_resulting_maximum_similarity = resulting_maximum_similarity;
					best_replacement_index			  = candidate_replacement_index;
				}
			}

			if (best_replacement_index >= nisml_representative_capacity)
				return;

			replacement_index = best_replacement_index;
#else
			replacement_index = static_cast<unsigned int>(random_number_generator.random_index(static_cast<int>(sample_index + 1u)));
			if (replacement_index >= nisml_representative_capacity)
				return;
#endif
			is_replacement = true;
		}

		unsigned int representative_index = get_nisml_representative_index(cache_index, replacement_index);
		if (is_replacement)
		{
			if (hippt::atomic_compare_exchange(&nisml_representative_write_locks[representative_index], 0u, 1u) != 0u)
				// Simple fail fast approach for representative replacements: if the selected representative slot is already being replaced, skip this
				// replacement. This avoids potential deadlocks and does not affect the guaranteed initial representative slots.
				return;

			// An initial representative may have claimed this slot but not published its data yet. Leave that slot to the initial representative.
			if (nisml_representative_valid[representative_index] == 0)
			{
				hippt::atomic_exchange(&nisml_representative_write_locks[representative_index], 0u);
				return;
			}
		}

		IlluminationAwareKDTreeNISMLCache& cache = nisml_cache[representative_index];
		cache.representative_position			 = position;
		cache.representative_view_direction		 = view_direction;
		cache.representative_normal				 = normal;
		cache.representative_sg_specular_weight	 = sg_specular_weight;
		cache.representative_alpha_x			 = alpha_x;
		cache.representative_alpha_y			 = alpha_y;

		if (nisml_representative_valid[representative_index] == 0)
		{
			nisml_representative_valid[representative_index] = 1;
			hippt::atomic_fetch_add(&nisml_representative_occupied_counts[cache_index], 1u);
		}

		if (hippt::atomic_compare_exchange(&nisml_representative_dirty[cache_index], static_cast<unsigned char>(0), static_cast<unsigned char>(1)) == 0)
		{
			hippt::atomic_fetch_add(nisml_pending_cell_count, 1u);
		}

		if (is_replacement)
			hippt::atomic_exchange(&nisml_representative_write_locks[representative_index], 0u);
	}

	HIPRT_DEVICE unsigned int get_nisml_cache_index(unsigned int node_index, unsigned int normal_face) const
	{
		return node_index * ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT + normal_face;
	}

	HIPRT_DEVICE unsigned int get_nisml_representative_index(unsigned int cache_index, unsigned int representative_index) const
	{
		return cache_index * nisml_representative_capacity + representative_index;
	}

	HIPRT_DEVICE void initialize_nisml_cache_for_guiding_cell(unsigned int node_index, unsigned int node_capacity)
	{
		if (nisml_cache == nullptr || nisml_representative_sample_counts == nullptr || nisml_representative_occupied_counts == nullptr ||
			nisml_representative_valid == nullptr || nisml_representative_write_locks == nullptr || nisml_representative_dirty == nullptr ||
			nisml_cache_ready == nullptr || node_index >= node_capacity || nisml_representative_capacity == 0u)
			return;

		for (unsigned int normal_face = 0; normal_face < ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT; normal_face++)
		{
			unsigned int cache_index = get_nisml_cache_index(node_index, normal_face);
			for (unsigned int representative_index = 0; representative_index < nisml_representative_capacity; representative_index++)
			{
				unsigned int flat_representative_index						= get_nisml_representative_index(cache_index, representative_index);
				nisml_cache[flat_representative_index]						= {};
				nisml_representative_valid[flat_representative_index]		= 0;
				nisml_representative_write_locks[flat_representative_index] = 0;
			}

			nisml_representative_sample_counts[cache_index]	  = 0;
			nisml_representative_occupied_counts[cache_index] = 0;
			nisml_representative_dirty[cache_index]			  = 0;
			nisml_cache_ready[cache_index]					  = 0;
		}
	}

	IlluminationAwareKDTreeNISMLCache* nisml_cache				   = nullptr;
	unsigned int nisml_representative_capacity					   = 1;
	AtomicType<unsigned int>* nisml_representative_sample_counts   = nullptr;
	AtomicType<unsigned int>* nisml_representative_occupied_counts = nullptr;
	unsigned char* nisml_representative_valid					   = nullptr;
	AtomicType<unsigned int>* nisml_representative_write_locks	   = nullptr;
	AtomicType<unsigned char>* nisml_representative_dirty		   = nullptr;
	// Indicates that log_importances contains a completely written, valid cache snapshot.
	// It does not indicate that representative collection is complete. A cache may remain ready while new representatives mark it dirty and trigger a later
	// rebuild.
	unsigned char* nisml_cache_ready				   = nullptr;
	AtomicType<unsigned int>* nisml_pending_cell_count = nullptr;
};

#endif
