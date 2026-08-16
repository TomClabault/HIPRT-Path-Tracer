/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NISML_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NISML_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNISMLCache.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"

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
		if (sample_index >= nisml_representative_capacity)
		{
			float replacement_probability = static_cast<float>(nisml_representative_capacity) / static_cast<float>(sample_index + 1u);
			if (random_number_generator() >= replacement_probability)
				return;

			replacement_index = static_cast<unsigned int>(random_number_generator.random_index(static_cast<int>(nisml_representative_capacity)));
		}

		if (hippt::atomic_compare_exchange(&nisml_representative_write_locks[cache_index], 0u, 1u) != 0u)
			// Simple fail fast approach: if the lock is already taken, we skip this representative. This may result in some representatives being skipped, but
			// it avoids potential deadlocks and should be fine for the purpose of representative collection.
			return;

		unsigned int representative_index		 = get_nisml_representative_index(cache_index, replacement_index);
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
			nisml_representative_occupied_counts[cache_index]++;
		}

		if (nisml_representative_dirty[cache_index] == 0)
		{
			nisml_representative_dirty[cache_index] = 1;
			hippt::atomic_fetch_add(nisml_pending_cell_count, 1u);
		}

		hippt::atomic_exchange(&nisml_representative_write_locks[cache_index], 0u);
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
				unsigned int flat_representative_index				  = get_nisml_representative_index(cache_index, representative_index);
				nisml_cache[flat_representative_index]				  = {};
				nisml_representative_valid[flat_representative_index] = 0;
			}

			nisml_representative_sample_counts[cache_index]	  = 0;
			nisml_representative_occupied_counts[cache_index] = 0;
			nisml_representative_write_locks[cache_index]	  = 0;
			nisml_representative_dirty[cache_index]			  = 0;
			nisml_cache_ready[cache_index]					  = 0;
		}
	}

	IlluminationAwareKDTreeNISMLCache* nisml_cache				 = nullptr;
	unsigned int nisml_representative_capacity					 = 1;
	AtomicType<unsigned int>* nisml_representative_sample_counts = nullptr;
	unsigned int* nisml_representative_occupied_counts			 = nullptr;
	unsigned char* nisml_representative_valid					 = nullptr;
	AtomicType<unsigned int>* nisml_representative_write_locks	 = nullptr;
	unsigned char* nisml_representative_dirty					 = nullptr;
	// Indicates that log_importances contains a completely written, valid cache snapshot.
	// It does not indicate that representative collection is complete. A cache may remain ready while new representatives mark it dirty and trigger a later
	// rebuild.
	unsigned char* nisml_cache_ready				   = nullptr;
	AtomicType<unsigned int>* nisml_pending_cell_count = nullptr;
};

#endif
