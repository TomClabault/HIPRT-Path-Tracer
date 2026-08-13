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
		if (nisml_cache == nullptr || nisml_representative_sample_counts == nullptr || nisml_representative_write_locks == nullptr ||
			nisml_representative_ready == nullptr || nisml_cache_ready == nullptr || nisml_pending_cell_count == nullptr || *nisml_pending_cell_count == 0u)
			return;

		if (node_index >= node_capacity)
			return;

		unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(normal);
		unsigned int cache_index = get_nisml_cache_index(node_index, normal_face);
		if (nisml_cache_ready[cache_index] != 0)
			return;

		unsigned int sample_index	= hippt::atomic_fetch_add(&nisml_representative_sample_counts[cache_index], 1u);
		bool replace_representative = sample_index == 0u || random_number_generator() < 1.0f / static_cast<float>(sample_index + 1u);
		if (!replace_representative)
			return;

		if (hippt::atomic_compare_exchange(&nisml_representative_write_locks[cache_index], 0u, 1u) != 0u)
			return;

		IlluminationAwareKDTreeNISMLCache& cache = nisml_cache[cache_index];
		cache.representative_position			 = position;
		cache.representative_view_direction		 = view_direction;
		cache.representative_normal				 = normal;
		cache.representative_sg_specular_weight	 = sg_specular_weight;
		cache.representative_alpha_x			 = alpha_x;
		cache.representative_alpha_y			 = alpha_y;
		nisml_representative_ready[cache_index]	 = 1;

		hippt::atomic_exchange(&nisml_representative_write_locks[cache_index], 0u);
	}

	HIPRT_DEVICE unsigned int get_nisml_cache_index(unsigned int node_index, unsigned int normal_face) const
	{
		return node_index * ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT + normal_face;
	}

	HIPRT_DEVICE void initialize_nisml_cache_for_guiding_cell(unsigned int node_index, unsigned int node_capacity)
	{
		if (nisml_cache == nullptr || node_index >= node_capacity)
			return;

		for (unsigned int normal_face = 0; normal_face < ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT; normal_face++)
		{
			unsigned int cache_index = get_nisml_cache_index(node_index, normal_face);

			nisml_cache[cache_index]						= {};
			nisml_representative_sample_counts[cache_index] = 0;
			nisml_representative_write_locks[cache_index]	= 0;
			nisml_representative_ready[cache_index]			= 0;
			nisml_cache_ready[cache_index]					= 0;
		}
	}

	IlluminationAwareKDTreeNISMLCache* nisml_cache				 = nullptr;
	AtomicType<unsigned int>* nisml_representative_sample_counts = nullptr;
	AtomicType<unsigned int>* nisml_representative_write_locks	 = nullptr;
	unsigned char* nisml_representative_ready					 = nullptr;
	unsigned char* nisml_cache_ready							 = nullptr;
	AtomicType<unsigned int>* nisml_pending_cell_count			 = nullptr;
};

#endif
