/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_DEVICE_ALIAS_TABLE_H
#define DEVICE_INCLUDES_DEVICE_ALIAS_TABLE_H

#include "HostDeviceCommon/Xorshift.h"

struct AliasTableDevice
{
	HIPRT_HOST_DEVICE unsigned int sample(Xorshift32Generator& rng) const
	{
		int random_index = rng.random_index(size);
		float probability = alias_table_probas[random_index];
		if (rng() > probability)
			// Picking the alias
			random_index = alias_table_alias[random_index];

		return random_index;
	}

	int* alias_table_alias = nullptr;
	float* alias_table_probas = nullptr;

	float sum_elements = 0.0f;
	unsigned int size = 0;
};

#endif
