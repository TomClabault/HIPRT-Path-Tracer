/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ALIAS_TABLE_HOST_H
#define RENDERER_ALIAS_TABLE_HOST_H

#include "Device/includes/AliasTable.h"

template <template <typename> typename DataContainer>
struct AliasTableHost
{
	AliasTableDevice to_device()
	{
		AliasTableDevice out;

		out.alias_table_probas = probas.data();
		out.alias_table_alias  = aliases.data();

		out.size		 = size;
		out.sum_elements = sum_elements;

		return out;
	}

	void free()
	{
		if constexpr (std::is_same<DataContainer<int>, std::vector<int>>::value)
		{
			probas.clear();
			aliases.clear();
		}
		else if constexpr (std::is_same<DataContainer<int>, OrochiBuffer<int>>::value)
		{
			probas.free();
			aliases.free();
		}
	}

	DataContainer<float> probas;
	DataContainer<int> aliases;

	float sum_elements = 0.0f;
	unsigned int size  = 0;
};

#endif
