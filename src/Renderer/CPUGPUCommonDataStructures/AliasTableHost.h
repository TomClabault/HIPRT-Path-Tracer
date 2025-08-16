/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
		out.alias_table_alias = aliases.data();

		out.size = size;
		out.sum_elements = sum_elements;

		return out;
	}

	DataContainer<float> probas;
	DataContainer<int> aliases;

	float sum_elements = 0.0f;
	unsigned int size = 0;
};

#endif
