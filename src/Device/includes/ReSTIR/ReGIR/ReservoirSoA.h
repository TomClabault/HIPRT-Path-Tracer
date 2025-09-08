/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_RESERVOIR_SOA_H
#define DEVICE_KERNELS_REGIR_RESERVOIR_SOA_H

#include "Device/includes/ReSTIR/ReGIR/Reservoir.h"

#ifndef __KERNELCC__
template <template <typename> typename DataContainer>
struct ReGIRHashGridSoAHost;
#endif

struct ReGIRSampleSoADevice
{
	using ReGIRSampleEmissiveTriangleIndicesPackingType = unsigned long long int;

	HIPRT_HOST_DEVICE void store_sample(int linear_reservoir_index, const ReGIRSample& sample)
	{
		set_emissive_triangle_index(linear_reservoir_index, sample.emissive_triangle_global_index);
		point_on_light_random_seed[linear_reservoir_index] = sample.point_on_light_random_seed;
	}

	HIPRT_HOST_DEVICE ReGIRSample read_sample(int linear_reservoir_index) const
	{
		ReGIRSample sample;

		sample.emissive_triangle_global_index = get_emissive_triangle_index(linear_reservoir_index);
		sample.point_on_light_random_seed = point_on_light_random_seed[linear_reservoir_index];

		return sample;
	}

	unsigned int* point_on_light_random_seed = nullptr;

private:
	AtomicType<ReGIRSampleEmissiveTriangleIndicesPackingType>* emissive_triangle_indices_packed = nullptr;

	unsigned int bits_per_emissive_triangle_global_index = 0;

	/**
	 * Bitwise AND the value at 'emissive_triangle_indices_packed[element_index]' with 'clear_mask'
	 * and then bitwise OR with 'bits' atomically
	 */
	HIPRT_DEVICE void update_element_atomically(unsigned int element_index, ReGIRSampleEmissiveTriangleIndicesPackingType clear_mask, ReGIRSampleEmissiveTriangleIndicesPackingType bits)
	{
		ReGIRSampleEmissiveTriangleIndicesPackingType old_val, new_val;

		do {
			// Current value
			old_val = hippt::atomic_load(&emissive_triangle_indices_packed[element_index]);

			// Clear and set
			new_val = (old_val & clear_mask) | bits;
		} while (hippt::atomic_compare_exchange(&emissive_triangle_indices_packed[element_index], old_val, new_val) != old_val); // attempt update until success
	}

	HIPRT_DEVICE void set_emissive_triangle_index(unsigned int linear_reservoir_index, unsigned int emissive_triangle_global_index)
	{
		unsigned int bit_offset_start_in_element = (linear_reservoir_index * bits_per_emissive_triangle_global_index) % (sizeof(ReGIRSampleEmissiveTriangleIndicesPackingType) * 8);
		unsigned int element_index = linear_reservoir_index * bits_per_emissive_triangle_global_index / (sizeof(ReGIRSampleEmissiveTriangleIndicesPackingType) * 8);

		if (bit_offset_start_in_element + bits_per_emissive_triangle_global_index > sizeof(ReGIRSampleEmissiveTriangleIndicesPackingType) * 8)
		{
			// If the index is straddling two differents elements

			unsigned int bits_in_first_element = sizeof(ReGIRSampleEmissiveTriangleIndicesPackingType) * 8 - bit_offset_start_in_element;
			unsigned int bits_in_second_element = bits_per_emissive_triangle_global_index - bits_in_first_element;

			ReGIRSampleEmissiveTriangleIndicesPackingType bits_in_first_element_mask = (1 << bits_in_first_element) - 1;
			ReGIRSampleEmissiveTriangleIndicesPackingType bits_in_second_element_mask = (1 << bits_in_second_element) - 1;

			ReGIRSampleEmissiveTriangleIndicesPackingType first_element = emissive_triangle_indices_packed[element_index];
			ReGIRSampleEmissiveTriangleIndicesPackingType second_element = emissive_triangle_indices_packed[element_index + 1];

			ReGIRSampleEmissiveTriangleIndicesPackingType clear_mask_element_1 = ~(bits_in_first_element_mask << bit_offset_start_in_element);
			ReGIRSampleEmissiveTriangleIndicesPackingType bits_element_1 = (emissive_triangle_global_index & bits_in_first_element_mask) << bit_offset_start_in_element;
			update_element_atomically(element_index, clear_mask_element_1, bits_element_1);

			ReGIRSampleEmissiveTriangleIndicesPackingType clear_mask_element_2 = ~bits_in_second_element_mask;
			ReGIRSampleEmissiveTriangleIndicesPackingType bits_element_2 = (emissive_triangle_global_index >> bits_in_first_element) & bits_in_second_element_mask;
			update_element_atomically(element_index + 1, clear_mask_element_2, bits_element_2);
		}
		else
		{
			// Packed mesh index not straddling
			ReGIRSampleEmissiveTriangleIndicesPackingType clear_mask = ~(static_cast<ReGIRSampleEmissiveTriangleIndicesPackingType>((1 << bits_per_emissive_triangle_global_index) - 1) << bit_offset_start_in_element);
			ReGIRSampleEmissiveTriangleIndicesPackingType bits = static_cast<ReGIRSampleEmissiveTriangleIndicesPackingType>(emissive_triangle_global_index & ((1 << bits_per_emissive_triangle_global_index) - 1)) << bit_offset_start_in_element;

			// Updating the bits atomically
			update_element_atomically(element_index, clear_mask, bits);
		}
	}

	HIPRT_DEVICE unsigned int get_emissive_triangle_index(unsigned int linear_reservoir_index) const
	{
		unsigned int bit_offset_start_in_element = (linear_reservoir_index * bits_per_emissive_triangle_global_index) % (sizeof(ReGIRSampleEmissiveTriangleIndicesPackingType) * 8);
		unsigned int element_index = linear_reservoir_index * bits_per_emissive_triangle_global_index / (sizeof(ReGIRSampleEmissiveTriangleIndicesPackingType) * 8);

		if (bit_offset_start_in_element + bits_per_emissive_triangle_global_index > sizeof(ReGIRSampleEmissiveTriangleIndicesPackingType) * 8)
		{
			// If the mesh index is straddling two differents elements

			unsigned int bits_in_first_element = sizeof(ReGIRSampleEmissiveTriangleIndicesPackingType) * 8 - bit_offset_start_in_element;
			unsigned int bits_in_second_element = bits_per_emissive_triangle_global_index - bits_in_first_element;

			unsigned int bits_in_first_element_mask = (1 << bits_in_first_element) - 1;
			unsigned int bits_in_second_element_mask = (1 << bits_in_second_element) - 1;

			unsigned int first_part = (emissive_triangle_indices_packed[element_index] >> bit_offset_start_in_element) & bits_in_first_element_mask;
			unsigned int second_part = (emissive_triangle_indices_packed[element_index + 1]) & bits_in_second_element_mask;

			return first_part | (second_part << bits_in_first_element);
		}
		else
			// Packed mesh index not straddling, just need to fetch the bits
			return (emissive_triangle_indices_packed[element_index] >> bit_offset_start_in_element) & ((1 << bits_per_emissive_triangle_global_index) - 1);
	}

	template <template <typename> typename DataContainer>
	friend struct ReGIRHashGridSoAHost;
};

struct ReGIRReservoirSoADevice
{
	HIPRT_HOST_DEVICE void store_reservoir_opt(int linear_reservoir_index, const ReGIRReservoir& reservoir)
	{
		UCW[linear_reservoir_index] = reservoir.UCW;
	}

	/**
	 * The template parameter can be used to indicate whether or not to read the UCW.
	 * 
	 * This makes sense to pass this parameter as false if you've already read the UCW
	 * of the reservoir by some other means
	 */
	template <bool readUCW = true>
	HIPRT_HOST_DEVICE ReGIRReservoir read_reservoir(int linear_reservoir_index) const
	{
		ReGIRReservoir reservoir;

		if constexpr (readUCW)
			reservoir.UCW = UCW[linear_reservoir_index];

		return reservoir;
	}

	float* UCW = nullptr;

	unsigned int number_of_reservoirs_per_cell = 0;
};

#endif
