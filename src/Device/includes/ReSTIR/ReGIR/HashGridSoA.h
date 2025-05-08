/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_H
#define DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_H

#include "Device/includes/ReSTIR/ReGIR/HashGridCellData.h"
#include "Device/includes/ReSTIR/ReGIR/ShadingSettings.h"
#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"

struct ReGIRHashGridSoADevice
{
	HIPRT_DEVICE void reset_reservoir(unsigned int hash_grid_cell_index, unsigned int reservoir_index_in_cell, int grid_index = -1)
	{
		if (grid_index == -1)
			grid_index = 0;

		unsigned int reservoirs_per_grid = m_total_number_of_cells * reservoirs.number_of_reservoirs_per_cell;
		int reservoir_index_in_grid = reservoirs_per_grid * grid_index + hash_grid_cell_index * reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		reservoirs.store_reservoir_opt(reservoir_index_in_grid, ReGIRReservoir());
		samples.store_sample(reservoir_index_in_grid, ReGIRReservoir().sample);
	}

	HIPRT_DEVICE void store_reservoir_and_sample_opt(const ReGIRReservoir& reservoir, ReGIRHashCellDataSoADevice& hash_cell_data, float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1)
	{
		if (grid_index != -1)
			// TODO fix this, we would need the number of reservoirs per grid in here but we don't have it
			return;

		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(world_position, camera_position, hash_key);
		if (!resolve_collision(hash_cell_data, hash_grid_cell_index, hash_key))
			return;

		int reservoir_index_in_grid = hash_grid_cell_index * reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		store_reservoir_and_sample_opt_from_index_in_grid(reservoir_index_in_grid, reservoir, grid_index);
	}

	HIPRT_DEVICE ReGIRReservoir read_full_reservoir_opt(const ReGIRHashCellDataSoADevice& hash_cell_data, float3 world_position, float3 camera_position, int reservoir_index_in_cell, int grid_index = -1) const
	{
		if (grid_index != -1)
			// TODO fix this, we would need the number of reservoirs per grid in here but we don't have it
			return ReGIRReservoir();

		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(world_position, camera_position, hash_key);
		if (!resolve_collision(hash_cell_data, hash_grid_cell_index, hash_key))
			return ReGIRReservoir();

		ReGIRReservoir reservoir;

		int reservoir_index_in_grid = hash_grid_cell_index * reservoirs.number_of_reservoirs_per_cell + reservoir_index_in_cell;

		float UCW = reservoirs.UCW[reservoir_index_in_grid];
		if (UCW <= 0.0f)
		{
			// If the reservoir doesn't have a valid sample, not even reading the rest of it
			ReGIRReservoir out;
			out.UCW = UCW;

			return out;
		}

		reservoir = reservoirs.read_reservoir<false>(reservoir_index_in_grid);
		reservoir.UCW = UCW;
		reservoir.sample = samples.read_sample(reservoir_index_in_grid);

		return reservoir;
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_no_collision_resolve(float3 world_position, float3 camera_position) const
	{
		unsigned int hash_key;
		unsigned int hash_cell_index = hash(world_position, camera_position, hash_key);

		return hash_cell_index;
	}

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_world_pos_with_collision_resolve(const ReGIRHashCellDataSoADevice& hash_cell_data, float3 world_position, float3 camera_position) const
	{
		unsigned int hash_key;
		unsigned int hash_grid_cell_index = hash(world_position, camera_position, hash_key);

		if (!resolve_collision(hash_cell_data, hash_grid_cell_index, hash_key))
			return ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY;
		else
			return hash_grid_cell_index;
	}

	HIPRT_DEVICE float3 get_cell_size(float3 world_position = make_float3(0, 0, 0), float3 camera_position = make_float3(0, 0, 0)) const
	{
		return make_float3(1.0f / grid_resolution.x, 1.0f / grid_resolution.y, 1.0f / grid_resolution.z);
	}

	HIPRT_DEVICE float3 jitter_world_position(float3 original_world_position, Xorshift32Generator& rng) const
	{
		float3 random_offset = make_float3(rng(), rng(), rng()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);

		return original_world_position + random_offset * get_cell_size() * 0.5f;
	}

	static constexpr float DEFAULT_GRID_SIZE = 2.5f;
	float3 grid_resolution = make_float3(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE);

	// These two SoAs are allocated to hold 'number_cells * number_reservoirs_per_cell'
	// So for a given 'hash_grid_cell_index', the cell contains reservoirs and samples going from 
	// reservoirs[hash_grid_cell_index * number_reservoirs_per_cell] to reservoirs[cell_index * number_reservoirs_per_cell + number_reservoirs_per_cell[
	ReGIRReservoirSoADevice reservoirs;
	ReGIRSampleSoADevice samples;

	unsigned int m_total_number_of_cells = 0;

	/**
	 * PCG for the first hash function
	 */
	HIPRT_DEVICE unsigned int h1_pcg(unsigned int seed) const
	{
		unsigned int state = seed * 747796405u + 2891336453u;
		unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		
		return (word >> 22u) ^ word;
	}

	HIPRT_HOST_DEVICE unsigned int h1_pcg(float seed) const
	{
		return h1_pcg(hippt::float_as_uint(seed));
	}

	/**
	 * xxhash32 for the second hash function
	 */
	HIPRT_DEVICE unsigned int h2_xxhash32(unsigned int seed) const
	{
		constexpr unsigned int PRIME32_2 = 2246822519U;
		constexpr unsigned int PRIME32_3 = 3266489917U;
		constexpr unsigned int PRIME32_4 = 668265263U;
		constexpr unsigned int PRIME32_5 = 374761393U;

		unsigned int h32 = seed + PRIME32_5;

		h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
		h32 = PRIME32_2*(h32^(h32 >> 15));
		h32 = PRIME32_3*(h32^(h32 >> 13));

		return h32^(h32 >> 16);
	}

	HIPRT_HOST_DEVICE unsigned int h2_xxhash32(float seed) const
	{
		return h2_xxhash32(hippt::float_as_uint(seed));
	}

	/**
	 * Returns the hash cell index of the given world position and camera position. Does not resolve collisions.
	 * The hash key for resolving collision is given in 'out_hash_key'
	 */
	HIPRT_DEVICE unsigned int hash(float3 world_position, float3 camera_position, unsigned int& out_hash_key) const
	{
		float3 relative_to_camera = world_position;// -camera_position;

		float cell_size = 1.0f / grid_resolution.x;

		unsigned int grid_coord_x = static_cast<int>(relative_to_camera.x * grid_resolution.x);
		unsigned int grid_coord_y = static_cast<int>(relative_to_camera.y * grid_resolution.y);
		unsigned int grid_coord_z = static_cast<int>(relative_to_camera.z * grid_resolution.z);

		// Using two hash functions as proposed in [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boissé, 2021]
		out_hash_key = hippt::max(1u, h2_xxhash32(cell_size + h2_xxhash32(grid_coord_z + h2_xxhash32(grid_coord_y + h2_xxhash32(grid_coord_x)))));
		
		unsigned int cell_hash = h1_pcg(cell_size + h1_pcg(grid_coord_z + h1_pcg(grid_coord_y + h1_pcg(grid_coord_x)))) % m_total_number_of_cells;

		return cell_hash;
	}

	/**
	 * Returns true if the collision was resolved with success and the new hash
	 * (or unchanged if there was no collision) is set in 'in_out_base_hash'
	 * 
	 * Returns false if the given 'in_out_hash_cell_index' refers to a hash cell that hasn't been
	 * allocated yet or if there was a collision but it couldn't be resolved and the collision resolution was
	 * aborted because too many iterations
	 */
	template <bool isInsertion = false>
	HIPRT_DEVICE bool resolve_collision(const ReGIRHashCellDataSoADevice& hash_cell_data, unsigned int& in_out_hash_cell_index, unsigned int hash_key) const
	{
		unsigned int existing_hash_key = hash_cell_data.hash_keys[in_out_hash_cell_index];
		if (existing_hash_key == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
		{
			// This is refering to a hash cell that hasn't been populated yet

			if (!isInsertion)
				// If we're not inserting, this means that we're querrying an empty cell
				return false;
			else
			{
				// This is refering to a hash cell that hasn't been populated yet and we're
				// inserting into it so we just found an empty cell first try
				// 
				// Let's try to insert atomically into it

				unsigned int previous_hash_key = hippt::atomic_compare_exchange(&hash_cell_data.hash_keys[in_out_hash_cell_index], ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY, hash_key);
				if (previous_hash_key == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
				{
					// (and we made sure sure through an atomic CAS that someone else wasn't
					// also competing for that empty cell)

					return true;
				}
				else if (previous_hash_key == hash_key)
				{
					// Another thread just inserted the same hash key at the same time but this
					// current thread here wasn't fast enough on the atomic compare exchange above
					// so the key was already inserted.

					// This thread has nothing else to do.
					return true;
				}
				else
				{
					// Another hash key has been inserted in the same position, we're going to have to
					// probe for a good position
				}
			}
		}

		if (existing_hash_key != hash_key)
		{
			// This is a collision

			unsigned int base_cell_index = in_out_hash_cell_index;

			// Linear probing
			for (int i = 1; i <= 32; i++)
			{
				unsigned int next_hash_cell_index = (base_cell_index + i) % m_total_number_of_cells;
				if (next_hash_cell_index == base_cell_index)
					// We looped on the whole hash table. Couldn't find an empty cell
					return false;

				unsigned int next_cell_hash_key = hash_cell_data.hash_keys[next_hash_cell_index];
				if (next_cell_hash_key == hash_key)
				{
					// Stopping if we found our proper cell (with our hash).
					//
					// This means that we have resolved the collision 

					in_out_hash_cell_index = next_hash_cell_index;

					// We found an empty cell
					return true;
				}
				else if (next_cell_hash_key == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
				{
					if (isInsertion)
					{
						// Stopping if we found an empty cell for insertion

						unsigned int previous_hash_key = hippt::atomic_compare_exchange(&hash_cell_data.hash_keys[next_hash_cell_index], ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY, hash_key);
						if (previous_hash_key == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
						{
							// (and we made sure sure through an atomic CAS that someone else wasn't
							// also competing for that empty cell)

							in_out_hash_cell_index = next_hash_cell_index;

							return true;
						}
						else if (previous_hash_key == hash_key)
						{
							// Another thread just inserted the same hash key at the same time but this
							// current thread here wasn't fast enough on the atomic compare exchange
							// above so the key was already inserted.

							in_out_hash_cell_index = next_hash_cell_index;

							// This thread has nothing else to do.
							return true;
						}
					}
					else
					{
						// This is a query but we've hit an empty cell during probing which means that we're querrying
						// a cell that has never been populated

						return false;
					}
				}
			}

			// Linear probing couldn't find a valid position in the hash map
			return false;
		}
		else
			// This is already our hash, no collision
			return true;
	}
	
private:
	HIPRT_DEVICE void store_reservoir_and_sample_opt_from_index_in_grid(int reservoir_index_in_grid, const ReGIRReservoir& reservoir, int grid_index = -1)
	{
		if (grid_index != -1)
			// TODO fix this, we would need the number of reservoirs per grid in here but we don't have it
			return;

		if (reservoir.UCW <= 0.0f)
		{
			// No need to store the rest if the UCW is invalid
			reservoirs.UCW[reservoir_index_in_grid] = reservoir.UCW;

			return;
		}

		reservoirs.store_reservoir_opt(reservoir_index_in_grid, reservoir);
		samples.store_sample(reservoir_index_in_grid, reservoir.sample);
	}
};

#endif
