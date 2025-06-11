/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

 #ifndef DEVICE_INCLUDES_HASH_GRID_H
 #define DEVICE_INCLUDES_HASH_GRID_H

#include "HostDeviceCommon/Math.h"

struct HashGrid
{
	static constexpr unsigned int UNDEFINED_CHECKSUM_OR_GRID_INDEX = 0xFFFFFFFF;

    /**
	 * Returns true if the collision was resolved with success and the new hash
	 * (or unchanged if there was no collision) is set in 'in_out_base_hash'
	 * 
	 * Returns false if the given 'in_out_hash_cell_index' refers to a hash cell that hasn't been
	 * allocated yet or if there was a collision but it couldn't be resolved and the collision resolution was
	 * aborted because too many iterations
	 */
	template <int maxLinearProbingSteps, bool isInsertion = false>
	HIPRT_DEVICE static bool resolve_collision(AtomicType<unsigned int>* checksum_buffer, unsigned int total_number_of_cells, unsigned int& in_out_hash_cell_index, unsigned int checksum, unsigned int opt_existing_checksum = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		unsigned int existing_checksum;
		if (opt_existing_checksum != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			// The current hash key was passed as an argument, no need to fetch from memory
			existing_checksum = opt_existing_checksum;
		else
			existing_checksum = checksum_buffer[in_out_hash_cell_index];

		if (existing_checksum == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		{
			// This is refering to a hash cell that hasn't been populated yet

			if (!isInsertion)
			{
				// If we're not inserting, this means that we're querrying an empty cell
				return false;
			}
			else
			{
				// This is refering to a hash cell that hasn't been populated yet and we're
				// inserting into it so we just found an empty cell first try
				// 
				// Let's try to insert atomically into it

				unsigned int previous_checksum = hippt::atomic_compare_exchange(&checksum_buffer[in_out_hash_cell_index], HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX, checksum);
				if (previous_checksum == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
				{
					// (and we made sure sure through an atomic CAS that someone else wasn't
					// also competing for that empty cell)

					return true;
				}
				else if (previous_checksum == checksum)
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

		if (existing_checksum != checksum)
		{
			// This is a collision

			unsigned int base_cell_index = in_out_hash_cell_index;

			// Linear probing
			for (int i = 1; i <= maxLinearProbingSteps; i++)
			{
				unsigned int next_hash_cell_index = (base_cell_index + i) % total_number_of_cells;
				if (next_hash_cell_index == base_cell_index)
					// We looped on the whole hash table. Couldn't find an empty cell
					return false;

				unsigned int next_cell_checksum = checksum_buffer[next_hash_cell_index];
				if (next_cell_checksum == checksum)
				{
					// Stopping if we found our proper cell (with our hash).
					//
					// This means that we have resolved the collision 

					in_out_hash_cell_index = next_hash_cell_index;

					return true;
				}
				else if (next_cell_checksum == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
				{
					if (isInsertion)
					{
						// Stopping if we found an empty cell for insertion

						unsigned int previous_checksum = hippt::atomic_compare_exchange(&checksum_buffer[next_hash_cell_index], HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX, checksum);
						if (previous_checksum == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
						{
							// (and we made sure sure through an atomic CAS that someone else wasn't
							// also competing for that empty cell)

							in_out_hash_cell_index = next_hash_cell_index;

							return true;
						}
						else if (previous_checksum == checksum)
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
};

 #endif
