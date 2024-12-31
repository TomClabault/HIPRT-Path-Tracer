/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_NESTED_DIELECTRICS_H
#define DEVICE_NESTED_DIELECTRICS_H

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"

#include <hiprt/hiprt_common.h>

/**
 * Reference:
 *
 * [1] [Simple Nested Dielectrics in Ray Traced Images, Schmidt, 2002]
 */
struct StackPriorityEntry
{
	// How many bits for encoding the packed priority
	// and its shift to locate the bits in the packed 32bits integer
	static constexpr unsigned int PRIORITY_BIT_MASK = 0b1111;
	static constexpr unsigned int PRIORITY_BIT_SHIFT = 0;
	static constexpr unsigned int PRIORITY_MAXIMUM = PRIORITY_BIT_MASK;
	// How many bits for encoding the topmost flag
	// and its shift to locate the bits in the packed 32bits integer
	static constexpr unsigned int TOPMOST_BIT_MASK = 0b1;
	static constexpr unsigned int TOPMOST_BIT_SHIFT = PRIORITY_BIT_SHIFT + 4;
	// How many bits for encoding the odd_parity flag
	// and its shift to locate the bits in the packed 32bits integer
	static constexpr unsigned int ODD_PARTIY_BIT_MASK = 0b1;
	static constexpr unsigned int ODD_PARTIY_BIT_SHIFT = TOPMOST_BIT_SHIFT + 1;

	// How many bits for encoding the material_index flag
	// and its shift to locate the bits in the packed 32bits integer
	// This is the rest of the bits after we've added the other flags
	static constexpr unsigned int COMBINED_OTHER_FLAGS = (PRIORITY_BIT_MASK << PRIORITY_BIT_SHIFT) | (TOPMOST_BIT_MASK << TOPMOST_BIT_SHIFT) | (ODD_PARTIY_BIT_MASK << ODD_PARTIY_BIT_SHIFT);
	static constexpr unsigned int MATERIAL_INDEX_BIT_SHIFT = ODD_PARTIY_BIT_SHIFT + 1;
	static constexpr unsigned int MATERIAL_INDEX_BIT_MASK = (0xffffffff & (~COMBINED_OTHER_FLAGS)) >> MATERIAL_INDEX_BIT_SHIFT;
	static constexpr unsigned int MATERIAL_INDEX_MAXIMUM = MATERIAL_INDEX_BIT_MASK;

	HIPRT_HOST_DEVICE StackPriorityEntry()
	{
		set_priority(0);
		set_odd_parity(true);
		set_topmost(true);
		// Setting the material index to the maximum
		set_material_index(MATERIAL_INDEX_BIT_MASK);
	}

	HIPRT_HOST_DEVICE void set_priority(int priority)
	{
		// Clear
		packed_data &= ~(PRIORITY_BIT_MASK << PRIORITY_BIT_SHIFT);
		// Set
		packed_data |= (priority & PRIORITY_BIT_MASK) << PRIORITY_BIT_SHIFT;
	}

	HIPRT_HOST_DEVICE void set_topmost(bool topmost)
	{
		// Clear
		packed_data &= ~(TOPMOST_BIT_MASK << TOPMOST_BIT_SHIFT);
		// Set
		packed_data |= (topmost == true) << TOPMOST_BIT_SHIFT;
	}

	HIPRT_HOST_DEVICE void set_odd_parity(bool odd_parity)
	{
		// Clear
		packed_data &= ~(ODD_PARTIY_BIT_MASK << ODD_PARTIY_BIT_SHIFT);
		// Set
		packed_data |= (odd_parity == true) << ODD_PARTIY_BIT_SHIFT;
	}

	HIPRT_HOST_DEVICE void set_material_index(int material_index)
	{
		// Clear
		packed_data &= ~(MATERIAL_INDEX_BIT_MASK << MATERIAL_INDEX_BIT_SHIFT);
		// Set
		packed_data |= (material_index & MATERIAL_INDEX_BIT_MASK) << MATERIAL_INDEX_BIT_SHIFT;
	}

	HIPRT_HOST_DEVICE int get_priority()
	{
		return (packed_data & (PRIORITY_BIT_MASK << PRIORITY_BIT_SHIFT)) >> PRIORITY_BIT_SHIFT;
	}

	HIPRT_HOST_DEVICE bool get_topmost()
	{
		return (packed_data & (TOPMOST_BIT_MASK << TOPMOST_BIT_SHIFT)) >> TOPMOST_BIT_SHIFT;
	}

	HIPRT_HOST_DEVICE bool get_odd_parity()
	{
		return (packed_data & (ODD_PARTIY_BIT_MASK << ODD_PARTIY_BIT_SHIFT)) >> ODD_PARTIY_BIT_SHIFT;
	}

	HIPRT_HOST_DEVICE int get_material_index()
	{
		return (packed_data & (MATERIAL_INDEX_BIT_MASK << MATERIAL_INDEX_BIT_SHIFT)) >> MATERIAL_INDEX_BIT_SHIFT;
	}

	// Packed data contains:
	//	- the priority of the stack entry
	//	- whether or not this is the topmost entry for that material in the stack
	//	- An odd_parity flag
	//	- The material index
	// 
	// We get the bits:
	// 
	// **** *** material index* **** **OT PRIO
	// 
	// With :
	// - O the odd_parity flag
	// - T the topmost flag
	// - PRIO the dielectric priority 
	unsigned int packed_data;
};

struct InteriorStack
{
	/**
	 * Pushes a new material index onto the stack
	 * 
	 * Returns true if that intersection should be skipped (because we are currently in a material with 
	 * higher priority than the material we just intersected)
	 * 
	 * Returns false if that intersection should not be skipped
	 */
	HIPRT_HOST_DEVICE bool push(int& out_incident_material_index, int& out_outgoing_material_index, bool& out_inside_material, int material_index, int material_priority)
	{
		// Index of the material we last entered before intersecting the
		// material we're currently inserting in the stack
		int last_entered_mat_index = 0;
		for (last_entered_mat_index = stack_position; last_entered_mat_index >= 0; last_entered_mat_index--)
			// The three conditions in order are:
			// 	- We found a materal in the stack that is not the material that we're currently intersecting
			//	- The entry of that material in the stack is the topmost (the last entry of its material kind)
			//	- The entry of that material in the stack is odd_parity = we've entered that material but haven't left it yet
			//
			//	= the last entered material
			if (stack[last_entered_mat_index].get_material_index() != material_index && stack[last_entered_mat_index].get_topmost() && stack[last_entered_mat_index].get_odd_parity())
				break;

		// Parity of the material we're inserting in the stack
		bool odd_parity = true;
		// Index in the stack of the previous material that is the same as
		// the one we're trying to insert in the stack.
		int previous_same_mat_index;

		for (previous_same_mat_index = stack_position; previous_same_mat_index >= 0; previous_same_mat_index--)
		{
			if (stack[previous_same_mat_index].get_material_index() == material_index)
			{
				// The previous stack entry of the same material is not the topmost anymore
				stack[previous_same_mat_index].set_topmost(false);
				// The current parity is the inverse of the previous one
				odd_parity = !stack[previous_same_mat_index].get_odd_parity();

				break;
			}
		}
		
		out_inside_material = !odd_parity;

		// Inserting the material in the stack
		if (stack_position < NestedDielectricsStackSize - 1)
			stack_position++;
		stack[stack_position].set_material_index(material_index);
		stack[stack_position].set_odd_parity(odd_parity);
		stack[stack_position].set_topmost(true);
		stack[stack_position].set_priority(material_priority);

		if (material_priority < stack[last_entered_mat_index].get_priority())
		{
			// Skipping the boundary because the intersected material has a
			// lower priority than the material we're currently in
			return true;
		}
		else
		{
			if (odd_parity)
			{
				// We are entering the material
				out_incident_material_index = stack[last_entered_mat_index].get_material_index();
				out_outgoing_material_index = material_index;
			}
			else
			{
				// Exiting material
				out_incident_material_index = material_index;
				out_outgoing_material_index = stack[last_entered_mat_index].get_material_index();
			}

			// Not skipping the boundary
			return false;
		}
	}

	HIPRT_HOST_DEVICE void pop(const bool inside_material)
	{
		int stack_top_mat_index = stack[stack_position].get_material_index();
		if (stack_position > 0)
			// Checking that we have room to pop.
			// For a very small stack (size of 2) that overflown 
			// (we couldn't push all the material we needed to because of 
			// stack size constraint), it can happen that the stack position
			// at this point is already 0 and we cannot pop.
			stack_position--;

		if (inside_material)
		{
			int previous_same_mat_index;
			for (previous_same_mat_index = stack_position; previous_same_mat_index >= 0; previous_same_mat_index--)
				if (stack[previous_same_mat_index].get_material_index() == stack_top_mat_index)
					break;

			if (previous_same_mat_index >= 0)
				for (int i = previous_same_mat_index + 1; i <= stack_position; i++)
					stack[i - 1] = stack[i];

			// For very small stacks (2 for example), we may not be able to pop twice
			// at all so we check the position on the stack first
			if (stack_position > 0)
				stack_position--;
		}

		for (int i = stack_position; i >= 0; i--)
		{
			if (stack[i].get_material_index() == stack_top_mat_index)
			{
				stack[i].set_topmost(true);
				break;
			}
		}
	}

	// We only need all of this if the stack size is actually > 0,
	// otherwise, we're just not going to do the nested dielectrics handling at all

	StackPriorityEntry stack[NestedDielectricsStackSize];
	static constexpr unsigned int MAX_MATERIAL_INDEX = StackPriorityEntry::MATERIAL_INDEX_MAXIMUM;

	// Stack position is pointing at the last valid entry.
	// Entry 0 is always present and represent air basically
	int stack_position = 0;
};

#endif
