/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_NESTED_DIELECTRICS_H
#define DEVICE_NESTED_DIELECTRICS_H

#include "HostDeviceCommon/KernelOptions.h"

#include <hiprt/hiprt_common.h>

 /**
  * Reference:
  *
  * [1] [Ray Tracing Gems 1 - Automatic Handling of Materials in Nested Volumes] https://www.realtimerendering.com/raytracinggems/rtg/index.html
  * [2] [Simple Nested Dielectrics in Ray Traced Images, Schmidt, 2002]
  */

template <int Strategy>
struct InteriorStackImpl {};

struct StackEntry
{
	// TODO do packing in there
	bool topmost = true;
	bool odd_parity = true;

	int material_index = -1;
};

template <>
struct InteriorStackImpl<ISS_AUTOMATIC>
{
	// TODO leaving material never used ? Or used only where we already know its value so not needed
	// Unused parameter at the end here to have the same signature as InteriorStackPriority
	HIPRT_HOST_DEVICE bool push(int& incident_material_index, int& outgoing_material_index, bool& inside_material, int material_index, int)
	{
		// Parity of the material we're inserting in the stack
		bool odd_parity = true;
		// Index in the stack of the previous material that is the same as
		// the one we're trying to insert in the stack.
		int previous_same_mat_index;

		for (previous_same_mat_index = stack_position; previous_same_mat_index >= 0; previous_same_mat_index--)
		{
			if (stack[previous_same_mat_index].material_index == material_index)
			{
				// The previous material is not the topmost anymore
				stack[previous_same_mat_index].topmost = false;
				// The current parity is the inverse of the previous one
				odd_parity = !stack[previous_same_mat_index].odd_parity;

				break;
			}
		}

		// Index of the material we last entered before intersecting the
		// material we're currently inserting in the stack. This for loop cannot
		// give us last_entered_mat_index = -1 because we will on the air in the
		// worst case scenario (the air is the stack[0] entry)
		int last_entered_mat_index = 0;
		for (last_entered_mat_index = stack_position; last_entered_mat_index >= 0; last_entered_mat_index--)
			if (stack[last_entered_mat_index].material_index != material_index && stack[last_entered_mat_index].topmost && stack[last_entered_mat_index].odd_parity)
				break;

		// Inserting the material in the stack
		if (stack_position < NestedDielectricsStackSize - 1)
			stack_position++;
		stack[stack_position].material_index = material_index;
		stack[stack_position].odd_parity = odd_parity;
		stack[stack_position].topmost = true;

		if (odd_parity)
		{
			// We are entering the material
			incident_material_index = stack[last_entered_mat_index].material_index;
			outgoing_material_index = material_index;
		}
		else
		{
			// Exiting material
			outgoing_material_index = stack[last_entered_mat_index].material_index;

			if (last_entered_mat_index < previous_same_mat_index)
				incident_material_index = material_index;
			else
			{
				incident_material_index = outgoing_material_index;

				// Return true because we are skipping the boundary we just hit
				return true;
			}
		}

		inside_material = !odd_parity;

		return false;
	}

	HIPRT_HOST_DEVICE void pop(bool inside_material)
	{
		int stack_top_mat_index = stack[stack_position].material_index;
		stack_position--;

		if (inside_material)
		{
			int previous_same_mat_index;
			for (previous_same_mat_index = stack_position; previous_same_mat_index >= 0; previous_same_mat_index--)
				if (stack[previous_same_mat_index].material_index == stack_top_mat_index)
					break;

			if (previous_same_mat_index >= 0)
				for (int i = previous_same_mat_index + 1; i <= stack_position; i++)
					stack[i - 1] = stack[i];

			stack_position--;
		}

		for (int i = stack_position; i >= 0; i--)
		{
			if (stack[i].material_index == stack_top_mat_index)
			{
				stack[i].topmost = true;
				break;
			}
		}
	}

	StackEntry stack[NestedDielectricsStackSize];

	int stack_position = 0;
};

struct StackPriorityEntry
{
	// How many bits for encoding the packed priority
	static constexpr unsigned int PRIORITY_BITS = 4;
	static constexpr unsigned int PRIORITY_MAXIMUM = (1 << PRIORITY_BITS) - 1;
	// How many bits for encoding the topmost flag
	static constexpr unsigned int TOPMOST_BITS = 1;
	// How many bits for encoding the odd_parity flag
	static constexpr unsigned int ODD_PARTIY_BITS = 1;

	// How many bits for encoding the material_index flag
	// 
	// This is the rest of the bits after we've added the other 
	// flags
	static constexpr unsigned int MATERIAL_INDEX_BITS = 32 - PRIORITY_BITS - TOPMOST_BITS - ODD_PARTIY_BITS;
	static constexpr unsigned int MATERIAL_INDEX_MAXIMUM = (1 << MATERIAL_INDEX_BITS) - 1;

	HIPRT_HOST_DEVICE StackPriorityEntry()
	{
		priority = 0;
		odd_parity = true;
		topmost = true;
		// Setting the material index to the maximum
		material_index = MATERIAL_INDEX_MAXIMUM;
	}

	// Using bitfields to pack the data in a single 32 bits variable and still keep a nice syntax
	unsigned int priority : ODD_PARTIY_BITS;
	bool odd_parity : PRIORITY_BITS;
	bool topmost : TOPMOST_BITS;
	unsigned int material_index : MATERIAL_INDEX_BITS;
};

template <>
struct InteriorStackImpl<ISS_WITH_PRIORITIES>
{
	HIPRT_HOST_DEVICE bool push(int& incident_material_index, int& outgoing_material_index, bool& inside_material, int material_index, int material_priority)
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
			if (stack[last_entered_mat_index].material_index != material_index && stack[last_entered_mat_index].topmost && stack[last_entered_mat_index].odd_parity)
				break;

		// Parity of the material we're inserting in the stack
		bool odd_parity = true;
		// Index in the stack of the previous material that is the same as
		// the one we're trying to insert in the stack.
		int previous_same_mat_index;

		for (previous_same_mat_index = stack_position; previous_same_mat_index >= 0; previous_same_mat_index--)
		{
			if (stack[previous_same_mat_index].material_index == material_index)
			{
				// The previous stack entry of the same material is not the topmost anymore
				stack[previous_same_mat_index].topmost = false;
				// The current parity is the inverse of the previous one
				odd_parity = !stack[previous_same_mat_index].odd_parity;

				break;
			}
		}
		
		inside_material = !odd_parity;

		// Inserting the material in the stack
		if (stack_position < NestedDielectricsStackSize - 1)
			stack_position++;
		stack[stack_position].material_index = material_index;
		stack[stack_position].odd_parity = odd_parity;
		stack[stack_position].topmost = true;
		stack[stack_position].priority = material_priority;

		if (material_priority < stack[last_entered_mat_index].priority)
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
				incident_material_index = stack[last_entered_mat_index].material_index;
				outgoing_material_index = material_index;
			}
			else
			{
				// Exiting material
				incident_material_index = material_index;
				outgoing_material_index = stack[last_entered_mat_index].material_index;
			}

			// Not skipping the boundary
			return false;
		}
	}

	HIPRT_HOST_DEVICE void pop(const bool inside_material)
	{
		int stack_top_mat_index = stack[stack_position].material_index;
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
				if (stack[previous_same_mat_index].material_index == stack_top_mat_index)
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
			if (stack[i].material_index == stack_top_mat_index)
			{
				stack[i].topmost = true;
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
