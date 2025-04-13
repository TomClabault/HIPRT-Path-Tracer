/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_NESTED_DIELECTRICS_H
#define DEVICE_NESTED_DIELECTRICS_H

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"

#include <hiprt/hiprt_common.h>

#ifdef __KERNELCC__
// On the GPU, the nested dielectrics stack is allocated in shared memory.
// This means that all the entries of the nested dielectrics stacks are in shared memory.
//
// For example, for thread blocks of 64 and a NestedDielectricStackSize of 3, this gives us
// a shared memory array of 3*64 = 192 entries.
// 
// We then need a mapping that "redirects" each thread to its proper entry in that 192-long array.
// 
// That's what this macro does, it takes an index in the stack as parameter (so 0, 1 or 2 for a NestedDielectricStackSize of 3)
// and maps it to the index to use in the shared memory array by using the threadIdx.
// 
// Note that the mapping is written to minimize shared memory bank conflicts
#if NestedDielectricsStackUseSharedMemory == KERNEL_OPTION_TRUE
// Only using the special mapping if we're using shared memory
#define NESTED_DIELECTRICS_STACK_INDEX_SHIFT(index) (index * KernelWorkgroupThreadCount + (blockDim.x * threadIdx.y + threadIdx.x))
#else
// If we're not using shared memory, the nested dielectrics stack is simply going to be in an
// array member of the NestedDielectricsInteriorStack structure so we can just index that array 
#define NESTED_DIELECTRICS_STACK_INDEX_SHIFT(x) (x)
#endif
#else
// This macro is used to offset the index used to index the priority stack.
// On the CPU, there is nothing to do, just use the given index, there is really nothing
// special. The special case is for the GPU, explained above the GPU macro definition
#define NESTED_DIELECTRICS_STACK_INDEX_SHIFT(x) (x)
#endif

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
	// This 'MATERIAL_INDEX_MAXIMUM' is just an alias basically
	static constexpr unsigned int MATERIAL_INDEX_MAXIMUM = MATERIAL_INDEX_BIT_MASK;

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

	HIPRT_HOST_DEVICE int get_priority() const { return (packed_data >> PRIORITY_BIT_SHIFT) & PRIORITY_BIT_MASK; }
	HIPRT_HOST_DEVICE bool get_topmost() const { return (packed_data >> TOPMOST_BIT_SHIFT) & TOPMOST_BIT_MASK; }
	HIPRT_HOST_DEVICE bool get_odd_parity() const { return (packed_data >> ODD_PARTIY_BIT_SHIFT) & ODD_PARTIY_BIT_MASK; }
	HIPRT_HOST_DEVICE int get_material_index() const { return (packed_data >> MATERIAL_INDEX_BIT_SHIFT) & MATERIAL_INDEX_BIT_MASK; }

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

#ifdef __KERNELCC__
#if NestedDielectricsStackUseSharedMemory == KERNEL_OPTION_TRUE
// Only declare this shared memory stack_entries on the GPU if shared memory
// is used for the nested dielectrics stack
__shared__ StackPriorityEntry stack_entries[NestedDielectricsStackSize * KernelWorkgroupThreadCount];
#endif
#endif

struct NestedDielectricsInteriorStack
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
		if (stack_position == NestedDielectricsStackSize - 1)
			// The stack is already at the maximum
			return false;
			
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
			if (stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(last_entered_mat_index)].get_material_index() != material_index 
			 && stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(last_entered_mat_index)].get_topmost()
			 && stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(last_entered_mat_index)].get_odd_parity())
				break;

		// Parity of the material we're inserting in the stack
		bool odd_parity = true;
		// Index in the stack of the previous material that is the same as
		// the one we're trying to insert in the stack.
		int previous_same_mat_index;
		for (previous_same_mat_index = stack_position; previous_same_mat_index >= 0; previous_same_mat_index--)
		{
			int stack_index = NESTED_DIELECTRICS_STACK_INDEX_SHIFT(previous_same_mat_index);
			if (stack_entries[stack_index].get_material_index() == material_index)
			{
				// The previous stack entry of the same material is not the topmost anymore
				stack_entries[stack_index].set_topmost(false);
				// The current parity is the inverse of the previous one
				odd_parity = !stack_entries[stack_index].get_odd_parity();

				break;
			}
		}
		
		out_inside_material = !odd_parity;

		// Inserting the material in the stack
		if (stack_position < NestedDielectricsStackSize - 1)
			stack_position++;

		int new_stack_index = NESTED_DIELECTRICS_STACK_INDEX_SHIFT(stack_position);
		stack_entries[new_stack_index].set_material_index(material_index);
		stack_entries[new_stack_index].set_odd_parity(odd_parity);
		stack_entries[new_stack_index].set_topmost(true);
		stack_entries[new_stack_index].set_priority(material_priority);

		if (material_priority < stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(last_entered_mat_index)].get_priority())
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
				out_incident_material_index = stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(last_entered_mat_index)].get_material_index();
				out_outgoing_material_index = material_index;
			}
			else
			{
				// Exiting material
				out_incident_material_index = material_index;
				out_outgoing_material_index = stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(last_entered_mat_index)].get_material_index();
			}

			// Not skipping the boundary
			return false;
		}
	}

	HIPRT_HOST_DEVICE void pop(const bool inside_material)
	{
		int stack_top_mat_index = stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(stack_position)].get_material_index();
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
				if (stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(previous_same_mat_index)].get_material_index() == stack_top_mat_index)
					break;

			if (previous_same_mat_index >= 0)
				for (int i = previous_same_mat_index + 1; i <= stack_position; i++)
					stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(i - 1)] = stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(i)];

			// For very small stacks (2 for example), we may not be able to pop twice
			// at all so we check the position on the stack first
			if (stack_position > 0)
				stack_position--;
		}

		for (int i = stack_position; i >= 0; i--)
		{
			if (stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(i)].get_material_index() == stack_top_mat_index)
			{
				stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(i)].set_topmost(true);
				break;
			}
		}
	}

	// We only need all of this if the stack size is actually > 0,
	// otherwise, we're just not going to do the nested dielectrics handling at all

	// Declaring the nested dielectric stack as a member of the structure if we're on
	// the CPU or if we're not using the shared memory stack on the GPU
#ifdef __KERNELCC__
#if NestedDielectricsStackUseSharedMemory == KERNEL_OPTION_FALSE
	StackPriorityEntry stack_entries[NestedDielectricsStackSize];
#endif
#else
	StackPriorityEntry stack_entries[NestedDielectricsStackSize];
#endif

	static constexpr unsigned int MAX_MATERIAL_INDEX = StackPriorityEntry::MATERIAL_INDEX_MAXIMUM;

	// Stack position is pointing at the last valid entry.
	// Entry 0 is always present and represent air basically
	int stack_position = 0;
};

#endif
