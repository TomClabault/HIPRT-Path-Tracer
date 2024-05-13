/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RAY_PAYLOAD_H
#define RAY_PAYLOAD_H

#include "HostDeviceCommon/Color.h"

enum RayState
{
	BOUNCE,
	MISSED
};

#define INTERIOR_STACK_SIZE 8

/**
 * References:
 * 
 * [1] [Ray Tracing Gems 1 - Automatic Handling of Materials in Nested Volumes] https://www.realtimerendering.com/raytracinggems/rtg/index.html
 */
struct StackEntry
{
	// TODO do packing in there
	bool topmost = true;
	bool odd_parity = true;

	int material_index = -1;
};

struct InteriorStack
{
	HIPRT_HOST_DEVICE void push(int& incident_material_index, int& outgoing_material_index, bool& leaving_material, int material_index)
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
		// material we're currently inserting in the stack
		int last_entered_mat_index = 0;
		for (last_entered_mat_index = stack_position; last_entered_mat_index >= 0; last_entered_mat_index--)
			if (stack[last_entered_mat_index].material_index != material_index && stack[last_entered_mat_index].topmost && stack[last_entered_mat_index].odd_parity)
				break;

		// Inserting the material in the stack
		if (stack_position < INTERIOR_STACK_SIZE - 1)
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
				incident_material_index = outgoing_material_index;
		}

		leaving_material = !odd_parity;
	}

	HIPRT_HOST_DEVICE void pop(bool leaving_material)
	{
		int stack_top_mat_index = stack[stack_position].material_index;
		stack_position--;

		if (leaving_material)
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

	StackEntry stack[INTERIOR_STACK_SIZE];

	int stack_position = 0;
};

struct RayPayload
{
	// Energy left in the ray after it bounces around the scene
	ColorRGB throughput = ColorRGB(1.0f);
	// Final color of the ray
	ColorRGB ray_color = ColorRGB(0.0f);
	// Camera ray is "Bounce" to give it a chance to hit the scene
	RayState next_ray_state = RayState::BOUNCE;
	// Type of BRDF found at the last intersection
	BRDF last_brdf_hit_type = BRDF::Uninitialized;
	// How far has the ray traveled in the current volume.
	float distance_in_volume = 0.0f;
	// The stack of materials being traversed. Used for nested dielectrics handling
	InteriorStack interior_stack;
	// Indices of the material we were in before hitting the current dielectric surface
	int incident_mat_index = -1, outgoing_mat_index = -1;
	// Whether or not we're exiting a material
	bool leaving_mat = false;

	HIPRT_HOST_DEVICE bool is_inside_volume() const
	{
		return interior_stack.stack_position > 0;
	}
	
	HIPRT_HOST_DEVICE bool is_leaving_volume() const
	{
		return leaving_mat;
	}
};

#endif
