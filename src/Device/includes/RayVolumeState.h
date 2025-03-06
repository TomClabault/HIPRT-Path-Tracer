/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RAY_VOLUME_STATE_H
#define DEVICE_RAY_VOLUME_STATE_H

#include "Device/includes/NestedDielectrics.h"
// Including dispersion for sampling a wavelength in the reconstruction of the first hit of RayVolumeState
#include "Device/includes/Dispersion.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"

struct RayVolumeState
{
	/**
	 * On the GPU, it is necessary that the RayVolumeState is initialized manually as opposed to in a default constructor for example.
	 * That's because the nested dielectrics stack is in shared memory and is thus a "global variable". 
	 * 
	 * If it were to be initialized in the RayVolumeState constructor, every declaration of a RayVolumeState variable
	 * would call the constructor and reinitialize the whole nested dielectrics stack.
	 */
	HIPRT_HOST_DEVICE RayVolumeState()
	{
#ifdef __KERNELCC__
		// On the CPU, the priority stack is a "global" variable because it is
		// in shared memory
#if NestedDielectricsStackUseSharedMemory == KERNEL_OPTION_TRUE
		// 'stack_entries' is in shared memory here
#define stack_variable stack_entries
#else
#define stack_variable interior_stack.stack_entries
#endif
#else
		// On the CPU, the priority stack is a member of the interior stack
#define stack_variable interior_stack.stack_entries
#endif

		for (int i = 0; i < NestedDielectricsStackSize; i++)
		{
			stack_variable[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(i)].set_priority(0);
			stack_variable[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(i)].set_odd_parity(true);
			stack_variable[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(i)].set_topmost(true);
			// Setting the material index to the maximum
			stack_variable[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(i)].set_material_index(NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX);
		}
	}

	HIPRT_HOST_DEVICE void reconstruct_first_hit(const DeviceUnpackedEffectiveMaterial& material, int* material_indices_buffer, int primitive_index, Xorshift32Generator& random_number_generator)
	{
		if (primitive_index == -1)
			// No primary hit i.e. straight into the envmap
			return;

		int mat_index = material_indices_buffer[primitive_index];

		interior_stack.push(
			incident_mat_index,
			outgoing_mat_index,
			inside_material,
			mat_index,
			material.get_dielectric_priority());

		if (material.dispersion_scale > 0.0f && material.specular_transmission > 0.0f && sampled_wavelength == 0.0f)
			// If we hit a dispersive material, we sample the wavelength that will be used
			// for computing the wavelength dependent IORs used for dispersion
			//
			// We're also not re-doing the sampling if a wavelength has already been sampled for that path
			//
			// Negating the wavelength to indicate that the throughput filter of the wavelength
			// hasn't been applied yet (applied in principled_glass_eval())
			sampled_wavelength = -sample_wavelength_uniformly(random_number_generator);
	}

	/**
	 * This function returns true if we're currently hitting an object from the outside.
	 * Returns false if we're hitting an object from the inside
	 *
	 * This function only returns valid results if called after the call to trace_ray()
	 * in the main ray tracing loop.
	 *
	 * Said otherwise, this function must only be called after tracing a camera ray
	 * with trace_ray() or tracing a bounce ray with trace_ray() but in particular,
	 * this function should *not* be called after sampling the indirect ray bounce
	 * with bsdf_sample()
	 */
	HIPRT_HOST_DEVICE bool after_trace_ray_is_outside_object() const
	{
		// Because this function must only be called after a call to trace_ray, we know
		// that the last intersected material is on top of the stack.
		// If the material on top of the stack is topmost, then this means that we're
		// entering this material so we're outside of it
		return interior_stack.stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(interior_stack.stack_position)].get_topmost();
	}

	/**
	 * Same as 'after_trace_ray_is_outside_object' but should be used from inside the trace_ray() function
	 * before the intersected material index is pushed onto the nested dielectrics stack
	 */
	HIPRT_HOST_DEVICE bool before_trace_ray_is_outside_object() const
	{
		// Because this function must only be called after a call to trace_ray, we know
		// that the last intersected material is on top of the stack.
		// If the material on top of the stack is topmost, then this means that we're
		// entering this material so we're outside of it
		return interior_stack.stack_entries[NESTED_DIELECTRICS_STACK_INDEX_SHIFT(interior_stack.stack_position)].get_topmost();
	}

	// How far has the ray traveled in the current volume.
	float distance_in_volume = 0.0f;
	// The stack of materials being traversed. Used for nested dielectrics handling
	NestedDielectricsInteriorStack interior_stack;
	// Indices of the material we were in before hitting the current dielectric surface
	int incident_mat_index = -1, outgoing_mat_index = -1;
	// Whether or not we're exiting a material
	bool inside_material = false;

	// For spectral dispersion. A random wavelength is sampled and replaces this value
	// when a glass object is hit. This wavelength can then be used to determine the IOR
	// that should be used for refractions/reflections on the dielectric object. 
	// 
	// The wavelength is also used to apply a throughput filter on the ray such that only the
	// sampled wavelength's color travels around the scene.
	//
	// If this value is negative, this is because the ray throughput filter hasn't been applied
	// yet. If the value is positive, the filter has been applied
	float sampled_wavelength = 0.0f;
};

#endif
