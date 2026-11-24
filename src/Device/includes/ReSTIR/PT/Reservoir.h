/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_RESERVOIR_H
#define DEVICE_RESTIR_PT_RESERVOIR_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"
#include "Device/includes/RayVolumeState.h"

#include "HostDeviceCommon/Material/MaterialPacked.h"
#include "HostDeviceCommon/Xorshift.h"

#ifndef __KERNELCC__
#include "Utils/Utils.h"

// For multithreaded console error logging on the CPU if NaNs are detected
#include <mutex>
static std::mutex restir_pt_log_mutex;
#endif // #ifndef __KERNELCC__

struct ReSTIRPTReservoirSample
{
	// 'rc_vertex' in this reservoir is x1 (with x0 the camera), i.e. the gbuffer hit
	float3_t rc_vertex = make_float3(-1.0f, -1.0f, -1.0f);
	Octahedral24BitNormalPadded32b rc_vertex_geometric_normal;
	float3_t rc_vertex_incident_light_direction = make_float3(-1.0f, -1.0f, -1.0f);
	Octahedral24BitNormalPadded32b rc_vertex_shading_normal;
	// Radiance arriving at the reconnection vertex from 'rc_vertex_incident_light_direction', with the bounces after the reconnection vertex
	// weighted by their BSDF PDF (an estimate of that radiance). The last BSDF * cosine toward the light is not divided by its PDF
	ColorRGB32F rc_vertex_incident_radiance;
	int rc_vertex_primitive_index = -1;

	float rc_vertex_texcoords_u = -1.0f;
	float rc_vertex_texcoords_v = -1.0f;

	float bsdf_throughput_luminance_at_sample_point = 1.0f;
	float target_function							= 0.0f;
	// Reuse the initial ray's alpha decisions when testing this sample from another pixel.
	unsigned int visible_to_sample_point_alpha_test_random_seed = 42;

	// TODO all 'at visible' point variables should be replaced by 'rc_vertex' variables
	BSDFIncidentLightInfo incident_light_info_at_visible_point = BSDFIncidentLightInfo::NO_INFO;
	BSDFIncidentLightInfo incident_light_info_at_sample_point  = BSDFIncidentLightInfo::NO_INFO;

	// Index of the pixel that produced this sample/reservoir during the initial candidates sampling
	// Used by some algorithms such as ReSTIR PG
	unsigned int pixel_index = static_cast<unsigned int>(-1);

	bool di_sample = false;

	// Flags of the target function BSDF approximation of the union domain reuse (SpatialTargetFunctionBSDFApproximation.h), 0 when the approximation is
	// not used: bit 0 = the reconnection vertex material is in an analytic class and the sample conditions of
	// ReSTIR_PT_target_function_bsdf_approximation_sample_flags() hold (the only bit of a DI sample), bit 1 = the analytic lobe of the reconnection vertex
	// is perfectly smooth, bits 2..7 = the 6-bit Fresnel code of that lobe. The byte is non-zero exactly when bit 0 is set
	unsigned char target_function_bsdf_approximation_flags = 0;
	// Analytic lobe of the reconnection vertex for the principled BSDF (ReSTIR_PT_target_function_bsdf_approximation_encode_sample_lobe()): alpha code and
	// specular_scale code. The fresnel_attenuated_diffuse code is in the padding byte of 'rc_vertex_shading_normal' and the unattenuated_diffuse code
	// (bits 0..6) + the metal bit (bit 7) in the padding byte of 'rc_vertex_geometric_normal'. These three bytes fill the struct padding after 'di_sample':
	// no size change (static_assert below)
	unsigned char target_function_bsdf_approximation_lobe_alpha	   = 0;
	unsigned char target_function_bsdf_approximation_lobe_specular = 0;

	HIPRT_DEVICE bool is_envmap_path() const
	{
		return rc_vertex_primitive_index == -1;
	}
};

// The target function BSDF approximation fields fill the padding of the sample, its size must not change
static_assert(sizeof(ReSTIRPTReservoirSample) == 84, "The size of ReSTIRPTReservoirSample changed");

struct ReSTIRPTReservoir
{
	HIPRT_DEVICE void add_one_candidate(ReSTIRPTReservoirSample new_sample, float weight, Xorshift32Generator& random_number_generator)
	{
		confidence++;

		weight_sum += weight;
		if (random_number_generator() < weight / weight_sum)
			sample = new_sample;
	}

	/**
	 * Combines 'other_reservoir' into this reservoir
	 *
	 * 'target_function' is the target function evaluated at the pixel that is doing the
	 *      resampling with the sample from the reservoir that we're combining (which is 'other_reservoir')
	 *
	 * 'jacobian_determinant' is the determinant of the jacobian. It is used
	 *      for converting the solid angle PDF (or UCW since the UCW is an estimate
	 * of the PDF) with respect to the shading point of the reservoir we're resampling to the solid angle PDF with respect to the shading point of 'this'
	 * reservoir
	 *
	 * 'random_number_generator' for generating the random number that will be used to stochastically
	 *      select the sample from 'other_reservoir' or not
	 */
	HIPRT_DEVICE bool combine_with(const ReSTIRPTReservoir& other_reservoir,
								   float mis_weight,
								   float target_function,
								   float jacobian_determinant,
								   Xorshift32Generator& random_number_generator)
	{
		// Bullet point 4. of the intro of Section 5.2 of [A Gentle Introduction to ReSTIR: Path Reuse in Real-time] https://intro-to-restir.cwyman.org/
		float reservoir_resampling_weight = mis_weight * target_function * other_reservoir.UCW * jacobian_determinant;

		weight_sum += reservoir_resampling_weight;
		confidence += other_reservoir.confidence;

		if (random_number_generator() < reservoir_resampling_weight / weight_sum)
		{
			sample				   = other_reservoir.sample;
			sample.target_function = target_function;

			return true;
		}

		return false;
	}

	HIPRT_DEVICE void end()
	{
		if (weight_sum == 0.0f)
			UCW = 0.0f;
		else
			UCW = 1.0f / sample.target_function * weight_sum;
	}

	HIPRT_DEVICE void end_with_normalization(float normalization_numerator, float normalization_denominator)
	{
		// Checking some limit values
		if (weight_sum == 0.0f || weight_sum > 1.0e10f || normalization_denominator == 0.0f || normalization_numerator == 0.0f)
			UCW = 0.0f;
		else
			UCW = 1.0f / sample.target_function * weight_sum * normalization_numerator / normalization_denominator;

		// Hard limiting confidence to avoid explosions if the user decides not to use any confidence-cap (confidence-cap == 0)
		confidence = hippt::min(confidence, 1000000);
	}

	HIPRT_DEVICE void sanity_check(int2_t pixel_coords)
	{
#ifndef __KERNELCC__
		if (confidence < 0)
		{
			std::lock_guard<std::mutex> lock(restir_pt_log_mutex);
			std::cerr << "Negative reservoir confidence value at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << confidence << std::endl;
			Debug::debugbreak();
		}
		else if (std::isnan(weight_sum) || std::isinf(weight_sum))
		{
			std::lock_guard<std::mutex> lock(restir_pt_log_mutex);
			std::cerr << "NaN or inf reservoir weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
			Debug::debugbreak();
		}
		else if (weight_sum < 0)
		{
			std::lock_guard<std::mutex> lock(restir_pt_log_mutex);
			std::cerr << "Negative reservoir weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << weight_sum << std::endl;
			Debug::debugbreak();
		}
		else if (std::abs(weight_sum) < std::numeric_limits<float>::min() && weight_sum != 0.0f)
		{
			std::lock_guard<std::mutex> lock(restir_pt_log_mutex);
			std::cerr << "Denormalized weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << weight_sum << std::endl;
			Debug::debugbreak();
		}
		else if (std::isnan(UCW) || std::isinf(UCW))
		{
			std::lock_guard<std::mutex> lock(restir_pt_log_mutex);
			std::cerr << "NaN or inf reservoir UCW at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
			Debug::debugbreak();
		}
		else if (UCW < 0)
		{
			std::lock_guard<std::mutex> lock(restir_pt_log_mutex);
			std::cerr << "Negative reservoir UCW at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << UCW << std::endl;
			Debug::debugbreak();
		}
		else if (std::isnan(sample.target_function) || std::isinf(sample.target_function))
		{
			std::lock_guard<std::mutex> lock(restir_pt_log_mutex);
			std::cerr << "NaN or inf reservoir sample.target_function at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
			Debug::debugbreak();
		}
		else if (sample.target_function < 0)
		{
			std::lock_guard<std::mutex> lock(restir_pt_log_mutex);
			std::cerr << "Negative reservoir sample.target_function at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << sample.target_function
					  << std::endl;
			Debug::debugbreak();
		}
#else  // #ifndef __KERNELCC__
		(void)pixel_coords;
#endif // #ifndef __KERNELCC__
	}

	ReSTIRPTReservoirSample sample;

	int confidence = 0;
	// TODO weight sum is never used at the same time as UCW so only one variable can be used for both to save space
	float weight_sum = 0.0f;
	// If the UCW is set to -1, this is because the reservoir was killed by visibility reuse
	float UCW = 0.0f;
};

/**
 * Initial candidate generation does not read the selected sample while streaming. Keeping the sample in the final output slot prevents the complete sample from
 * remaining live across path traversal and material evaluation. Only the running weight sum stays local to the kernel.
 */
struct ReSTIRPTInitialCandidatesReservoir
{
	HIPRT_DEVICE ReSTIRPTInitialCandidatesReservoir(ReSTIRPTReservoir* output_reservoirs, unsigned int pixel_index)
		: output_reservoirs(output_reservoirs), pixel_index(pixel_index)
	{
	}

	HIPRT_DEVICE void add_one_candidate(const ReSTIRPTReservoirSample& new_sample, float weight, Xorshift32Generator& random_number_generator)
	{
		weight_sum += weight;
		if (random_number_generator() < weight / weight_sum)
			get_output_reservoir()->sample = new_sample;
	}

	HIPRT_DEVICE void end_with_normalization(float normalization_numerator, float normalization_denominator)
	{
		ReSTIRPTReservoir* output_reservoir = get_output_reservoir();

		output_reservoir->confidence = 1;
		output_reservoir->weight_sum = weight_sum;

		if (weight_sum == 0.0f)
			// Preserve the default sample written by a local reservoir when no candidate was selected.
			output_reservoir->sample = ReSTIRPTReservoirSample();

		if (weight_sum == 0.0f || weight_sum > 1.0e10f || normalization_denominator == 0.0f || normalization_numerator == 0.0f)
			output_reservoir->UCW = 0.0f;
		else
			output_reservoir->UCW = 1.0f / output_reservoir->sample.target_function * weight_sum * normalization_numerator / normalization_denominator;
	}

	HIPRT_DEVICE void sanity_check(int2_t pixel_coords)
	{
#ifndef __KERNELCC__
		ReSTIRPTReservoir validation_reservoir;
		if (weight_sum > 0.0f)
			validation_reservoir.sample = get_output_reservoir()->sample;
		validation_reservoir.weight_sum = weight_sum;
		validation_reservoir.sanity_check(pixel_coords);
#endif // #ifndef __KERNELCC__
	}

	/**
	 * Address of the output reservoir of the pixel, computed again at each use. Computed once, the address would be kept in two registers across
	 * the whole path of the initial candidates kernel
	 */
	HIPRT_DEVICE ReSTIRPTReservoir* get_output_reservoir() const
	{
		return &output_reservoirs[hippt::hide_from_optimizer(pixel_index)];
	}

	// Output reservoirs of all the pixels
	ReSTIRPTReservoir* output_reservoirs;
	unsigned int pixel_index;
	float weight_sum = 0.0f;
};

#endif // #ifndef DEVICE_RESTIR_PT_RESERVOIR_H
