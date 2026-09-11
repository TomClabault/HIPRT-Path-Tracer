/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_CORE_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_CORE_DEVICE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDirectIlluminationTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeIlluminationSignature.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSpatialSampleMoments.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeUserSettings.h"
#include "Device/includes/PathGuiding/VMF.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "HostDeviceCommon/Xorshift.h"

#include <cstdint>

// Indexed by sqrtf(1.0f / effectiveKappa) to get the cosine of the maximum angle allowed between two distributions VMF for the mean radiance weighted
// directions split criterion.
HIPRT_DEVICE __constant__ inline float COSINE_MAX_ANGLE_DIRECTION_LUT[256] = {
	0.998629535f,	0.997953926f,	0.997154292f,	0.996233062f,	0.995192291f,	0.99403378f,	0.992759144f,	 0.991369859f,	 0.9898673f,
	0.988252757f,	0.986527462f,	0.984692599f,	0.982749322f,	0.980698764f,	0.978542053f,	0.976280316f,	 0.973914694f,	 0.971446348f,
	0.968876455f,	0.966206218f,	0.963436856f,	0.960569607f,	0.957605723f,	0.954546462f,	0.951393087f,	 0.948146858f,	 0.944809033f,
	0.941380858f,	0.93786357f,	0.934258389f,	0.93056652f,	0.92678915f,	0.922927445f,	0.918982553f,	 0.914955598f,	 0.910847684f,
	0.906659895f,	0.902393291f,	0.898048912f,	0.893627777f,	0.889130882f,	0.884559206f,	0.879913705f,	 0.875195318f,	 0.870404963f,
	0.86554354f,	0.860611931f,	0.855611002f,	0.8505416f,		0.845404556f,	0.840200687f,	0.834930791f,	 0.829595655f,	 0.824196048f,
	0.818732727f,	0.813206436f,	0.807617905f,	0.801967849f,	0.796256975f,	0.790485975f,	0.78465553f,	 0.77876631f,	 0.772818975f,
	0.766814172f,	0.760752541f,	0.754634709f,	0.748461295f,	0.742232908f,	0.735950148f,	0.729613607f,	 0.723223867f,	 0.716781503f,
	0.71028708f,	0.703741157f,	0.697144285f,	0.690497006f,	0.683799856f,	0.677053363f,	0.67025805f,	 0.663414431f,	 0.656523014f,
	0.649584301f,	0.642598789f,	0.635566965f,	0.628489315f,	0.621366314f,	0.614198437f,	0.606986148f,	 0.59972991f,	 0.592430177f,
	0.5850874f,		0.577702025f,	0.570274492f,	0.562805237f,	0.55529469f,	0.547743279f,	0.540151424f,	 0.532519544f,	 0.52484805f,
	0.517137352f,	0.509387854f,	0.501599957f,	0.493774056f,	0.485910545f,	0.478009811f,	0.47007224f,	 0.462098213f,	 0.454088107f,
	0.446042296f,	0.437961149f,	0.429845035f,	0.421694317f,	0.413509354f,	0.405290503f,	0.397038118f,	 0.388752549f,	 0.380434144f,
	0.372083245f,	0.363700195f,	0.355285331f,	0.346838988f,	0.338361497f,	0.329853187f,	0.321314385f,	 0.312745412f,	 0.304146588f,
	0.295518231f,	0.286860654f,	0.278174168f,	0.26945908f,	0.260715696f,	0.251944316f,	0.243145239f,	 0.234318761f,	 0.225465173f,
	0.216584765f,	0.207677822f,	0.198744626f,	0.189785455f,	0.180800586f,	0.171790289f,	0.162754833f,	 0.153694482f,	 0.144609497f,
	0.135500134f,	0.126366645f,	0.117209279f,	0.108028282f,	0.0988238918f,	0.0895963455f,	0.0803458743f,	 0.0710727049f,	 0.0617770597f,
	0.0524591563f,	0.0431192073f,	0.0337574206f,	0.0243739993f,	0.014969141f,	0.00554303871f, -0.00390412019f, -0.0133721534f, -0.022860884f,
	-0.0323701405f, -0.0418997569f, -0.0514495726f, -0.0610194328f, -0.0706091882f, -0.0802186952f, -0.0898478158f,	 -0.0994964179f, -0.109164375f,
	-0.118851567f,	-0.128557878f,	-0.1382832f,	-0.148027429f,	-0.157790469f,	-0.167572227f,	-0.177372617f,	 -0.18719156f,	 -0.197028982f,
	-0.206884813f,	-0.21675899f,	-0.226651457f,	-0.236562162f,	-0.246491059f,	-0.256438107f,	-0.26640327f,	 -0.276386519f,	 -0.28638783f,
	-0.296407182f,	-0.306444563f,	-0.316499962f,	-0.326573375f,	-0.336664804f,	-0.346774254f,	-0.356901735f,	 -0.367047262f,	 -0.377210856f,
	-0.387392539f,	-0.397592341f,	-0.407810293f,	-0.418046433f,	-0.428300801f,	-0.438573442f,	-0.448864405f,	 -0.459173743f,	 -0.46950151f,
	-0.479847767f,	-0.490212576f,	-0.500596004f,	-0.510998121f,	-0.521418998f,	-0.531858711f,	-0.54231734f,	 -0.552794964f,	 -0.563291669f,
	-0.57380754f,	-0.584342667f,	-0.59489714f,	-0.605471055f,	-0.616064506f,	-0.626677592f,	-0.637310412f,	 -0.647963069f,	 -0.658635667f,
	-0.66932831f,	-0.680041105f,	-0.690774163f,	-0.701527591f,	-0.712301503f,	-0.723096011f,	-0.733911228f,	 -0.74474727f,	 -0.755604253f,
	-0.766482294f,	-0.777381512f,	-0.788302024f,	-0.799243952f,	-0.810207415f,	-0.821192534f,	-0.832199431f,	 -0.843228228f,	 -0.854279047f,
	-0.865352012f,	-0.876447246f,	-0.887564872f,	-0.898705015f,	-0.909867797f,	-0.921053344f,	-0.932261779f,	 -0.943493227f,	 -0.954747812f,
	-0.966025658f,	-0.977326888f,	-0.988651628f,
};

struct IlluminationAwareKDTreeCoreDevice
{
	static constexpr float DIRECTION_LUT_MAX_U		  = 0.802656898f;
	static constexpr float Z_SCORE_1_MINUS_1E_MINUS_4 = 3.7190165f;
	// Keep unresolved distinct from INVALID_NODE_INDEX so replay can resolve a sample lazily when needed.
	static constexpr unsigned int UNRESOLVED_TRAINING_SAMPLE_GUIDING_NODE_INDEX =
		IlluminationAwareKDTreeDirectIlluminationTrainingSample::UNRESOLVED_GUIDING_NODE_INDEX;

	static constexpr float FLOAT_RESULTANT_LENGTH_BELOW_ONE = 0.99999994f;

	/**
	 * Normal difference formula below "We decide to split if both cells have at least 1000 samples(Sec. 4.1) and..." in Appendix A.
	 */
	HIPRT_DEVICE static bool normal_difference_exceeds_threshold_fp32(const IlluminationAwareKDTreeIlluminationSignature& first,
																	  float first_coefficient,
																	  const IlluminationAwareKDTreeIlluminationSignature& second,
																	  float second_coefficient)
	{
		float first_count  = static_cast<float>(first.valid_observation_count);
		float second_count = static_cast<float>(second.valid_observation_count);
		float first_mean   = first_coefficient * first.scalar_radiance_sum / first_count;
		float second_mean  = second_coefficient * second.scalar_radiance_sum / second_count;
		float difference   = first_mean - second_mean;
		if (!hippt::is_finite(first_mean) || !hippt::is_finite(second_mean) || !hippt::is_finite(difference) || !(difference > 0.0f))
			return false;

		float first_numerator	 = first_count * first.squared_scalar_radiance_sum - first.scalar_radiance_sum * first.scalar_radiance_sum;
		float second_numerator	 = second_count * second.squared_scalar_radiance_sum - second.scalar_radiance_sum * second.scalar_radiance_sum;
		first_numerator			 = first_numerator > 0.0f ? first_numerator : 0.0f;
		second_numerator		 = second_numerator > 0.0f ? second_numerator : 0.0f;
		float first_denominator	 = first_count * first_count * first_count;
		float second_denominator = second_count * second_count * second_count;
		float variance			 = first_coefficient * first_coefficient * first_numerator / first_denominator +
						 second_coefficient * second_coefficient * second_numerator / second_denominator;
		if (!hippt::is_finite(variance))
			return false;
		if (variance <= 1.0e-30f)
			return true;

		float squared_difference		 = difference * difference;
		float squared_threshold_variance = Z_SCORE_1_MINUS_1E_MINUS_4 * Z_SCORE_1_MINUS_1E_MINUS_4 * variance;
		if (!hippt::is_finite(squared_difference) || !hippt::is_finite(squared_threshold_variance))
			return difference / hippt::sqrt(variance) > Z_SCORE_1_MINUS_1E_MINUS_4;
		return squared_difference > squared_threshold_variance;
	}

	HIPRT_DEVICE bool should_split_samples(const IlluminationAwareKDTreeIlluminationSignature& guiding_signature_float)
	{
		return guiding_signature_float.valid_observation_count >= static_cast<float>(user_settings.minimum_sample_count_for_splitting);
	}

	HIPRT_DEVICE bool should_split_mean_radiance(const IlluminationAwareKDTreeIlluminationSignature& guiding_signature_float,
												 const IlluminationAwareKDTreeIlluminationSignature& lookahead_signature_float)
	{
		if (lookahead_signature_float.valid_observation_count < static_cast<float>(user_settings.minimum_sample_count_for_splitting))
			return false;

		IlluminationAwareKDTreeIlluminationSignature difference_cell{};
		difference_cell.valid_observation_count = guiding_signature_float.valid_observation_count > lookahead_signature_float.valid_observation_count
													  ? guiding_signature_float.valid_observation_count - lookahead_signature_float.valid_observation_count
													  : 0u;
		difference_cell.scalar_radiance_sum		= guiding_signature_float.scalar_radiance_sum - lookahead_signature_float.scalar_radiance_sum > 0.0f
													  ? guiding_signature_float.scalar_radiance_sum - lookahead_signature_float.scalar_radiance_sum
													  : 0.0f;
		difference_cell.squared_scalar_radiance_sum =
			guiding_signature_float.squared_scalar_radiance_sum - lookahead_signature_float.squared_scalar_radiance_sum > 0.0f
				? guiding_signature_float.squared_scalar_radiance_sum - lookahead_signature_float.squared_scalar_radiance_sum
				: 0.0f;
		if (difference_cell.valid_observation_count < static_cast<float>(user_settings.minimum_sample_count_for_splitting))
			return false;

		float guiding_sample_count			  = static_cast<float>(guiding_signature_float.valid_observation_count);
		float lookahead_sample_count		  = static_cast<float>(lookahead_signature_float.valid_observation_count);
		float threshold						  = user_settings.mean_radiance_split_threshold;
		float positive_difference_coefficient = (1.0f - threshold) * (guiding_sample_count - lookahead_sample_count);
		float positive_lookahead_coefficient  = guiding_sample_count - (1.0f - threshold) * lookahead_sample_count;
		if (normal_difference_exceeds_threshold_fp32(difference_cell, positive_difference_coefficient, lookahead_signature_float,
													 positive_lookahead_coefficient))
			return true;

		float negative_difference_coefficient = (1.0f + threshold) * (lookahead_sample_count - guiding_sample_count);
		float negative_lookahead_coefficient  = (1.0f + threshold) * lookahead_sample_count - guiding_sample_count;
		return normal_difference_exceeds_threshold_fp32(difference_cell, negative_difference_coefficient, lookahead_signature_float,
														negative_lookahead_coefficient);
	}

	HIPRT_DEVICE static float kappa_coth_kappa_minus_one_fp32(float kappa)
	{
		if (kappa < 1.0e-3f)
		{
			float kappa_squared = kappa * kappa;
			float kappa_fourth	= kappa_squared * kappa_squared;
			float kappa_sixth	= kappa_fourth * kappa_squared;

			return kappa_squared / 3.0f - kappa_fourth / 45.0f + 2.0f * kappa_sixth / 945.0f;
		}

		if (kappa > 20.0f)
			return kappa - 1.0f;

		return kappa / hippt::tanhf(kappa) - 1.0f;
	}

	HIPRT_DEVICE static VMF estimate_mean_direction_model(const IlluminationAwareKDTreeIlluminationSignature& signature)
	{
		VMF result{};
		result.sharpness = VMF::INVALID_SHARPNESS;

		float b1 = signature.scalar_radiance_sum;
		float b2 = signature.squared_scalar_radiance_sum;
		if (!hippt::is_finite(b1) || !hippt::is_finite(b2) || !hippt::is_finite(signature.weighted_direction_sum.x) ||
			!hippt::is_finite(signature.weighted_direction_sum.y) || !hippt::is_finite(signature.weighted_direction_sum.z) || !(b1 > 0.0f) || !(b2 > 0.0f))
			return result;

		float3_t direction_sum	   = signature.weighted_direction_sum;
		float direction_sum_length = hippt::length(direction_sum);
		if (!hippt::is_finite(direction_sum_length) || !(direction_sum_length > 1.0e-20f))
			return result;

		float resultant_length = direction_sum_length / b1;
		if (!hippt::is_finite(resultant_length))
			return result;

		// The exact value lies in [0,1]. Keep the clamp below one representable in FP32 so the denominator remains positive.
		if (resultant_length < 0.0f)
			resultant_length = 0.0f;
		if (resultant_length > FLOAT_RESULTANT_LENGTH_BELOW_ONE)
			resultant_length = FLOAT_RESULTANT_LENGTH_BELOW_ONE;

		float resultant_length_squared	   = resultant_length * resultant_length;
		float resultant_length_denominator = 1.0f - resultant_length_squared;
		if (!hippt::is_finite(resultant_length_denominator) || !(resultant_length_denominator > 0.0f))
			return result;

		float underlying_concentration = resultant_length * (3.0f - resultant_length_squared) / resultant_length_denominator;
		if (!hippt::is_finite(underlying_concentration) || !(underlying_concentration >= 0.0f))
			return result;

		float angular_term = kappa_coth_kappa_minus_one_fp32(underlying_concentration);
		if (!hippt::is_finite(angular_term))
			return result;

		// Eq. 4 of the paper
		float y			  = (b1 * b1 / b2) * angular_term + 1.0f;
		float y_squared	  = y * y;
		float y_cubed	  = y_squared * y;
		float numerator	  = y_cubed + 1.69934861f * y_squared + 5.38753272f * y + 9.85021305f;
		float denominator = y_squared + 0.67453491f * y + 4.31180006f;
		if (!hippt::is_finite(y) || !hippt::is_finite(y_squared) || !hippt::is_finite(y_cubed) || !hippt::is_finite(numerator) ||
			!hippt::is_finite(denominator) || !(denominator != 0.0f))
			return result;

		// Eq. 4 of the paper
		float concentration_argument = (y - 1.0f) * numerator / denominator;
		float concentration			 = hippt::sqrt(concentration_argument > 0.0f ? concentration_argument : 0.0f);
		if (!hippt::is_finite(concentration))
			return result;

		direction_sum /= direction_sum_length;
		if (!hippt::is_finite(direction_sum.x) || !hippt::is_finite(direction_sum.y) || !hippt::is_finite(direction_sum.z))
			return result;

		result.axis		 = direction_sum;
		result.sharpness = concentration;
		return result;
	}

	HIPRT_DEVICE bool should_split_mean_direction(const VMF& guiding_model, const IlluminationAwareKDTreeIlluminationSignature& lookahead_signature)
	{
		if (guiding_model.sharpness == VMF::INVALID_SHARPNESS)
			return false;

		// The guiding sample count is checked before its cached model is built; apply the same requirement to the lookahead.
		if (lookahead_signature.valid_observation_count < static_cast<float>(user_settings.minimum_sample_count_for_splitting))
			return false;

		VMF lookahead_model = estimate_mean_direction_model(lookahead_signature);
		if (lookahead_model.sharpness == VMF::INVALID_SHARPNESS)
			return false;

		float uncertainty = hippt::sqrt(1.0f / guiding_model.sharpness + 1.0f / lookahead_model.sharpness);
		if (!hippt::is_finite(uncertainty) || uncertainty >= DIRECTION_LUT_MAX_U)
			return false;

		float table_position = uncertainty / DIRECTION_LUT_MAX_U * 255.0f;
		if (!hippt::is_finite(table_position) || table_position < 0.0f || table_position > 255.0f)
			return false;

		unsigned int lower_index = hippt::min(static_cast<unsigned int>(table_position), 254u);
		float interpolation		 = table_position - static_cast<float>(lower_index);
		float threshold_cosine	 = hippt::lerp(COSINE_MAX_ANGLE_DIRECTION_LUT[lower_index], COSINE_MAX_ANGLE_DIRECTION_LUT[lower_index + 1], interpolation);
		float measured_cosine	 = hippt::dot(guiding_model.axis, lookahead_model.axis);

		// A smaller cosine means a greater observed angle which would be no good and we no to split that bad boy
		return measured_cosine < threshold_cosine;
	}

	HIPRT_DEVICE bool should_split_mean_direction(const IlluminationAwareKDTreeIlluminationSignature& guiding_signature,
												  const IlluminationAwareKDTreeIlluminationSignature& lookahead_signature)
	{
		if (guiding_signature.valid_observation_count < static_cast<float>(user_settings.minimum_sample_count_for_splitting))
			return false;

		VMF guiding_model = estimate_mean_direction_model(guiding_signature);

		return should_split_mean_direction(guiding_model, lookahead_signature);
	}

	HIPRT_DEVICE unsigned int find_guiding_cell(float3_t position) const
	{
		unsigned int node_index = 0;

		while (true)
		{
			const IlluminationAwareKDTreeNode& node = nodes[node_index];

			if (node.flags & IlluminationAwareKDTreeNodeFlag_Guiding)
				// We stop at the first guiding cell, even if it may have lookahead cells, we only want guiding cells from this function
				return node_index;

			unsigned int left_child_index  = node.left_child_index;
			unsigned int right_child_index = left_child_index + 1;

			if (left_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || right_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
				// If we're here, the cell doesn't have children and it's not a guiding cell either so we return an invalid index to indicate that the position
				// is not inside a guiding cell
				return IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

			float position_component = node.split_axis == 0 ? position.x : (node.split_axis == 1 ? position.y : position.z);

			if (position_component < node.split_position)
				node_index = left_child_index;
			else
				node_index = right_child_index;
		}
	}

	HIPRT_DEVICE unsigned int find_lookahead_cell(float3_t position) const
	{
		unsigned int node_index = 0;

		while (true)
		{
			const IlluminationAwareKDTreeNode& node = nodes[node_index];

			unsigned int left_child_index  = node.left_child_index;
			unsigned int right_child_index = left_child_index + 1;

			if (node.flags & IlluminationAwareKDTreeNodeFlag_Lookahead &&
				(left_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || right_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX))
				// We found the deepest lookahead cell which doesn't have children so we stop here
				return node_index;

			if (left_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || right_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
				// If we're here, the cell doesn't have children and it's not a lookahead cell either so we return an invalid index to indicate that the
				// position is not inside a lookahead cell
				return IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

			float position_component = node.split_axis == 0 ? position.x : (node.split_axis == 1 ? position.y : position.z);
			if (position_component < node.split_position)
				node_index = left_child_index;
			else
				node_index = right_child_index;
		}
	}

	HIPRT_DEVICE void append_direct_illumination_training_sample(const IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample)
	{
#if DirectLightSamplingStrategy != LSS_BASE_LIGHT_TREE_SG
		return;
#elif DirectLightNEEEstimator != LSS_NEURAL_MANY_LIGHTS && !DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator)
		return;
#endif

		// Invalid samples must not consume buffer space or affect b0.
		if (!sample.valid_for_spatial_training)
			return;

		unsigned int sample_index = hippt::atomic_fetch_add(training_sample_count, 0u);
		// The counter may exceed capacity, but memory must never be written
		// outside the allocated buffer.
		if (sample_index >= training_sample_capacity)
			return;

		sample_index = hippt::atomic_fetch_add(training_sample_count, 1u);
		if (sample_index >= training_sample_capacity)
			return;

		training_samples[sample_index] = sample;
	}

	HIPRT_DEVICE void atomic_add_illumination_signature(IlluminationAwareKDTreeIlluminationSignature* signatures,
														unsigned int node_index,
														float spatial_radiance_weight,
														float3_t incoming_direction)
	{
		// b0 counts all valid samples, including samples with L == 0.
		hippt::atomic_fetch_add_gpu(&signatures[node_index].valid_observation_count, 1u);

		hippt::atomic_fetch_add_gpu(&signatures[node_index].scalar_radiance_sum, spatial_radiance_weight);
		hippt::atomic_fetch_add_gpu(&signatures[node_index].squared_scalar_radiance_sum, spatial_radiance_weight * spatial_radiance_weight);

		hippt::atomic_fetch_add_gpu(&signatures[node_index].weighted_direction_sum.x, spatial_radiance_weight * incoming_direction.x);
		hippt::atomic_fetch_add_gpu(&signatures[node_index].weighted_direction_sum.y, spatial_radiance_weight * incoming_direction.y);
		hippt::atomic_fetch_add_gpu(&signatures[node_index].weighted_direction_sum.z, spatial_radiance_weight * incoming_direction.z);
	}

	HIPRT_DEVICE void atomic_add_spatial_moments(IlluminationAwareKDTreeSpatialSampleMoments* moments, unsigned int node_index, float3_t position)
	{
		hippt::atomic_fetch_add_gpu(&moments[node_index].positive_radiance_sample_count, 1u);

		hippt::atomic_fetch_add_gpu(&moments[node_index].position_sum.x, position.x);
		hippt::atomic_fetch_add_gpu(&moments[node_index].position_sum.y, position.y);
		hippt::atomic_fetch_add_gpu(&moments[node_index].position_sum.z, position.z);

		hippt::atomic_fetch_add_gpu(&moments[node_index].position_squared_sum.x, position.x * position.x);
		hippt::atomic_fetch_add_gpu(&moments[node_index].position_squared_sum.y, position.y * position.y);
		hippt::atomic_fetch_add_gpu(&moments[node_index].position_squared_sum.z, position.z * position.z);
	}

	HIPRT_DEVICE void accumulate_sample_into_existing_tree(IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample)
	{
		// The append path filters invalid samples before storing them, so every sample in the counted prefix is valid here.

		float3_t position			  = sample.position;
		float3_t incoming_direction	  = sample.incoming_direction;
		float spatial_radiance_weight = sample.spatial_radiance_weight;

		unsigned int node_index = sample.cached_guiding_node_index;
		if (node_index == UNRESOLVED_TRAINING_SAMPLE_GUIDING_NODE_INDEX)
		{
			// Samples from paths without a guiding-cell lookup resolve lazily here.
			node_index						 = find_guiding_cell(position);
			sample.cached_guiding_node_index = node_index;
		}
		if (node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			return;

		// Level zero is the guiding cell.
		//
		// Levels one through the configured maximum are the lookahead cells along the sample's
		// unique spatial path.
		for (unsigned int level = 0; level <= IlluminationAwareKDTreeMaximumLookaheadLevelCount; level++)
		{
			// The sample is accumulated only at the deepest existing lookahead cell.
			// An ordered reduction pass propagates it to its guiding ancestors.
			if (level == IlluminationAwareKDTreeMaximumLookaheadLevelCount)
				break;

			IlluminationAwareKDTreeNode& node = nodes[node_index];

			// A missing child means that this lookahead path has not yet been
			// constructed deeply enough. Accumulation stops here.
			if (!(node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
				break;

			unsigned int left_child_index  = node.left_child_index;
			unsigned int right_child_index = left_child_index + 1;
			// Follow exactly one child because the sample position belongs to
			// exactly one k-d cell at this level.
			float position_component = node.split_axis == 0 ? position.x : (node.split_axis == 1 ? position.y : position.z);
			if (position_component < node.split_position)
				node_index = left_child_index;
			else
				node_index = right_child_index;
		}

		atomic_add_illumination_signature(batch_signatures, node_index, spatial_radiance_weight, incoming_direction);

		// Candidate k-d split placement uses only non-zero samples.
		//
		// Zero-radiance samples still contribute to the illumination
		// signature above, but not to the spatial mean and variance.
		if (spatial_radiance_weight > 0.0f)
			atomic_add_spatial_moments(batch_spatial_moments, node_index, position);
	}

	HIPRT_DEVICE void compute_split_axis_and_position(const IlluminationAwareKDTreeSpatialSampleMoments& moments,
													  uint8_t& out_split_axis,
													  float& out_split_position) const
	{
		float countf = static_cast<float>(moments.positive_radiance_sample_count);
		if (countf <= 0.0f)
		{
			out_split_axis	   = IlluminationAwareKDTreeNode::INVALID_SPLIT_AXIS;
			out_split_position = 0.0f;

			return;
		}

		float3_t mean_position = moments.position_sum / countf;
		float3_t variance	   = (moments.position_squared_sum / countf) - (mean_position * mean_position);

		if (variance.x >= variance.y && variance.x >= variance.z)
		{
			out_split_axis	   = 0;
			out_split_position = mean_position.x;
		}
		else if (variance.y >= variance.z)
		{
			out_split_axis	   = 1;
			out_split_position = mean_position.y;
		}
		else
		{
			out_split_axis	   = 2;
			out_split_position = mean_position.z;
		}
	}

	HIPRT_DEVICE unsigned int reserve_physical_nodes(unsigned int amount_to_reserve)
	{
		// Read the current value without modifying it.
		unsigned int current_count = hippt::atomic_fetch_add(node_count, 0u);

		// Capture the first node index allocated during this expansion pass. The host resets this marker before each level.
		if (node_count_before_expansion != nullptr)
			hippt::atomic_compare_exchange(node_count_before_expansion, IlluminationAwareKDTreeNode::INVALID_NODE_INDEX, current_count);

		// Another thread may modify node_count between our read and write,
		// so retry until we either reserve the range or discover that the
		// pool is full.
		while (true)
		{
			// Writing "current_count + amount > capacity" could overflow
			if (current_count + amount_to_reserve > node_capacity)
				return IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

			unsigned int desired_count	= current_count + amount_to_reserve;
			unsigned int observed_count = hippt::atomic_compare_exchange(node_count, current_count, desired_count);

			if (observed_count == current_count)
				// The compare-and-swap succeeded.
				return current_count;

			// Another thread allocated nodes first, atomicCAS returned the newer counter value, so retry from it.
			current_count = observed_count;
		}
	}

	HIPRT_DEVICE void append_child_pair_to_frontier(unsigned int* next_frontier,
													AtomicType<unsigned int>* next_frontier_count,
													unsigned int left_child,
													unsigned int right_child)
	{
		// Atomically reserve two contiguous frontier entries.
		unsigned int output_index = hippt::atomic_fetch_add(next_frontier_count, 2u);

		next_frontier[output_index + 0] = left_child;
		next_frontier[output_index + 1] = right_child;
	}

	IlluminationAwareKDTreeUserSettings user_settings;

	IlluminationAwareKDTreeNode* nodes			   = nullptr;
	IlluminationAwareKDTreeNodeBounds* node_bounds = nullptr;
	unsigned int* parent_indices				   = nullptr;

	AtomicType<unsigned int>* node_count				  = nullptr;
	AtomicType<unsigned int>* node_count_before_expansion = nullptr;
	unsigned int node_capacity							  = 0;

	unsigned int* active_guiding_nodes					= nullptr;
	AtomicType<unsigned int>* active_guiding_node_count = nullptr;
	uint8_t* needs_split								= nullptr;

	// Two ping ponging frontier buffers for when we create lookahead cells
	//
	// Nodes at the lookahead depth currently being processed.
	unsigned int* current_frontier					 = nullptr;
	AtomicType<unsigned int>* current_frontier_count = nullptr;
	// Children that form the next lookahead depth.
	unsigned int* next_frontier					  = nullptr;
	AtomicType<unsigned int>* next_frontier_count = nullptr;

	IlluminationAwareKDTreeDirectIlluminationTrainingSample* training_samples = nullptr;
	AtomicType<unsigned int>* training_sample_count							  = nullptr;
	unsigned int training_sample_capacity									  = 0;

	IlluminationAwareKDTreeIlluminationSignature* batch_signatures	 = nullptr;
	IlluminationAwareKDTreeIlluminationSignature* history_signatures = nullptr;

	IlluminationAwareKDTreeSpatialSampleMoments* batch_spatial_moments	 = nullptr;
	IlluminationAwareKDTreeSpatialSampleMoments* history_spatial_moments = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_CORE_DEVICE_H
