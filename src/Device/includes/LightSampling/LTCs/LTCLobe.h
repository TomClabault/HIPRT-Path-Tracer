/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTCS_LOBE_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTCS_LOBE_H

enum LTCLobe
{
	COAT_LOBE	  = 0,
	METALLIC_LOBE = 1,
	SPECULAR_LOBE = 2,
	DIFFUSE_LOBE  = 3,
};

struct LTCLobeSampleProbabilities
{
	float coat_proba;
	float metallic_proba;
	float specular_proba;
	// Diffuse proba is implicit as 1.0 - (sum other probabilities)
	// float diffuse_proba;
};

#endif
