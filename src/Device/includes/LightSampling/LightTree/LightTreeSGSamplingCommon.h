/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_COMMON_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_COMMON_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"
#include "HostDeviceCommon/Material/LightProposalState.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE void get_sg_specular_importance_parameters(const DeviceUnpackedPrincipledFullMaterial& material,
														float& sg_specular_weight,
														float& alpha_x,
														float& alpha_y)
{
	LightProposalInputs proposal_inputs = make_light_proposal_inputs(material);
	get_sg_specular_importance_parameters(proposal_inputs, sg_specular_weight, alpha_x, alpha_y);
}

#endif // #ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_COMMON_H
