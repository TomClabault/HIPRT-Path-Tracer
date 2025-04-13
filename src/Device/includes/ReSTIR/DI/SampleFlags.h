/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SAMPLE_FLAGS_H
#define DEVICE_RESTIR_DI_SAMPLE_FLAGS_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"

enum ReSTIRDISampleFlags
{
    RESTIR_DI_FLAGS_NONE = 0,
    // The sample is an evmap sample and 'point_on_light_source'
    // should be interpreted as a direction, not a point on a light source
    RESTIR_DI_FLAGS_ENVMAP_SAMPLE = 1 << 0,
    // The sample is a BSDF sample and we're indicating which lobe it comes from
    // so that when evaluating the reservoir in FinalShading, we know what lobe
    // the sample comes from and we can properly evaluate the BSDF
    //
    // We're reusing the values from the BSDFIncidentLightInfo enum here to be able
    // to convert easily from the ReSTIRDI flags back to BSDFIncidentLightInfo (i.e.
    // retrieve which lobe we sampled from the ReSTIRDISampleFlags)
    RESTIR_DI_FLAGS_SAMPLED_FROM_COAT_LOBE = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_COAT_LOBE,
    RESTIR_DI_FLAGS_SAMPLED_FROM_FIRST_METAL_LOBE = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_FIRST_METAL_LOBE,
    RESTIR_DI_FLAGS_SAMPLED_FROM_SECOND_METAL_LOBE = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SECOND_METAL_LOBE,
    RESTIR_DI_FLAGS_SAMPLED_FROM_SPECULAR_LOBE = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE,
    RESTIR_DI_FLAGS_SAMPLED_FROM_GLASS_REFLECT_LOBE = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFLECT_LOBE,
    RESTIR_DI_FLAGS_SAMPLED_FROM_GLASS_REFRACT_LOBE = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE,
    // This sample *AT ITS OWN PIXEL* is unoccluded. This can be used to avoid tracing
    // rays for visibility since we know it's unoccluded already
    RESTIR_DI_FLAGS_UNOCCLUDED = 1 << 7
};

#endif
