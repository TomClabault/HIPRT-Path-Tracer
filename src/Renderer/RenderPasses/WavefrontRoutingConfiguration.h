#ifndef RENDERER_WAVEFRONT_ROUTING_CONFIGURATION_H
#define RENDERER_WAVEFRONT_ROUTING_CONFIGURATION_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"

struct WavefrontRoutingConfiguration
{
	// Number of material families the paths are sorted into before shading. The material families are only defined
	// for the principled BSDF: the other BSDF models only have the KernelMaterialSpecializationAll family
	unsigned int material_family_count;
	bool route_deferred_completion_by_material_family;
	bool requires_terminal_trace_completion;
};

inline WavefrontRoutingConfiguration get_wavefront_routing_configuration(int bsdf_model, int nee_estimator)
{
	unsigned int material_family_count				  = bsdf_model == BSDF_PRINCIPLED ? KernelMaterialSpecializationCount : 1;
	bool route_deferred_completion_by_material_family = bsdf_model == BSDF_PRINCIPLED && nee_estimator == LSS_RIS_BSDF_AND_LIGHT;
	bool requires_terminal_trace_completion = nee_estimator == LSS_BSDF || nee_estimator == LSS_MIS_LIGHT_BSDF || nee_estimator == LSS_RIS_BSDF_AND_LIGHT ||
											  nee_estimator == LSS_RISLTC || nee_estimator == LSS_LEARNING_TO_CLUSTER_MIS;

	return { material_family_count, route_deferred_completion_by_material_family, requires_terminal_trace_completion };
}

inline bool wavefront_path_strategy_uses_pass(int path_sampling_strategy, bool nisml_enabled)
{
	return !nisml_enabled && path_sampling_strategy == PATH_SAMPLING_BSDF_WAVEFRONT;
}

#endif // #ifndef RENDERER_WAVEFRONT_ROUTING_CONFIGURATION_H
