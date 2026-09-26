#ifndef RENDERER_WAVEFRONT_ROUTING_CONFIGURATION_H
#define RENDERER_WAVEFRONT_ROUTING_CONFIGURATION_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"

struct WavefrontRoutingConfiguration
{
	bool route_shading_by_material_family;
	bool route_deferred_completion_by_material_family;
	bool requires_terminal_trace_completion;
};

inline WavefrontRoutingConfiguration get_wavefront_routing_configuration(int bsdf_model, bool material_specialization_enabled, int nee_estimator)
{
	bool route_shading_by_material_family			  = bsdf_model == BSDF_PRINCIPLED && material_specialization_enabled;
	bool route_deferred_completion_by_material_family = route_shading_by_material_family && nee_estimator == LSS_RIS_BSDF_AND_LIGHT;
	bool requires_terminal_trace_completion = nee_estimator == LSS_BSDF || nee_estimator == LSS_MIS_LIGHT_BSDF || nee_estimator == LSS_RIS_BSDF_AND_LIGHT ||
											  nee_estimator == LSS_RISLTC || nee_estimator == LSS_LEARNING_TO_CLUSTER_MIS;

	return { route_shading_by_material_family, route_deferred_completion_by_material_family, requires_terminal_trace_completion };
}

inline bool wavefront_path_strategy_uses_pass(int path_sampling_strategy, bool nisml_enabled)
{
	return !nisml_enabled && path_sampling_strategy == PATH_SAMPLING_BSDF_WAVEFRONT;
}

#endif // #ifndef RENDERER_WAVEFRONT_ROUTING_CONFIGURATION_H
