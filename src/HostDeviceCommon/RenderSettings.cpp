/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "HostDeviceCommon/RenderSettings.h"
#include "Renderer/GPURenderer.h"

#ifndef __KERNELCC__
HIPRT_HOST bool HIPRTRenderSettings::use_prev_frame_g_buffer(GPURenderer* renderer) const
{
	bool need_g_buffer = restir_di_settings.temporal_pass.use_last_frame_g_buffer;
	// If ReSTIR DI isn't used, we don't need the last frame's g-buffer
	need_g_buffer &= renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI;
	// If the temporal reuse isn't used, don't need the G-buffer
	need_g_buffer &= restir_di_settings.temporal_pass.do_temporal_reuse_pass;
	// Not using the g-buffer if accumulating because we're rendering still frames
	// when accumulating which means that we don't need the previous frame's g-buffer
	// (only required for unbiasedness *in motion*)
	need_g_buffer &= !accumulate;

	return need_g_buffer;
}
#endif
