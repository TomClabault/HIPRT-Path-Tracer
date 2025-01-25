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
	// If ReSTIR DI isn't used, we don't need the last frame's g-buffer
	// (as far as the codebase goes at the time of writing this function anyways)
	bool need_g_buffer = renderer->get_ReSTIR_DI_render_pass()->using_ReSTIR_DI();
	// If the temporal reuse isn't used, don't need the G-buffer
	need_g_buffer &= restir_di_settings.temporal_pass.do_temporal_reuse_pass;

	return need_g_buffer;
}
#endif
