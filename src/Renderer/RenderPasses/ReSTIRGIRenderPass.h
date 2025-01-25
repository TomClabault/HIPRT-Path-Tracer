/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_GI_RENDER_PASS_H
#define RESTIR_GI_RENDER_PASS_H

#include "Renderer/RenderPasses/RenderPass.h"

class GPURenderer;

class ReSTIRGIRenderPass : public RenderPass
{
public:
	ReSTIRGIRenderPass() {}
	ReSTIRGIRenderPass(GPURenderer* renderer);

private:
	GPURenderer* m_renderer = nullptr;
};

#endif
