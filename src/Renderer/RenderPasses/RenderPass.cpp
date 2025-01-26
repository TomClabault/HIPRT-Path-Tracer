/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/RenderPass.h"

RenderPass::RenderPass() {}
RenderPass::RenderPass(GPURenderer* renderer) : RenderPass(renderer, "Unnamed render pass") {}
RenderPass::RenderPass(GPURenderer* renderer, const std::string& name) : m_renderer(renderer), m_render_data(&m_renderer->get_render_data()), m_name(name) {}
