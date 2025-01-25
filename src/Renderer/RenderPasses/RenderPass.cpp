/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/RenderPass.h"

RenderPass::RenderPass() : m_renderer(nullptr) {}
RenderPass::RenderPass(GPURenderer* renderer) : m_renderer(renderer) {}
