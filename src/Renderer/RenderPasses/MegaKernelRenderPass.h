/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef MEGAKERNEL_RENDER_PASS_H
#define MEGAKERNEL_RENDER_PASS_H

#include "Renderer/RenderPasses/RenderPass.h"

class MegaKernelRenderPass : public RenderPass
{
public:
	static const std::string MEGAKERNEL_RENDER_PASS_NAME;
	static const std::string MEGAKERNEL_KERNEL;

	MegaKernelRenderPass();
	MegaKernelRenderPass(GPURenderer* renderer);

	virtual void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {});

	virtual void resize(unsigned int new_width, unsigned int new_height);
	
	virtual bool pre_render_update(float delta_time);
	virtual bool launch();
	virtual void post_render_update();

	virtual void update_render_data() {};
	virtual void reset();

private:
	int2 m_render_resolution = make_int2(0, 0);
};

#endif
