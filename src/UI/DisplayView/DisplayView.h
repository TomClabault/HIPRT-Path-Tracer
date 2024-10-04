/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_VIEW_H
#define DISPLAY_VIEW_H

#include "OpenGL/OpenGLProgram.h"
#include "UI/DisplayView/DisplayTextureType.h"
#include "UI/DisplayView/DisplayViewEnum.h"

#include <memory>

class DisplayView
{
public:
	DisplayView() {};
	DisplayView(DisplayViewType display_view_type, std::shared_ptr<OpenGLProgram> display_program);

	std::shared_ptr<OpenGLProgram> get_display_program();
	DisplayViewType get_display_view_type() const;

private:
	// What display view type is currently displayed by the system
	DisplayViewType m_display_view_type = DisplayViewType::DEFAULT;

	// Fragment shader + vertex shader used for displaying the view on the viewport
	std::shared_ptr<OpenGLProgram> m_display_program = nullptr;
};

#endif
