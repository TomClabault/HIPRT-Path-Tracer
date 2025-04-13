/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/DisplayView/DisplayView.h"

DisplayView::DisplayView(DisplayViewType display_view_type, std::shared_ptr<OpenGLProgram> display_program)
{
	m_display_view_type = display_view_type;
	m_display_program = display_program;
}

std::shared_ptr<OpenGLProgram> DisplayView::get_display_program()
{
	return m_display_program;
}

DisplayViewType DisplayView::get_display_view_type() const
{
	return m_display_view_type;
}
