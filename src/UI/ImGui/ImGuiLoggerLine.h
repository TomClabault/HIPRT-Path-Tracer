/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_LOGGER_LINE_H
#define IMGUI_LOGGER_LINE_H

#include "UI/ImGui/ImGuiLoggerSeverity.h"

#include <string>

struct ImGuiLoggerLine
{
	ImGuiLoggerLine(char* line_string, ImGuiLoggerSeverity line_severity) : string(line_string), severity(line_severity) {};
	ImGuiLoggerLine(const std::string& line_string, ImGuiLoggerSeverity line_severity) : string(line_string), severity(line_severity) {};

	ImGuiLoggerSeverity severity;
	std::string string;
};

#endif