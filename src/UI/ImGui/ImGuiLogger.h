/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_LOGGER_H
#define IMGUI_LOGGER_H

#include "UI/ImGui/ImGuiLoggerSeverity.h"
#include "UI/ImGui/ImGuiLoggerLine.h"

#include "imgui.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

/**
 * Class derived from imgui_demo.cpp "ExampleAppLog"
 */

class ImGuiLogger
{
public:
    static const char* BACKGROUND_KERNEL_PARSING_LINE_NAME;
    static const char* BACKGROUND_KERNEL_COMPILATION_LINE_NAME;

    ImGuiLogger();
    ~ImGuiLogger();

    void add_line_with_name(ImGuiLoggerSeverity severity, const char* line_name, const char* fmt, ...) IM_FMTARGS(4);
    void add_line(ImGuiLoggerSeverity severity, const char* fmt, ...) IM_FMTARGS(3);

    void draw(const char* title, bool* p_open = NULL);
    void clear();

    void update_line(const char* line_name, const char* fmt, ...);

    static ImU32 get_severity_color(ImGuiLoggerSeverity severity);

private:
    void add_line_internal(ImGuiLoggerSeverity severity, const char* line_name, const char* fmt, va_list args);

    void set_line_name(std::shared_ptr<ImGuiLoggerLine> line, const char* line_name);

    std::string compute_formatted_string(const char* fmt, va_list args);
    void compute_actual_lines(std::shared_ptr<ImGuiLoggerLine> logger_line);

    std::pair<std::shared_ptr<ImGuiLoggerLine>, std::string_view*> get_line_from_index(int index);

    static std::string get_severity_prefix(ImGuiLoggerSeverity severity);

    // Each time you call add_log(), one entry is added in there with the whole text
    // and severity.
    // We're using shared_ptr here because when adding new lines to m_log_lines, the
    // vector may be resized in which case, all instances of ImGuiLoggerLine will be
    // moved and references/pointers that we had on it become invalid.
    // By using shared_ptr, we're allocating the lines on the heap and thus we always
    // keep valid references to them
    std::vector<std::shared_ptr<ImGuiLoggerLine>> m_log_lines;

    // If you call add_log() with a text that contains multiple "\n" (i.e. multiple lines)
    // each individual line will be added in that vector. This is used for drawing properly
    // because drawing needs the actual lines separated by \n, not the "lines" that the entire
    // string that the user gave when calling add_log()
    std::vector<std::vector<std::string_view>> m_actual_lines;
    // For a given ImGuiLoggerLine, the value is the index in 'm_actual_lines' of that ImGuiLoggerLine
    // so this map can be used to retrieve the actual lines (vector of std::string_view)
    // of an ImGuiLoggerLine
    std::unordered_map<std::shared_ptr<ImGuiLoggerLine>, int> m_index_in_actual_lines;
    // Cache for the get_line_from_index() method. If we ask for the same index twice, 
    // we can just look in the cache for what the line was for this index.
    // The cache is invalidated if a call to 'update_line()' modifies the number of
    // actual lines of an ImGuiLoggerLine (by giving a text that contains more '\n' than
    // the previous for example)
    std::unordered_map<int, std::pair<std::shared_ptr<ImGuiLoggerLine>, std::string_view*>> m_index_to_line_cache;
    // This variable is equivalent to: for (auto& l : m_actual_lines) total += l.size();
    int m_total_number_of_lines = 0;

    // User given names to their associated line
    std::unordered_map<const char*, std::shared_ptr<ImGuiLoggerLine>> m_names_to_lines;

    ImGuiTextFilter m_text_filter;

    bool m_auto_scroll = true;  // Keep scrolling if already at the bottom.

    // For logger thread safety
    std::mutex m_mutex;

    // Used for threads that may still want to access this logger after
    // it's been destroyed by another thread
    bool m_destroyed = false;
};

#endif
