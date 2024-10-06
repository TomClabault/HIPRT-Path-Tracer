/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_LOGGER_H
#define IMGUI_LOGGER_H

#include "imgui.h"

#include <mutex>
#include <string>
#include <vector>

enum ImGuiLoggerSeverity
{
    IMGUI_LOGGER_INFO,
    IMGUI_LOGGER_WARNING,
    IMGUI_LOGGER_ERROR
};

/**
 * Class from imgui_demo.cpp "ExampleAppLog"
 */
class ImGuiLogger
{
public:
    ImGuiLogger()
    {
        m_auto_scroll = true;
        clear();
    }

    void clear()
    {
        m_text_buffer.clear();
        m_line_offsets.clear();
        m_line_offsets.push_back(0);
    }

    void add_line(ImGuiLoggerSeverity severity, const char* fmt, ...) IM_FMTARGS(2)
    {
        // For logger's thread safety
        std::lock_guard<std::mutex> lock(m_mutex);

        std::string fmt_prefix_str = ImGuiLogger::add_severity_prefix(severity, fmt);
        fmt_prefix_str += "\n";
        const char* fmt_prefix = fmt_prefix_str.c_str();


        int old_size = m_text_buffer.size();
        va_list args;
        va_start(args, fmt);
        m_text_buffer.appendfv(fmt_prefix, args);
        vprintf(fmt_prefix, args); // Logging to the console as well
        va_end(args);
        for (int new_size = m_text_buffer.size(); old_size < new_size; old_size++)
            if (m_text_buffer[old_size] == '\n')
                m_line_offsets.push_back(old_size + 1);

        m_line_severities.push_back(severity);

    }

    void draw(const char* title, bool* p_open = NULL)
    {
        if (!ImGui::Begin(title, p_open))
        {
            ImGui::End();
            return;
        }

        // Options menu
        if (ImGui::BeginPopup("Options"))
        {
            ImGui::Checkbox("Auto-scroll", &m_auto_scroll);
            ImGui::EndPopup();
        }

        // Main window
        if (ImGui::Button("Options"))
            ImGui::OpenPopup("Options");
        ImGui::SameLine();
        bool clear_button = ImGui::Button("Clear");
        ImGui::SameLine();
        bool copy = ImGui::Button("Copy");
        ImGui::SameLine();
        m_text_filter.Draw("Filter", -100.0f);
        ImGui::SameLine();

        ImGui::Separator();

        if (ImGui::BeginChild("scrolling", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar))
        {
            if (clear_button)
                clear();
            if (copy)
                ImGui::LogToClipboard();
            if (m_text_buffer.size() == 0)
            {
                ImGui::EndChild();
                ImGui::End();

                return;
            }

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            const char* buf = m_text_buffer.begin();
            const char* buf_end = m_text_buffer.end();
            if (m_text_filter.IsActive())
            {
                // In this example we don't use the clipper when Filter is enabled.
                // This is because we don't have random access to the result of our filter.
                // A real application processing logs with ten of thousands of entries may want to store the result of
                // search/filter.. especially if the filtering function is not trivial (e.g. reg-exp).
                for (int line_no = 0; line_no < m_line_offsets.Size; line_no++)
                {
                    const char* line_start = buf + m_line_offsets[line_no];
                    const char* line_end = (line_no + 1 < m_line_offsets.Size) ? (buf + m_line_offsets[line_no + 1] - 1) : buf_end;
                    if (m_text_filter.PassFilter(line_start, line_end))
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGuiLogger::get_severity_color(m_line_severities[line_no]));
                        ImGui::TextUnformatted(line_start, line_end);
                        ImGui::PopStyleColor();
                    }
                }
            }
            else
            {
                // The simplest and easy way to display the entire buffer:
                //   ImGui::TextUnformatted(buf_begin, buf_end);
                // And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward
                // to skip non-visible lines. Here we instead demonstrate using the clipper to only process lines that are
                // within the visible area.
                // If you have tens of thousands of items and their processing cost is non-negligible, coarse clipping them
                // on your side is recommended. Using ImGuiListClipper requires
                // - A) random access into your data
                // - B) items all being the  same height,
                // both of which we can handle since we have an array pointing to the beginning of each line of text.
                // When using the filter (in the block of code above) we don't have random access into the data to display
                // anymore, which is why we don't use the clipper. Storing or skimming through the search result would make
                // it possible (and would be recommended if you want to search through tens of thousands of entries).
                ImGuiListClipper clipper;
                clipper.Begin(m_line_offsets.Size);
                while (clipper.Step())
                {
                    for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
                    {
                        const char* line_start = buf + m_line_offsets[line_no];
                        const char* line_end = (line_no + 1 < m_line_offsets.Size) ? (buf + m_line_offsets[line_no + 1] - 1) : buf_end;

                        ImGui::PushStyleColor(ImGuiCol_Text, ImGuiLogger::get_severity_color(m_line_severities[line_no]));
                        ImGui::TextUnformatted(line_start, line_end);
                        ImGui::PopStyleColor();
                    }
                }
                clipper.End();
            }
            ImGui::PopStyleVar();

            // Keep up at the bottom of the scroll region if we were already at the bottom at the beginning of the frame.
            // Using a scrollbar or mouse-wheel will take away from the bottom edge.
            if (m_auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                ImGui::SetScrollHereY(1.0f);
        }
        ImGui::EndChild();
        ImGui::End();
    }

    static ImU32 get_severity_color(ImGuiLoggerSeverity severity)
    {
        switch (severity)
        {
        case IMGUI_LOGGER_INFO:
            return IM_COL32(255, 255, 255, 255);

        case IMGUI_LOGGER_WARNING:
            return IM_COL32(255, 255, 0, 255);

        case IMGUI_LOGGER_ERROR:
            return IM_COL32(255, 0, 0, 255);

        default:
            return IM_COL32(255, 255, 255, 255);
        }
    }

private:
    static std::string add_severity_prefix(ImGuiLoggerSeverity severity, const char* fmt)
    {
        std::string prefix;
        switch (severity)
        {
        case IMGUI_LOGGER_INFO:
            prefix = "[INFO] ";
            break;

        case IMGUI_LOGGER_WARNING:
            prefix = "[WARN] ";
            break;

        case IMGUI_LOGGER_ERROR:
            prefix = "[ERR ] ";
            break;

        default:
            prefix = "";
            break;
        }

        return prefix + std::string(fmt);
    }

    ImGuiTextBuffer m_text_buffer;
    ImGuiTextFilter m_text_filter;
    ImVector<int> m_line_offsets; // Index to lines offset. We maintain this with AddLog() calls.
    std::vector<ImGuiLoggerSeverity> m_line_severities;

    bool m_auto_scroll;  // Keep scrolling if already at the bottom.

    // For logger thread safety
    std::mutex m_mutex;
};

#endif
