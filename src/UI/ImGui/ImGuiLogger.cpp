#include "UI/ImGui/ImGuiLogger.h"

ImGuiLogger g_imgui_logger;

const char* ImGuiLogger::BACKGROUND_KERNEL_PARSING_LINE_NAME = "BackgroundKernelParsingLineName";
const char* ImGuiLogger::BACKGROUND_KERNEL_COMPILATION_LINE_NAME = "BackgroundKernelPrecompilationLineName";

ImGuiLogger::ImGuiLogger()
{
    clear();
}

ImGuiLogger::~ImGuiLogger()
{
    m_destroyed = true;
}

void ImGuiLogger::add_line_with_name(ImGuiLoggerSeverity severity, const char* line_name, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    add_line_internal(severity, line_name, fmt, args);
    va_end(args);
}

void ImGuiLogger::add_line(ImGuiLoggerSeverity severity, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    add_line_internal(severity, nullptr, fmt, args);
    va_end(args);
}

void ImGuiLogger::draw(const char* title, bool* p_open)
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
        if (m_log_lines.size() == 0)
        {
            ImGui::EndChild();
            ImGui::End();

            return;
        }

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        if (m_text_filter.IsActive())
        {
            // In this example we don't use the clipper when Filter is enabled.
            // This is because we don't have random access to the result of our filter.
            // A real application processing logs with ten of thousands of entries may want to store the result of
            // search/filter.. especially if the filtering function is not trivial (e.g. reg-exp).
            for (int line_no = 0; line_no < m_actual_lines.size(); line_no++)
            {
                std::pair<std::shared_ptr<ImGuiLoggerLine>, std::string_view*> line_view_pair = get_line_from_index(line_no);

                std::shared_ptr<ImGuiLoggerLine> line = line_view_pair.first;
                std::string_view* str_view = line_view_pair.second;

                const char* line_start = str_view->data();
                const char* line_end = line_start + str_view->length();

                if (m_text_filter.PassFilter(line_start, line_end))
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGuiLogger::get_severity_color(line->severity));
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
            clipper.Begin(m_total_number_of_lines);
            while (clipper.Step())
            {
                for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
                {
                    std::pair<std::shared_ptr<ImGuiLoggerLine>, std::string_view*> line_view_pair = get_line_from_index(line_no);

                    std::shared_ptr<ImGuiLoggerLine> line = line_view_pair.first;
                    std::string_view* str_view = line_view_pair.second;

                    const char* line_start = str_view->data();
                    const char* line_end = line_start + str_view->length();

                    ImGui::PushStyleColor(ImGuiCol_Text, ImGuiLogger::get_severity_color(line->severity));
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

void ImGuiLogger::clear()
{
    m_log_lines.clear();
    m_actual_lines.clear();
    m_index_in_actual_lines.clear();
    m_index_to_line_cache.clear();
    m_names_to_lines.clear();
    m_total_number_of_lines = 0;
}

void ImGuiLogger::update_line(const char* line_name, const char* fmt, ...)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_destroyed)
        return;

    auto find = m_names_to_lines.find(line_name);
    if (find == m_names_to_lines.end())
    {
        add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Cannot update line with name %s. There is no such line. Did you forget to call add_line(severity, LINE_NAME, ...)?", line_name);
        return;
    }

    std::shared_ptr<ImGuiLoggerLine> line = find->second;
    std::string prefix = ImGuiLogger::get_severity_prefix(line->severity);

    va_list args;
    va_start(args, fmt);
    std::string formatted_string = prefix + compute_formatted_string(fmt, args) + "\n";
    va_end(args);

    // Updating the line
    line->string = formatted_string;

    // Updating the actual lines
    int nb_actual_lines_before_update = m_actual_lines[m_index_in_actual_lines[line]].size();
    compute_actual_lines(line);

    // Updating a line invalidates the cache if the number of actual lines
    // changed
    int nb_actual_lines_after_update = m_actual_lines[m_index_in_actual_lines[line]].size();
    if (nb_actual_lines_before_update != nb_actual_lines_after_update)
        m_index_to_line_cache.clear();

    m_total_number_of_lines -= nb_actual_lines_before_update;
    m_total_number_of_lines += nb_actual_lines_after_update;
}

ImU32 ImGuiLogger::get_severity_color(ImGuiLoggerSeverity severity)
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
        return IM_COL32(255, 0, 255, 255);
    }
}

std::string ImGuiLogger::get_severity_prefix(ImGuiLoggerSeverity severity)
{
    switch (severity)
    {
    case IMGUI_LOGGER_INFO:
        return "[INFO] ";

    case IMGUI_LOGGER_WARNING:
        return "[WARN] ";

    case IMGUI_LOGGER_ERROR:
        return "[ERR ] ";

    default:
        return "";
    }
}

void ImGuiLogger::add_line_internal(ImGuiLoggerSeverity severity, const char* line_name, const char* fmt, va_list args)
{
    // For logger's thread safety
    std::lock_guard<std::mutex> lock(m_mutex);

    std::string prefix = ImGuiLogger::get_severity_prefix(severity);
    std::string formatted_string = prefix + compute_formatted_string(fmt, args) + "\n";
    std::cout << formatted_string; // Also printing to the console

    int line_index = m_log_lines.size();

    std::shared_ptr<ImGuiLoggerLine> logger_line = std::make_shared<ImGuiLoggerLine>(formatted_string, severity);
    m_log_lines.push_back(logger_line);
    m_index_in_actual_lines[logger_line] = line_index;

    compute_actual_lines(logger_line);
    m_total_number_of_lines += m_actual_lines[line_index].size();

    if (line_name != nullptr)
        set_line_name(logger_line, line_name);
}

void ImGuiLogger::set_line_name(std::shared_ptr<ImGuiLoggerLine> line, const char* line_name)
{
    m_names_to_lines[line_name] = line;
}

std::string ImGuiLogger::compute_formatted_string(const char* fmt, va_list args)
{
    // Copying the arg list because the first call to vsnprintf modifies args
    // and so if we use args again in the second call to vsnprintf, we're going
    // to get garbage in the formatted output 
    va_list args_copy;
    va_copy(args_copy, args);

    // Calculating formatted string length by calling with NULL. + 1 for the '\0'
    int string_length = vsnprintf(NULL, 0, fmt, args_copy) + 1;

    va_end(args_copy);

    std::vector<char> string_buffer(string_length);
    vsnprintf(string_buffer.data(), string_length, fmt, args);

    return std::string(string_buffer.data());
}

void ImGuiLogger::compute_actual_lines(std::shared_ptr<ImGuiLoggerLine> logger_line)
{
    int string_length = logger_line->string.size();

    int index = m_index_in_actual_lines.at(logger_line);
    if (index >= m_actual_lines.size())
        // If the entry doesn't exist yet
        m_actual_lines.push_back(std::vector<std::string_view>());

    std::vector<std::string_view>& actual_lines = m_actual_lines[index];
    // We're going to recompute the lines so clearing them first
    actual_lines.clear();

    int previous_line_feed_pos = 0;
    for (int character_pos = 0; character_pos < string_length; character_pos++)
    {
        if (logger_line->string[character_pos] == '\n')
        {
            const char* line_start = logger_line->string.c_str() + previous_line_feed_pos;
            int line_length = character_pos - previous_line_feed_pos;

            actual_lines.push_back(std::string_view(line_start, line_length));

            // + 1 to skip the '\n'
            previous_line_feed_pos = character_pos + 1;
        }
    }
}

std::pair<std::shared_ptr<ImGuiLoggerLine>, std::string_view*> ImGuiLogger::get_line_from_index(int index)
{
    const auto& find = m_index_to_line_cache.find(index);
    if (find != m_index_to_line_cache.end())
        return find->second;

    int total = 0;
    for (int actual_line_index = 0; actual_line_index < m_actual_lines.size(); actual_line_index++)
    {
        std::vector<std::string_view>& actual_lines = m_actual_lines[actual_line_index];

        total += actual_lines.size();
        if (total > index)
        {
            // This means that the line we're looking for is in the current ImGuiLoggerLine

            int offset = total - actual_lines.size();
            int index_in_actual_lines = index - offset;

            std::shared_ptr<ImGuiLoggerLine> logger_line = m_log_lines[actual_line_index];
            std::string_view* line = &actual_lines[index_in_actual_lines];

            std::pair<std::shared_ptr<ImGuiLoggerLine>, std::string_view*> pair(logger_line, line);
            m_index_to_line_cache[index] = pair;

            return pair;
        }

    }

    return std::make_pair(nullptr, nullptr);
}