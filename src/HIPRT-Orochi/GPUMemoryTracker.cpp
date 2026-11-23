/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/GPUMemoryTracker.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif // #ifndef NOMINMAX
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif // #ifndef WIN32_LEAN_AND_MEAN

#include <windows.h>

#include <dxgi1_4.h>
#include <pdh.h>
#include <pdhmsg.h>
#endif // #ifdef _WIN32

#include <chrono>
#include <format>

const std::string GPUMemoryTracker::UNATTRIBUTED_CATEGORY = "Other/Unattributed";

const std::string GPUMemoryTracker::SCENE_GEOMETRY_CATEGORY			  = "Scene/Geometry";
const std::string GPUMemoryTracker::SCENE_BVH_CATEGORY				  = "Scene/BVH";
const std::string GPUMemoryTracker::SCENE_MATERIALS_CATEGORY		  = "Scene/Materials";
const std::string GPUMemoryTracker::SCENE_TEXTURES_CATEGORY			  = "Scene/Textures";
const std::string GPUMemoryTracker::SCENE_EMISSIVE_TRIANGLES_CATEGORY = "Scene/Emissive triangles";
const std::string GPUMemoryTracker::SCENE_ENVMAP_CATEGORY			  = "Scene/Environment map";

const std::string GPUMemoryTracker::LIGHT_SAMPLING_POWER_ALIAS_TABLE_CATEGORY = "Light sampling/Power alias table";
const std::string GPUMemoryTracker::LIGHT_SAMPLING_LIGHT_TREE_ATS_CATEGORY	  = "Light sampling/Light tree ATS";
const std::string GPUMemoryTracker::LIGHT_SAMPLING_LIGHT_TREE_SG_CATEGORY	  = "Light sampling/Light tree SG";

const std::string GPUMemoryTracker::RENDERER_FRAMEBUFFERS_CATEGORY			  = "Renderer/Framebuffers";
const std::string GPUMemoryTracker::RENDERER_PER_PIXEL_BUFFERS_CATEGORY		  = "Renderer/Per-pixel buffers";
const std::string GPUMemoryTracker::RENDERER_ADAPTIVE_SAMPLING_CATEGORY		  = "Renderer/Adaptive sampling";
const std::string GPUMemoryTracker::RENDERER_STATUS_BUFFERS_CATEGORY		  = "Renderer/Status buffers";
const std::string GPUMemoryTracker::RENDERER_BVH_TRAVERSAL_STACK_CATEGORY	  = "Renderer/BVH traversal global stack";
const std::string GPUMemoryTracker::RENDERER_BSDF_PRECOMPUTED_TABLES_CATEGORY = "Renderer/BSDF precomputed tables";
const std::string GPUMemoryTracker::RENDERER_DENOISER_CATEGORY				  = "Renderer/Denoiser";

const std::string GPUMemoryTracker::RENDER_PASSES_GROUP = "Render passes";

void GPUMemoryTracker::register_allocation(const void* allocation_key, size_t byte_size, GPUMemoryType memory_type, const std::string& fallback_category)
{
	if (allocation_key == nullptr)
		return;

	TrackedAllocation allocation;
	allocation.category	   = get_current_category();
	allocation.byte_size   = byte_size;
	allocation.memory_type = memory_type;

	if (allocation.category.empty())
		allocation.category = fallback_category.empty() ? UNATTRIBUTED_CATEGORY : fallback_category;

	TrackerState& state = get_state();
	std::lock_guard<std::mutex> lock(state.mutex);

	state.allocations[allocation_key] = allocation;
}

void GPUMemoryTracker::unregister_allocation(const void* allocation_key)
{
	TrackerState& state = get_state();
	std::lock_guard<std::mutex> lock(state.mutex);

	state.allocations.erase(allocation_key);
}

std::string GPUMemoryTracker::get_allocation_category(const void* allocation_key)
{
	TrackerState& state = get_state();
	std::lock_guard<std::mutex> lock(state.mutex);

	auto find = state.allocations.find(allocation_key);
	if (find == state.allocations.end())
		return "";

	return find->second.category;
}

std::map<std::string, size_t> GPUMemoryTracker::get_category_byte_sizes(GPUMemoryType memory_type)
{
	std::map<std::string, size_t> category_byte_sizes;

	TrackerState& state = get_state();
	std::lock_guard<std::mutex> lock(state.mutex);

	for (const auto& key_to_allocation : state.allocations)
		if (key_to_allocation.second.memory_type == memory_type)
			category_byte_sizes[key_to_allocation.second.category] += key_to_allocation.second.byte_size;

	return category_byte_sizes;
}

size_t GPUMemoryTracker::get_total_byte_size(GPUMemoryType memory_type)
{
	size_t total_byte_size = 0;

	TrackerState& state = get_state();
	std::lock_guard<std::mutex> lock(state.mutex);

	for (const auto& key_to_allocation : state.allocations)
		if (key_to_allocation.second.memory_type == memory_type)
			total_byte_size += key_to_allocation.second.byte_size;

	return total_byte_size;
}

std::string GPUMemoryTracker::get_current_category()
{
	std::vector<std::string>& category_stack = get_thread_category_stack();
	if (category_stack.empty())
		return "";

	return category_stack.back();
}

void GPUMemoryTracker::push_category(const std::string& category)
{
	get_thread_category_stack().push_back(category);
}

void GPUMemoryTracker::pop_category()
{
	std::vector<std::string>& category_stack = get_thread_category_stack();
	if (!category_stack.empty())
		category_stack.pop_back();
}

GPUProcessMemoryInfo GPUMemoryTracker::query_process_memory_info(const std::string& device_name, size_t device_total_memory_bytes)
{
	GPUProcessMemoryInfo memory_info;

#ifdef _WIN32
	static GPUAdapterIdentifier adapter = find_gpu_adapter(device_name, device_total_memory_bytes);
	if (!adapter.found)
		return memory_info;

	static GPUProcessMemoryInfo last_memory_info;
	static std::chrono::steady_clock::time_point last_query_time;

	std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
	if (last_memory_info.available && now - last_query_time < std::chrono::milliseconds(500))
		return last_memory_info;

	memory_info.adapter_dedicated_memory_bytes = adapter.dedicated_memory_bytes;
	memory_info.available					   = read_gpu_memory_performance_counters(adapter, memory_info);

	last_memory_info = memory_info;
	last_query_time	 = now;
#endif // #ifdef _WIN32

	return memory_info;
}

GPUMemoryTracker::GPUAdapterIdentifier GPUMemoryTracker::find_gpu_adapter(const std::string& device_name, size_t device_total_memory_bytes)
{
	GPUAdapterIdentifier best_adapter;

#ifdef _WIN32
	IDXGIFactory1* factory = nullptr;
	if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
		return best_adapter;

	// The HIP device properties available here don't carry the LUID of the adapter: the adapter is the one with the same
	// name as the device and, among those, the one whose dedicated memory is the closest to the memory of the device
	bool best_adapter_name_matches		  = false;
	size_t best_adapter_memory_difference = 0;

	IDXGIAdapter1* candidate_adapter = nullptr;
	for (unsigned int adapter_index = 0; factory->EnumAdapters1(adapter_index, &candidate_adapter) != DXGI_ERROR_NOT_FOUND; adapter_index++)
	{
		DXGI_ADAPTER_DESC1 adapter_description;
		bool description_valid = SUCCEEDED(candidate_adapter->GetDesc1(&adapter_description));
		candidate_adapter->Release();

		if (!description_valid || (adapter_description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
			continue;

		// DXGI adapter names are wide strings, GPU names are ASCII
		std::string adapter_name;
		for (unsigned int character_index = 0; adapter_description.Description[character_index] != L'\0'; character_index++)
			adapter_name += static_cast<char>(adapter_description.Description[character_index]);

		bool adapter_name_matches		 = adapter_name == device_name;
		size_t adapter_memory_difference = adapter_description.DedicatedVideoMemory > device_total_memory_bytes
											   ? adapter_description.DedicatedVideoMemory - device_total_memory_bytes
											   : device_total_memory_bytes - adapter_description.DedicatedVideoMemory;

		bool better_adapter = !best_adapter.found;
		better_adapter |= adapter_name_matches && !best_adapter_name_matches;
		better_adapter |= adapter_name_matches == best_adapter_name_matches && adapter_memory_difference < best_adapter_memory_difference;
		if (!better_adapter)
			continue;

		best_adapter.found					= true;
		best_adapter.luid_high_part			= static_cast<unsigned int>(adapter_description.AdapterLuid.HighPart);
		best_adapter.luid_low_part			= static_cast<unsigned int>(adapter_description.AdapterLuid.LowPart);
		best_adapter.dedicated_memory_bytes = adapter_description.DedicatedVideoMemory;

		best_adapter_name_matches	   = adapter_name_matches;
		best_adapter_memory_difference = adapter_memory_difference;
	}

	factory->Release();
#endif // #ifdef _WIN32

	return best_adapter;
}

bool GPUMemoryTracker::read_gpu_memory_performance_counters(const GPUAdapterIdentifier& adapter, GPUProcessMemoryInfo& memory_info)
{
#ifdef _WIN32
	static PDH_HQUERY query								= nullptr;
	static PDH_HCOUNTER process_dedicated_usage_counter = nullptr;
	static PDH_HCOUNTER process_shared_usage_counter	= nullptr;
	static PDH_HCOUNTER adapter_dedicated_usage_counter = nullptr;
	static bool query_opened							= false;

	if (!query_opened)
	{
		query_opened = true;

		if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS)
		{
			query = nullptr;

			return false;
		}

		// The instances of the counters are named after the process and the adapter. One adapter can have several
		// physical GPUs ("phys_N"): the wildcard selects all of them and their values are summed
		std::wstring adapter_instance = std::format(L"luid_0x{:08X}_0x{:08X}_phys_*", adapter.luid_high_part, adapter.luid_low_part);
		std::wstring process_instance = std::format(L"pid_{}_", GetCurrentProcessId()) + adapter_instance;

		std::wstring process_dedicated_usage_path = L"\\GPU Process Memory(" + process_instance + L")\\Dedicated Usage";
		std::wstring process_shared_usage_path	  = L"\\GPU Process Memory(" + process_instance + L")\\Shared Usage";
		std::wstring adapter_dedicated_usage_path = L"\\GPU Adapter Memory(" + adapter_instance + L")\\Dedicated Usage";

		bool counters_added = true;
		counters_added &= PdhAddEnglishCounterW(query, process_dedicated_usage_path.c_str(), 0, &process_dedicated_usage_counter) == ERROR_SUCCESS;
		counters_added &= PdhAddEnglishCounterW(query, process_shared_usage_path.c_str(), 0, &process_shared_usage_counter) == ERROR_SUCCESS;
		counters_added &= PdhAddEnglishCounterW(query, adapter_dedicated_usage_path.c_str(), 0, &adapter_dedicated_usage_counter) == ERROR_SUCCESS;
		if (!counters_added)
		{
			PdhCloseQuery(query);
			query = nullptr;

			return false;
		}
	}

	if (query == nullptr)
		return false;

	if (PdhCollectQueryData(query) != ERROR_SUCCESS)
		return false;

	bool counters_read = true;
	counters_read &= sum_performance_counter_instances(process_dedicated_usage_counter, memory_info.dedicated_memory_used_bytes);
	counters_read &= sum_performance_counter_instances(process_shared_usage_counter, memory_info.shared_memory_used_bytes);
	counters_read &= sum_performance_counter_instances(adapter_dedicated_usage_counter, memory_info.adapter_dedicated_memory_used_bytes);

	return counters_read;
#else  // #ifdef _WIN32
	return false;
#endif // #ifdef _WIN32
}

bool GPUMemoryTracker::sum_performance_counter_instances(void* counter, size_t& out_sum)
{
	out_sum = 0;

#ifdef _WIN32
	// The first call only returns the size of the buffer needed for the values of all the instances
	DWORD buffer_byte_size = 0;
	DWORD instance_count   = 0;
	if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &buffer_byte_size, &instance_count, nullptr) != PDH_MORE_DATA)
		return false;

	std::vector<unsigned char> buffer(buffer_byte_size);
	PDH_FMT_COUNTERVALUE_ITEM_W* instance_values = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
	if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &buffer_byte_size, &instance_count, instance_values) != ERROR_SUCCESS)
		return false;

	for (unsigned int instance_index = 0; instance_index < instance_count; instance_index++)
		out_sum += static_cast<size_t>(instance_values[instance_index].FmtValue.largeValue);

	return true;
#else  // #ifdef _WIN32
	return false;
#endif // #ifdef _WIN32
}

GPUMemoryTracker::TrackerState& GPUMemoryTracker::get_state()
{
	static TrackerState* state = new TrackerState();

	return *state;
}

std::vector<std::string>& GPUMemoryTracker::get_thread_category_stack()
{
	thread_local std::vector<std::string> category_stack;

	return category_stack;
}

GPUMemoryCategoryScope::GPUMemoryCategoryScope(const std::string& category)
{
	GPUMemoryTracker::push_category(category);
}

GPUMemoryCategoryScope::~GPUMemoryCategoryScope()
{
	GPUMemoryTracker::pop_category();
}
