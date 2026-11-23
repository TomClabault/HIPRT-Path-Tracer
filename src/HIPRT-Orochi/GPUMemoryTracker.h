/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_MEMORY_TRACKER_H
#define GPU_MEMORY_TRACKER_H

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

enum class GPUMemoryType
{
	// Memory allocated on the GPU (VRAM)
	DEVICE,
	// Page-locked host memory: system RAM, not VRAM
	HOST_PINNED,
};

/**
 * Memory that the operating system attributes to this process on the GPU used for rendering, as
 * reported by the "GPU Process Memory" performance counters (the numbers of the task manager).
 *
 * It counts everything the process allocated on that GPU, whatever the API (HIP buffers, HIPRT internals, compiled kernels,
 * OpenImageDenoise, OpenGL textures and buffers, ...) and is what the tracked allocations are compared against to show how much
 * memory is not tracked
 */
struct GPUProcessMemoryInfo
{
	// Whether or not the operating system could be queried. False on platforms other than Windows
	// or if the GPU could not be found among the DXGI adapters
	bool available = false;

	// Memory used by this process in the dedicated memory of the GPU
	size_t dedicated_memory_used_bytes = 0;
	// Memory used by this process in the system memory shared with the GPU
	size_t shared_memory_used_bytes = 0;

	// Memory used by all the processes in the dedicated memory of the GPU
	size_t adapter_dedicated_memory_used_bytes = 0;
	// Size of the dedicated memory of the GPU
	size_t adapter_dedicated_memory_bytes = 0;
};

/**
 * Keeps track of every GPU allocation of the renderer and attributes each one of them to a category
 * so that the memory usage of the renderer can be broken down in the UI.
 *
 * Allocations are identified by a key that is unique to the allocation: usually the device pointer
 * of the allocation but it can be any pointer that stays the same for the lifetime of the allocation.
 *
 * The category of an allocation is the innermost GPUMemoryCategoryScope active on the allocating thread.
 * A buffer that is reallocated outside of any scope (resized from the UI for example) keeps the
 * category it was in before its reallocation: the caller passes it as the fallback category.
 *
 * Categories are written "Group/Name": the UI groups the categories by the part before the '/'
 */
class GPUMemoryTracker
{
public:
	static const std::string UNATTRIBUTED_CATEGORY;

	static const std::string SCENE_GEOMETRY_CATEGORY;
	static const std::string SCENE_BVH_CATEGORY;
	static const std::string SCENE_MATERIALS_CATEGORY;
	static const std::string SCENE_TEXTURES_CATEGORY;
	static const std::string SCENE_EMISSIVE_TRIANGLES_CATEGORY;
	static const std::string SCENE_ENVMAP_CATEGORY;

	static const std::string LIGHT_SAMPLING_POWER_ALIAS_TABLE_CATEGORY;
	static const std::string LIGHT_SAMPLING_LIGHT_TREE_ATS_CATEGORY;
	static const std::string LIGHT_SAMPLING_LIGHT_TREE_SG_CATEGORY;

	static const std::string RENDERER_FRAMEBUFFERS_CATEGORY;
	static const std::string RENDERER_PER_PIXEL_BUFFERS_CATEGORY;
	static const std::string RENDERER_ADAPTIVE_SAMPLING_CATEGORY;
	static const std::string RENDERER_STATUS_BUFFERS_CATEGORY;
	static const std::string RENDERER_BVH_TRAVERSAL_STACK_CATEGORY;
	static const std::string RENDERER_BSDF_PRECOMPUTED_TABLES_CATEGORY;
	static const std::string RENDERER_DENOISER_CATEGORY;

	static const std::string RENDER_PASSES_GROUP;

	/**
	 * Registers an allocation of 'byte_size' bytes.
	 *
	 * Registering a key that is already registered replaces the previous allocation of that key.
	 *
	 * 'fallback_category' is the category used if no GPUMemoryCategoryScope is active on the calling thread.
	 * If it is empty too, the allocation is attributed to UNATTRIBUTED_CATEGORY
	 */
	static void register_allocation(const void* allocation_key,
									size_t byte_size,
									GPUMemoryType memory_type			 = GPUMemoryType::DEVICE,
									const std::string& fallback_category = "");
	/**
	 * No effect if the key isn't registered
	 */
	static void unregister_allocation(const void* allocation_key);

	/**
	 * Returns the category of a registered allocation or an empty string if the key isn't registered
	 */
	static std::string get_allocation_category(const void* allocation_key);

	/**
	 * Returns the total size in bytes of the allocations of each category for the given memory type.
	 * Categories with no allocation left are not returned
	 */
	static std::map<std::string, size_t> get_category_byte_sizes(GPUMemoryType memory_type);
	static size_t get_total_byte_size(GPUMemoryType memory_type);

	/**
	 * Category of the innermost GPUMemoryCategoryScope of the calling thread or an empty string if there is none
	 */
	static std::string get_current_category();
	static void push_category(const std::string& category);
	static void pop_category();

	/**
	 * Returns the memory used by this process on the GPU with the given name and total memory.
	 *
	 * The performance counters are read at most twice per second: calls in between return the previous values
	 */
	static GPUProcessMemoryInfo query_process_memory_info(const std::string& device_name, size_t device_total_memory_bytes);

private:
	/**
	 * Identifies a GPU for the operating system
	 */
	struct GPUAdapterIdentifier
	{
		bool found = false;

		// LUID of the adapter, used in the instance names of the performance counters
		unsigned int luid_high_part = 0;
		unsigned int luid_low_part	= 0;

		size_t dedicated_memory_bytes = 0;
	};

	/**
	 * Finds the DXGI adapter of the GPU with the given name and total memory
	 */
	static GPUAdapterIdentifier find_gpu_adapter(const std::string& device_name, size_t device_total_memory_bytes);
	/**
	 * Reads the "GPU Process Memory" and "GPU Adapter Memory" performance counters of the given adapter into 'memory_info'.
	 * Returns false if the counters could not be read
	 */
	static bool read_gpu_memory_performance_counters(const GPUAdapterIdentifier& adapter, GPUProcessMemoryInfo& memory_info);
	/**
	 * Sums the values of all the instances matched by a performance counter with a wildcard instance name.
	 * 'counter' is a PDH_HCOUNTER
	 */
	static bool sum_performance_counter_instances(void* counter, size_t& out_sum);

	struct TrackedAllocation
	{
		std::string category;

		size_t byte_size = 0;

		GPUMemoryType memory_type = GPUMemoryType::DEVICE;
	};

	struct TrackerState
	{
		std::mutex mutex;

		std::unordered_map<const void*, TrackedAllocation> allocations;
	};

	/**
	 * The state is allocated on first use and never destroyed: GPU buffers that are global or static objects
	 * register themselves during static initialization and unregister themselves during static destruction,
	 * in an order relative to the tracker that isn't specified
	 */
	static TrackerState& get_state();
	static std::vector<std::string>& get_thread_category_stack();
};

/**
 * Attributes all the GPU allocations made on the current thread during the lifetime of
 * this object to the given category (unless a more nested scope is created)
 */
class GPUMemoryCategoryScope
{
public:
	GPUMemoryCategoryScope(const std::string& category);
	~GPUMemoryCategoryScope();

	GPUMemoryCategoryScope(const GPUMemoryCategoryScope& other)			   = delete;
	GPUMemoryCategoryScope& operator=(const GPUMemoryCategoryScope& other) = delete;
};

#endif // #ifndef GPU_MEMORY_TRACKER_H
