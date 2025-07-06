/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"

extern ImGuiLogger g_imgui_logger;

// TODO make this class not a singleton but a global variable instead

/**
 * Singleton class so that threads are accessible everywhere to be .join()
 * whenever we want without having to pass them around in function calls etc...
 * 
 * This class works by creating threads and storing them in std::vectors.
 * Which std::vector is the thread going to be stored in depends on the key that is given.
 * Keys are basically used to give some kind of "name" to threads. The main use for that
 * is that all threads with the same key can be joined at the same time. So for example,
 * if you add 2 threads, both with the key 'MY_THREAD_KEY', they will both be added to
 * the same std::vector (these std::vectors are in the thread_map which is an attribute
 * of this class). Then, when you decide to join threads with the 'MY_THREAD_KEY' key,
 * all threads of the corresponding std::vector will be .join()
 */
class ThreadManager
{
public:
	static std::string COMPILE_RAY_VOLUME_STATE_SIZE_KERNEL_KEY;
	static std::string COMPILE_NEE_PLUS_PLUS_FINALIZE_ACCUMULATION_KERNEL_KEY;
	static std::string COMPILE_KERNELS_THREAD_KEY;
	// Key for the thread that will ** launch ** the threads that will precompile kernels
	// in the background (needed because ** launching ** the precompilation itself takes quite a
	// bit of time so we're doing that on a thread with this key
	static std::string GPU_RENDERER_PRECOMPILE_KERNELS_THREAD_KEY;

	static std::string RENDER_WINDOW_CONSTRUCTOR;
	static std::string RENDER_WINDOW_RENDERER_INITIAL_RESIZE;

	static std::string RENDERER_SET_ENVMAP;
	static std::string RENDERER_BUILD_BVH;
	static std::string RENDERER_UPLOAD_MATERIALS;
	static std::string RENDERER_UPLOAD_TEXTURES;
	static std::string RENDERER_UPLOAD_EMISSIVE_TRIANGLES;
	static std::string RENDERER_UPLOAD_TRIANGLE_AREAS;
	static std::string RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE;

	static std::string RENDERER_PRECOMPILE_KERNELS;
	static std::string RESTIR_DI_PRECOMPILE_KERNELS;

	static std::string SCENE_TEXTURES_LOADING_THREAD_KEY;
	static std::string SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES;
	static std::string SCENE_LOADING_COMPUTE_TRIANGLE_AREAS;
	static std::string ENVMAP_LOAD_FROM_DISK_THREAD;

	/**
	 * If the passed parameter is true, the ThreadManager will execute all
	 * started threads on the main thread instead of on a separate thread.
	 */
	static void set_monothread(bool is_monothread)
	{
		m_monothread = is_monothread;
	}

	template <typename T>
	static void set_thread_data(const std::string& key, std::shared_ptr<T> state)
	{
		m_threads_states[key] = std::static_pointer_cast<void>(state);
	}

	template <class _Fn, class... _Args>
	static void start_thread(std::string key, _Fn function, _Args... args)
	{
		const std::unordered_set<std::string>& dependencies = m_dependencies[key];
		if (!dependencies.empty())
			start_with_dependencies(dependencies, key, function, args...);
		else
		{
			if (m_monothread)
			{
				start_serial_thread(key, function, args...);

				// Creates the entry in the map if it doesn't exist. Doesn't do anything if it already exists.
				// This is so that other parts of the ThreadManager don't scream when trying to join threads
				// that haven't been started ("started" meaning that there is an entry in the map) for example
				bool empty = m_threads_map[key].empty();
			}
			else
				// Starting the thread and adding it to the list of threads for the given key
				m_threads_map[key].push_back(std::thread(function, args...));
		}
	}

	/**
	 * This function starts a thread on the main thread i.e. not asynchronously and waits for
	 * the completion of the given function before returning
	 */
	template <class _Fn, class... _Args>
	static void start_serial_thread(std::string key, _Fn function, _Args... args)
	{
		// Creating an entry in the map to 'fake' that we've started a thread
		bool empty = m_threads_map[key].empty();

		function(args...);
	}

	static void join_threads(const std::string& key)
	{
		std::lock_guard<std::mutex> lock(m_join_mutexes[key]);

		auto find = m_threads_map.find(key);
		if (find != m_threads_map.end())
		{
			if (find->second.empty())
				// No threads to wait for
				return;

			for (std::thread& thread : find->second)
			{
				// TODO: This is just for debugging. 
				// There seems to be some very rare bug in the ThreadManager where sometimes, 
				// we're trying to join (with thread.join()) below a thread that has a NULL
				// handle from the 'ParseEmissiveTrianglesKey' thread key
				//
				// UPDATE: This seems to happen when we're calling join_all_threads() while
				// we're are still starting some thread with dependencies: we end up in a situation where we're
				// trying to join on a dependecy that has already been joined by join_all_threads() so we're going to need
				// some kind of way for join_all_threads() to wait for all threads to at least have started
				if (thread.native_handle() == 0)
					Utils::debugbreak();

				if (thread.joinable())
					thread.join();
			}
		}
		else
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to joing threads with key \"%s\" but no threads have been started with this key.", key.c_str());

			return;
		}

		m_threads_map[key].clear();
	}

	/**
	 * Joins all the threads that have been started so far except the threads
	 * launch with a key in the 'execeptions' vector passed as parameter
	 */
	static void join_all_threads(const std::unordered_set<std::string>& exceptions = {})
	{
		// Joining all the threads and their dependencies
		for (const auto& key_to_threads : m_threads_map)
		{
			std::deque<std::string> dependencies_to_wait_for;
			std::deque<std::string> dependencies_to_analyze;

			const std::string& thread_key = key_to_threads.first;
			if (exceptions.find(thread_key) != exceptions.end())
				// This thread is in the exception list. Not joining these threads
				continue;

			if (!m_dependencies[thread_key].empty())
				for (const std::string& dependency : m_dependencies[thread_key])
					dependencies_to_analyze.push_back(dependency);
			// Pushing the thread key itself we want to wait for and then we'll
			// push its dependencies in front of it so that we wait for the dependencies first
			dependencies_to_wait_for.push_front(thread_key);

			while (!dependencies_to_analyze.empty())
			{
				std::string new_dependency = dependencies_to_analyze.front();
				dependencies_to_analyze.pop_front();
				dependencies_to_wait_for.push_front(new_dependency);

				const std::unordered_set<std::string>& dependencies = m_dependencies[new_dependency];
				for (const std::string& dependency : dependencies)
					dependencies_to_analyze.push_front(dependency);
			}

			std::unordered_set<std::string> dependencies_already_joined;
			for (const std::string& dependency : dependencies_to_wait_for)
			{
				if (dependencies_already_joined.find(dependency) == dependencies_already_joined.end())
				{
					// Dependency not joined yet
					dependencies_already_joined.insert(dependency);
					join_threads(dependency);
				}
			}
		}
	}

	static void detach_threads(const std::string& key)
	{
		auto find = m_threads_map.find(key);
		if (find == m_threads_map.end())
			return;

		for (std::thread& thread : find->second)
			if (thread.joinable())
				thread.detach();
	}

	/**
	 * Adds a dependecy on 'dependency_key' from 'key' such that all the threads started with key
	 * 'key' only start after all threads from 'dependency_key' are finished
	 */
	static void add_dependency(const std::string& key, const std::string& dependency_key)
	{
		m_dependencies[key].insert(dependency_key);
	}

private:
	template <class _Fn, class... _Args>
	static void start_with_dependencies(const std::unordered_set<std::string>& dependencies, const std::string& thread_key_to_start, _Fn function, _Args... args)
	{
		// These threads have a dependency

		// Executing the given function after waiting for the dependencies
		if (m_monothread)
		{
			// Waiting for the dependencies before starting the thread
			wait_for_dependencies(dependencies);

			start_serial_thread(thread_key_to_start, function, args...);
		}
		else
		{
			// Starting a thread that will wait for the dependencies before calling the given function
			m_threads_map[thread_key_to_start].push_back(std::thread([thread_key_to_start, dependencies, function, args...]() 
			{
				wait_for_dependencies(dependencies);

				std::thread function_thread(function, args...);
				function_thread.join();
			}));
		}
	}

	static void wait_for_dependencies(const std::unordered_set<std::string>& dependencies)
	{
		for (const std::string& dependency : dependencies)
			join_threads(dependency);
	}

private:
	// If true, the ThreadManager will execute all threads serially
	static bool m_monothread;

	// The states are used to keep the data that the threads need alive
	static std::unordered_map<std::string, std::shared_ptr<void>> m_threads_states;

	// The ThreadManager can hold as many thread as we want and to find the thread
	// we want amongst all the threads stored, we use keys, hence the unordered_map
	static std::unordered_map<std::string, std::vector<std::thread>> m_threads_map;

	// Because of the dependency management system, it may be possible that we call .join()
	// on the same thread (or threads with the same thread key) from multiple threads.
	// Calling .join() concurrently on the same thread is likely to result in a race condition
	// so we need a synchronization system, using mutexes
	static std::unordered_map<std::string, std::mutex> m_join_mutexes;

	// For each thread key, maps to a vector of the dependencies of these threads
	// (thread with the thread key given as key to the map)
	static std::unordered_map<std::string, std::unordered_set<std::string>> m_dependencies;
};

#endif
