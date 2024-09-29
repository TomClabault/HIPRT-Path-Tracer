/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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
	static std::string COMPILE_KERNEL_PASS_THREAD_KEY;
	static std::string SCENE_TEXTURES_LOADING_THREAD_KEY;
	static std::string SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES;
	static std::string ENVMAP_LOAD_THREAD_KEY;

	~ThreadManager();

	/**
	 * If the passed parameter is true, the ThreadManager will execute all
	 * started threads on the main thread instead of on a separate thread.
	 */
	static void set_monothread(bool monothread);

	template <typename T>
	static void set_thread_data(const std::string& key, std::shared_ptr<T> state)
	{
		m_threads_states[key] = std::static_pointer_cast<void>(state);
	}

	template <class _Fn, class... _Args>
	static void start_thread(std::string key, _Fn function, _Args... args)
	{
		std::vector<std::string>& dependencies = m_dependencies[key];
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
			{
				lock_thread_key(key);

				// Starting the thread and adding it to the list of threads for the given key
				m_threads_map[key].push_back(std::thread(function, args...));
			}
		}
	}

	template <class _Fn, class... _Args>
	static void start_with_dependencies(const std::vector<std::string>& dependencies, const std::string& thread_key_to_start, _Fn function, _Args... args)
	{
		// These threads have a dependency

		// Executing the given function after waiting for the dependencies
		if (m_monothread)
		{
			// Waiting for the condition variable to signal that the threads of
			// 'dependency' are finished (join_threads() sends the signal)
			for (const std::string& dependency : dependencies)
				m_condition_variables[dependency].second.wait(m_unique_locks[dependency]);

			start_serial_thread(thread_key_to_start, function, args...);
		}
		else
		{
			lock_thread_key(thread_key_to_start);

			// Starting a thread that will wait for the dependencies before calling the given function
			m_threads_map[thread_key_to_start].push_back(std::thread([dependencies, function, args...]() {
				// Waiting for the condition variable to signal that the threads of
				// 'dependency' are finished (join_threads() sends the signal)
				for (const std::string& dependency : dependencies)
					if (m_condition_variables.find(dependency) != m_condition_variables.end())
						//  The condition variable exists, i.e. some dependency threads have actually
						// been started (otherwise, there is no one to wait for)
						m_condition_variables[dependency].second.wait(m_unique_locks[dependency]);

				std::thread function_thread(function, args...);
				function_thread.join();
			}));
		}
	}

	static void lock_thread_key(const std::string& key)
	{
		if (m_unique_locks.find(key) == m_unique_locks.end())
			// No unique lock created yet, creating on the mutex of these threads.
			// The constructor automatically locks the mutex
			m_unique_locks[key] = std::unique_lock(m_condition_variables[key].first);
		else if (!m_unique_locks[key].owns_lock())
			// Locking the mutex of this thread key so that any thread keys with
			// dependencies on this one cannot start until this mutex is unlock
			// (the mutex will be unlocked in join_threads())
			m_unique_locks[key].lock();
	}

	/**
	 * This function starts a thread on the main thread i.e. not asynchronously and waits for
	 * the completion of the given function before returning
	 */
	template <class _Fn, class... _Args>
	static void start_serial_thread(std::string key, _Fn function, _Args... args)
	{
		function(args...);
	}

	static void join_threads(std::string key);

	/**
	 * Adds a dependecy on 'dependency_key' from 'key' such that all the threads started with key
	 * 'key' only start after all threads from 'dependency_key' are finished
	 */
	static void add_dependency(const std::string& key, const std::string& dependency_key);

private:
	// If true, the ThreadManager will execute all threads serially
	static bool m_monothread;

	// The states are used to keep the data that the threads need alive
	static std::unordered_map<std::string, std::shared_ptr<void>> m_threads_states;

	// The ThreadManager can hold as many thread as we want and to find the thread
	// we want amongst all the threads stored, we use keys, hence the unordered_map
	static std::unordered_map<std::string, std::vector<std::thread>> m_threads_map;

	// For each thread key, maps to a vector of the dependencies of these threads
	// (thread with the thread key given as key to the map)
	static std::unordered_map<std::string, std::vector<std::string>> m_dependencies;

	// For a given key, a condition variable and its mutex.
	// This allows managing dependencies between thread keys: a thread 
	// key can wait on all threads of another key to complete by waiting on the condition variable
	static std::unordered_map<std::string, std::pair<std::mutex, std::condition_variable>> m_condition_variables;
	// Unique locks used to lock the mutexes of 'm_condition_variables'
	static std::unordered_map<std::string, std::unique_lock<std::mutex>> m_unique_locks;
};

#endif
