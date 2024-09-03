/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "PerformanceMetricsComputer.h"

#include <cmath>
#include <iostream>

float PerformanceMetricsComputer::data_getter(void* data, int index)
{
	return static_cast<double*>(data)[index];
}

void PerformanceMetricsComputer::init_key(const std::string& key)
{
	if (m_key_init.find(key) == m_key_init.end())
	{
		// Key data not init yet
		m_key_init[key] = true;

		m_values[key] = std::vector<double>(m_window_size, 0.0f);
		m_values_count[key] = 0;
		m_values_sum[key] = 0.0;
		m_values_sum_of_squares[key] = 0.0;
		m_data_indices[key] = 0;
		m_min_max_data[key] = std::multiset<double>();
	}
}

std::vector<double>& PerformanceMetricsComputer::get_data(const std::string& key)
{
	init_key(key);

	return m_values.find(key)->second;
}

int PerformanceMetricsComputer::get_value_count(const std::string& key)
{
	return m_values_count[key];
}

int PerformanceMetricsComputer::get_data_index(const std::string& key)
{
	return m_data_indices[key];
}

void PerformanceMetricsComputer::add_value(const std::string& key, double new_value)
{
	init_key(key);

	// Where are we going to insert the next value in the m_values vector
	unsigned int next_index = m_data_indices[key];
	m_data_indices[key] = m_data_indices[key] + 1;
	if (m_data_indices[key] == m_window_size)
		m_data_indices[key] = 0;

	double removed_value = m_values[key].at(next_index);
	m_values[key].at(next_index) = new_value;

	// Whether or not we've reached the maximum number of values we can
	// store. If true, we are now removing a value every single time we want to insert one
	bool at_capacity = false;
	int& current_value_count = m_values_count[key];
	if (current_value_count < m_window_size && next_index < current_value_count)
		// This is a special case when we just resized the window to 
		// a size larger than the previous one. 
		// 
		// This can cause issues in the following situation:
		//
		//  - The window size is 100. We have input 180 values so far. 
		//		This means that we're at capacity and we've stared from the beginning, 
		//		overriding the first 80 values
		//	- The window is resized to a size of 250
		//	- We keep adding new values and we are currently at the value 230.
		//	- Without this if() statement right here, we would have m_values_count = 250, not 230
		//		because we started incrementing m_values_count[key] right when we resized the window
		//		(when we were overriding the 80th value). We counted 20 values too many between the 80th
		//		and the 100th.
		//		We're now "at_capacity" at value 230 instead of at value 250 and that causes issues in the
		//		rest of the perf metrics computer so this if() statement here prevents incrementing m_values_count
		//		"falsely"
		at_capacity = false;
	else
		at_capacity = ++m_values_count[key] > m_window_size;
	m_values_count[key] -= at_capacity;

	// Updating the sums and sums of squares according to the value we removed / added
	m_values_sum[key] -= removed_value;
	m_values_sum[key] += new_value;
	m_values_sum_of_squares[key] -= removed_value * removed_value;
	m_values_sum_of_squares[key] += new_value * new_value;

	auto& multiset = m_min_max_data.find(key)->second;
	if (at_capacity)
	{
		auto position = multiset.find(removed_value);
		multiset.erase(position);
	}
	multiset.insert(new_value);
}

double PerformanceMetricsComputer::get_current_value(const std::string& key)
{
	// m_data_indices[key] is the index of the value that we're going to insert next
	// but we want the index of the value last inserted so we -1 that value
	int current_index = m_data_indices[key];
	int previous_index = current_index - 1;
	if (previous_index == -1)
		previous_index = m_values_count[key] - 1;

	return m_values[key][previous_index];
}

double PerformanceMetricsComputer::get_average(const std::string& key)
{
	if (m_values_count[key] == 0)
	{
		std::cerr << "Trying to get the average value of the key \"" << key << "\" but this key has no values!" << std::endl;

		return -1.0;
	}

	return m_values_sum[key] / m_values_count[key];
}

double PerformanceMetricsComputer::get_variance(const std::string& key)
{
	if (m_values_count[key] == 0)
	{
		std::cerr << "Trying to get the variance value of the key \"" << key << "\" but this key has no values!" << std::endl;

		return -1.0;
	}

	double average = get_average(key);
	return m_values_sum_of_squares[key] / m_values_count[key] - average * average;
}

double PerformanceMetricsComputer::get_standard_deviation(const std::string& key)
{
	if (m_values_count[key] == 0)
	{
		std::cerr << "Trying to get the standard deviation value of the key \"" << key << "\" but this key has no values!" << std::endl;

		return -1.0;
	}

	return std::sqrt(get_variance(key));
}

double PerformanceMetricsComputer::get_min(const std::string& key)
{
	if (m_min_max_data[key].size() == 0)
	{
		std::cerr << "Trying to get the minimum value of the key \"" << key << "\" but this key has no values!" << std::endl;

		return -1.0;
	}

	return *m_min_max_data[key].begin();
}

double PerformanceMetricsComputer::get_max(const std::string& key)
{
	if (m_min_max_data[key].size() == 0)
	{
		std::cerr << "Trying to get the maximum value of the key \"" << key << "\" but this key has no values!" << std::endl;

		return -1.0;
	}

	// rbegin() is the last element
	// end() would be past the last element so we're not using end() here
	return *m_min_max_data[key].rbegin();
}

int PerformanceMetricsComputer::get_window_size() const
{
	return m_window_size;
}

int& PerformanceMetricsComputer::get_window_size()
{
	return m_window_size;
}

void PerformanceMetricsComputer::resize_window(int new_size)
{
	if (m_window_size == new_size)
		return;

	resize_values_vectors(new_size);
	recompute_data(new_size);

	m_window_size = new_size;
}

void PerformanceMetricsComputer::resize_values_vectors(int new_size)
{
	for (auto& pair_kv : m_values)
		// Resizing the values vectors.
		// If resizing to a smaller size, .resize() throws away the elements at the end.
		// If resizing to a greater size, .resize() inserts new element at the end.
		// This is what we want.
		pair_kv.second.resize(new_size, 0.0);
}

void PerformanceMetricsComputer::recompute_data(int new_size)
{
	// This function could be recomputing elements in a smarter way but because recompute_data
	// isn't expected to be called that often at all, let's keep it simple

	if (new_size > m_window_size)
		// Nothing to recompute, new elements will be added later
		return;

	// Else, elements were removed from the end, we need to recompute the sum,
	// sums of squares, ... without taking these removed elements into account

	for (auto pair_kv : m_values)
	{
		const std::string& key = pair_kv.first;

		if (m_values_count[key] < new_size)
		{
			// There wasn't enough values so resizing the vector didn't remove any value,
			// nothing to recompute

			continue;
		}

		m_values_count[key] = std::min(m_values_count[key], new_size);
		m_values_sum[key] = 0.0;
		m_values_sum_of_squares[key] = 0.0;
		m_data_indices[key] = m_data_indices[key] > new_size ? 0 : m_data_indices[key];
		m_min_max_data[key].clear();

		std::vector<double>& values = pair_kv.second;
		for (int i = 0; i < new_size; i++)
		{
			double value = values[i];

			m_min_max_data[key].insert(value);
			m_values_sum[key] += value;
			m_values_sum_of_squares[key] += value * value;
		}
	}
}
