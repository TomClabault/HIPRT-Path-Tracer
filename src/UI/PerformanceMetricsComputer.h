/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef PERFORMANCE_METRICS_COMPUTER
#define PERFORMANCE_METRICS_COMPUTER

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class PerformanceMetricsComputer
{
public:
	static float data_getter(void* data, int index);

	void init_key(const std::string& key);
	std::vector<double>& get_data(const std::string& key);
	int get_value_count(const std::string& key);
	int get_data_index(const std::string& key);

	void add_value(const std::string& key, double value);

	double get_current_value(const std::string& key);
	double get_average(const std::string& key);
	double get_variance(const std::string& key);
	double get_standard_deviation(const std::string& key);
	double get_min(const std::string& key);
	double get_max(const std::string& key);

	int get_window_size() const;
	int& get_window_size();
	void resize_window(int new_size);

private:
	void resize_values_vectors(int new_size);

	/**
	 * This function is called when resizing the window. 
	 * Because we have thrown away elements that were at the end of the vectors,
	 * we're going to have to recompute the average, sums, ... not to take into account
	 * the elements that were removed
	 */
	void recompute_data(int new_size);

	int m_window_size = 100;

	// Whether or not we have already initialized all the unordered maps for a given key
	std::unordered_map<std::string, bool> m_key_init;
	// The values for the given key. There are m_values_count[key] valid values
	// in the vector at the given key
	std::unordered_map<std::string, std::vector<double>> m_values;
	// How many valid values are present in m_values from the beginning of its vector<double>
	std::unordered_map<std::string, int> m_values_count;
	// Sum of the last m_window_size values
	std::unordered_map<std::string, double> m_values_sum;
	// Sum of the last m_window_size values squared
	std::unordered_map<std::string, double> m_values_sum_of_squares;
	// Where is the next value going to be inserted in m_values
	std::unordered_map<std::string, unsigned int> m_data_indices;
	// Using a multiset here allows to easily retrieve the minimum and maximum of the values
	std::unordered_map<std::string, std::multiset<double>> m_min_max_data;
};

#endif
