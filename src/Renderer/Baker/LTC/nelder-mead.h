#ifndef NELDER_MEAD_CPP_H
#define NELDER_MEAD_CPP_H

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace nelder_mead {

template <typename T> class Vec {
  public:
    Vec() {}
    Vec(unsigned int n) : n(n) {
        val.resize(n, 0);
    }
    Vec(std::initializer_list<T> c) {
        n = c.size();
        val.resize(n);
        std::copy(c.begin().c.end(), val.begin());
    }
    Vec(const Vec &lhs) {
        val = lhs.val;
        n   = lhs.n;
    }
    Vec(const std::vector<T> &lhs) {
        val = lhs;
        n   = lhs.size();
    }
    T operator()(unsigned int idx) const {
        if (idx >= n) {
            throw std::range_error("Element access out of range");
        }
        return val[idx];
    }
    T &operator()(unsigned int idx) {
        if (idx >= n) {
            throw std::range_error("Element access out of range");
        }
        return val[idx];
    }

    Vec operator=(const Vec &rhs) {
        val = rhs.val;
        n   = rhs.n;
        return *this;
    }

    Vec operator=(const std::vector<T> &rhs) {
        val = rhs;
        n   = rhs.size();
        return *this;
    }

    Vec operator+(const Vec &rhs) const {
        Vec lhs(n);
        for (unsigned int i = 0; i < n; i++) {
            lhs.val[i] = val[i] + rhs.val[i];
        }
        return lhs;
    }
    Vec operator-(const Vec &rhs) const {
        Vec lhs(n);
        for (unsigned int i = 0; i < n; i++) {
            lhs.val[i] = val[i] - rhs.val[i];
        }
        return lhs;
    }

    Vec operator/(T rhs) const {
        Vec lhs(n);
        for (unsigned int i = 0; i < n; i++) {
            lhs.val[i] = val[i] / rhs;
        }
        return lhs;
    }
    Vec &operator+=(const Vec &rhs) {
        if (n != rhs.n)
            throw std::invalid_argument(
                "The two vectors must have the same length");
        for (unsigned int i = 0; i < n; i++) {
            val[i] += rhs.val[i];
        }
        return *this;
    }

    Vec &operator-=(const Vec &rhs) {
        if (n != rhs.n)
            throw std::invalid_argument(
                "The two vectors must have the same length");
        for (unsigned int i = 0; i < n; i++) {
            val[i] -= rhs.val[i];
        }
        return *this;
    }

    Vec &operator*=(T rhs) {
        for (unsigned int i = 0; i < n; i++) {
            val[i] *= rhs;
        }
        return *this;
    }

    Vec &operator/=(T rhs) {
        for (unsigned int i = 0; i < n; i++) {
            val[i] /= rhs;
        }
        return *this;
    }
    unsigned int size() const {
        return n;
    }
    unsigned int resize(unsigned int _n) {
        val.resize(_n);
        n = _n;
        return n;
    }
    T length() const {
        T ans = 0;
        for (unsigned int i = 0; i < n; i++) {
            ans += (val[i] * val[i]);
        }
        return std::sqrt(ans);
    }
    std::vector<T> &vec() {
        return val;
    }
    friend Vec operator*(T a, const Vec &b) {
        Vec c(b.size());
        for (unsigned int i = 0; i < b.size(); i++) {
            c.val[i] = a * b.val[i];
        }
        return c;
    }

  private:
    std::vector<T> val;
    unsigned int   n;
};

template <typename Function, typename T = double>
std::vector<T> find_min(Function &func, std::vector<T> &initial_point,
                        bool                               adaptive = false,
                        const std::vector<std::vector<T>> &initial_simplex = {},
                        T tol_fun = 1e-3, T tol_x = 1e-3,
                        unsigned int max_iter      = 50000,
                        unsigned int max_fun_evals = 50000) {
    unsigned int func_evals_count = 0;
    auto         f                = [&](Vec<T> &p) {
        func_evals_count++;
        return func(p.vec());
    };

    // Getting the dimension of function input
    unsigned int dimension = initial_point.size();
    if (dimension <= 0)
        throw std::invalid_argument(
            "A starting point must have at least one dimension.");

    // Setting parameters
    T alpha, beta, gamma, delta;
    if (adaptive) {
        // Using the results of doi:10.1007/s10589-010-9329-3
        alpha = 1;
        beta  = 1 + 2 / dimension;
        gamma = 0.75 - 1 / (2 * dimension);
        delta = 1 - 1 / dimension;
    } else {
        alpha = 1;
        beta  = 2;
        gamma = 0.5;
        delta = 0.5;
    }

    // Generate initial simplex
    std::vector<Vec<T>> simplex(dimension + 1);
    if (initial_simplex.empty()) {
        simplex[0] = initial_point;
        for (unsigned int i = 1; i <= dimension; i++) {
            Vec<T> p(initial_point);
            T tau = (p(i - 1) < 1e-6 and p(i - 1) > -1e-6) ? 0.00025 : 0.05;
            p(i - 1) += tau;
            simplex[i] = p;
        }
    } else {
        if (initial_simplex.size() != dimension + 1)
            throw std::invalid_argument(
                "The initial simplex must have dimension + 1 elements");
        for (unsigned int i = 0; i < initial_simplex.size(); i++) {
            simplex[i] = initial_simplex[i];
        }
    }

    std::vector<std::pair<bool, T>> value_cache(dimension + 1);
    for (auto &v : value_cache) {
        v.first = false;
    }
    unsigned int biggest_idx  = 0;
    unsigned int smallest_idx = 0;
    T            biggest_val;
    T            second_biggest_val;
    T            smallest_val;
    while (max_iter--) {
        // Find the points that generate the biggest, second biggest and
        // smallest value
        T val;
        if (not value_cache[0].first) {
            val                   = f(simplex[0]);
            value_cache[0].first  = true;
            value_cache[0].second = val;
        } else
            val = value_cache[0].second;
        biggest_val        = val;
        smallest_val       = val;
        second_biggest_val = val;
        biggest_idx        = 0;
        smallest_idx       = 0;
        for (unsigned int i = 1; i < simplex.size(); i++) {
            T val;
            if (not value_cache[i].first) {
                val                   = f(simplex[i]);
                value_cache[i].first  = true;
                value_cache[i].second = val;
            } else {
                val = value_cache[i].second;
            }
            if (val > biggest_val) {
                biggest_idx = i;
                biggest_val = val;
            } else if (val < smallest_val) {
                smallest_idx = i;
                smallest_val = val;
            }
        }

        // Calculate the difference of function values and the distance between
        // points in the simplex, so that we can know when to stop the
        // optimization
        T max_val_diff   = 0;
        T max_point_diff = 0;
        for (unsigned int i = 0; i < simplex.size(); i++) {
            T val = value_cache[i].second;
            if (i != biggest_idx and val > second_biggest_val) {
                second_biggest_val = val;
            } else if (i != smallest_idx) {
                if (std::abs(val - smallest_val) > max_val_diff)
                    max_val_diff = std::abs(val - smallest_val);
                T diff = (simplex[i] - simplex[smallest_idx]).length();
                if (diff > max_point_diff)
                    max_point_diff = diff;
            }
        }
        if ((max_val_diff <= tol_fun and max_point_diff <= tol_x) or
            (func_evals_count >= max_fun_evals) or (max_iter == 0)) 
        {
            if (max_iter == 0)
                std::cout << "Max iterations exceeded" << std::endl;
            else if (func_evals_count >= max_fun_evals)
                std::cout << "Maximum objective function eval count exceeded" << std::endl;
            else
                std::cout << "Target tolerance reached" << std::endl;

            std::vector<T> res = simplex[smallest_idx].vec();
            return res;
        }

        // Calculate the centroid
        Vec<T> x_bar(dimension);
        for (unsigned int i = 0; i < dimension; i++)
            x_bar(i) = 0;
        for (unsigned int i = 0; i < simplex.size(); i++) {
            if (i != biggest_idx)
                x_bar += simplex[i];
        }
        x_bar /= dimension;

        // Calculate the reflection point
        Vec<T> x_r            = x_bar + alpha * (x_bar - simplex[biggest_idx]);
        T      reflection_val = f(x_r);
        if (reflection_val < smallest_val) {
            // Expansion
            Vec<T> x_e           = x_bar + beta * (x_r - x_bar);
            T      expansion_val = f(x_e);
            if (expansion_val < reflection_val) {
                simplex[biggest_idx]            = x_e;
                value_cache[biggest_idx].second = expansion_val;
            } else {
                simplex[biggest_idx]            = x_r;
                value_cache[biggest_idx].second = reflection_val;
            }
        } else if (reflection_val >= second_biggest_val) {
            // Contraction
            Vec<T> x_c(dimension);
            bool   outside = false;
            if (reflection_val < biggest_val) {
                // Outside contraction
                outside = true;
                x_c     = x_bar + gamma * (x_r - x_bar);
            } else if (reflection_val >= biggest_val) {
                // Inside contraction
                x_c = x_bar - gamma * (x_r - x_bar);
            }
            T contraction_val = f(x_c);
            if ((outside and contraction_val <= reflection_val) or
                (not outside and contraction_val <= biggest_val)) {
                simplex[biggest_idx]            = x_c;
                value_cache[biggest_idx].second = contraction_val;
            } else {
                // Shrinking
                for (unsigned int i = 0; i < dimension; i++) {
                    if (i != smallest_idx) {
                        simplex[i] =
                            simplex[smallest_idx] +
                            delta * (simplex[i] - simplex[smallest_idx]);
                        value_cache[i].first = false;
                    }
                }
            }
        } else {
            // Reflection is good enough
            simplex[biggest_idx]            = x_r;
            value_cache[biggest_idx].second = reflection_val;
        }
    }

    std::cout << "Max iterations exceeded" << std::endl;

    std::vector<T> res = simplex[smallest_idx].vec();
    return res;
}

}; // namespace nelder_mead
#endif
