#include <complex>
#include <vector>
#include <iostream>

template<typename T>
inline void print_vector(const std::vector<T> & vec) {
	std::cout << "[";
    for (auto i : vec) {
        std::cout << +i << ",";
    }
	std::cout << "]\n";
}

template<typename T>
inline void print_complex_vector(const std::vector<std::complex<T>> & vec) {
	std::cout << "[";
    for (auto i : vec) {
        std::cout << "(" << i.real() << "+" << i.imag() << "j),";
    }
	std::cout << "]\n";
}

const float PI = std::acos(-1);
