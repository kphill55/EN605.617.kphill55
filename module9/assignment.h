#include <vector>
#include <iostream>
#include <thrust/host_vector.h>

template<typename T>
inline void print_thrust_vector(const thrust::host_vector<T> & vec) {
	std::cout << "[";
    for (auto i : vec) {
        std::cout << +i << ",";
    }
	std::cout << "]\n";
}
