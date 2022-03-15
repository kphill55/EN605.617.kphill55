#include <vector>
#include <iostream>

template<typename T>
__global__
void add_matrices(T * const result_matrix,
	const T * const matrix_i, const T * const matrix_j)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    result_matrix[i] = matrix_i[i] + matrix_j[i];
}

template<typename T>
__global__
void subtract_matrices(T * const result_matrix,
	const T * const matrix_i, const T * const matrix_j)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    result_matrix[i] = matrix_i[i] - matrix_j[i];
}

template<typename T>
__global__
void multiply_matrices(T * const result_matrix,
	const T * const matrix_i, const T * const matrix_j)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    result_matrix[i] = matrix_i[i] * matrix_j[i];
}

template<typename T>
__global__
void modulo_matrices(T * const result_matrix,
	const T * const matrix_i, const T * const matrix_j)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    result_matrix[i] = matrix_i[i] % matrix_j[i];
}

template<typename T>
inline void print_vector(const std::vector<T> & vec) {
	std::cout << "[";
    for (auto i : vec) {
        std::cout << +i << ",";
    }
	std::cout << "]\n";
}
