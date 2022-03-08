#include <vector>
#include <iostream>
#include "benchmarking.h"
#include "assignment.h"

using u32 = unsigned int;

template<typename T>
__global__
void add_matrices_register(T * const result_matrix,
	const T * const matrix_i, const T * const matrix_j)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the values needed to registers
	T tmp_i = matrix_i[i];
	T tmp_j = matrix_j[i];
	T tmp_result = tmp_i + tmp_j;

	// Fill the result matrix from the register operation
    result_matrix[i] = tmp_result;
}

template<typename T>
__global__
void subtract_matrices_register(T * const result_matrix,
	const T * const matrix_i, const T * const matrix_j)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the values needed to registers
	T tmp_i = matrix_i[i];
	T tmp_j = matrix_j[i];
	T tmp_result = tmp_i - tmp_j;

	// Fill the result matrix from the register operation
    result_matrix[i] = tmp_result;
}

template<typename T>
__global__
void multiply_matrices_register(T * const result_matrix,
	const T * const matrix_i, const T * const matrix_j)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the values needed to registers
	T tmp_i = matrix_i[i];
	T tmp_j = matrix_j[i];
	T tmp_result = tmp_i * tmp_j;

	// Fill the result matrix from the register operation
    result_matrix[i] = tmp_result;
}

template<typename T>
__global__
void modulo_matrices_register(T * const result_matrix,
	const T * const matrix_i, const T * const matrix_j)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the values needed to registers
	T tmp_i = matrix_i[i];
	T tmp_j = matrix_j[i];
	T tmp_result = tmp_i % tmp_j;

	// Fill the result matrix from the register operation
    result_matrix[i] = tmp_result;
}

void run_4_kernels_shared(u32 * results, const u32 * const data1, const u32 * const data2,
const u32 n_blocks, const u32 block_size) {
    add_matrices_register<<<n_blocks, block_size>>>(results, data1, data2);
    subtract_matrices_register<<<n_blocks, block_size>>>(results, data1, data2);
    multiply_matrices_register<<<n_blocks, block_size>>>(results, data1, data2);
    modulo_matrices_register<<<n_blocks, block_size>>>(results, data1, data2);
}

void run_registers(u32 * results, const u32 * const data1, const u32 * const data2,
const u32 n_blocks, const u32 block_size, const size_t array_size) {
    // Allocate results buffer
    u32 * device_results;
    cudaMallocHost((void **)&device_results, array_size * sizeof(u32));
    u32 * arr1;
    cudaMallocHost((void **)&arr1, array_size * sizeof(u32));
    u32 * arr2;
    cudaMallocHost((void **)&arr2, array_size * sizeof(u32));
    
    // Copy data1 memory to GPU memory
    cudaMemcpy(arr1, data1,
        array_size * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(arr2, data2,
        array_size * sizeof(u32), cudaMemcpyHostToDevice);

    run_4_kernels_shared(device_results, arr1, arr2, n_blocks, block_size);

	cudaMemcpy(results, device_results,
        array_size * sizeof(u32), cudaMemcpyDeviceToHost);

    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(device_results);
	cudaDeviceReset();
}

int main(int argc, char * argv[]) {
    // Parse command line
    unsigned int block_size = 0; // Threads per block
	unsigned int n_threads = 0; // Total threads we want
	unsigned int n_blocks = 0; // Number of blocks to hold all the threads
	unsigned int N_INTS = 0;
    
    if (argc == 4) {
        n_threads = std::stol(std::string(argv[1]));
		block_size = std::stol(std::string(argv[2]));
		n_blocks = (n_threads / block_size) > 0 ? n_threads / block_size : 1;
		N_INTS = std::stol(std::string(argv[3]));
    }
    else {
        std::cout << "Usage: " << argv[0] << " [block size] [number of threads per block]" << std::endl;
		return 0;
    }

    // Allocate host memory ones vectors for 2 separate runs
    std::vector<u32> ones(N_INTS, 1);
    std::vector<u32> twos(N_INTS, 2);

    // Allocate two host destination vectors
    std::vector<u32> dest(N_INTS);

    // Run shared memory 4 kernels
    TIC();
    run_registers(dest.data(), ones.data(), twos.data(), n_blocks, block_size, N_INTS);
    std::cout << "Register kernel took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    print_vector(dest);
}
