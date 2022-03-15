#include <vector>
#include <iostream>
#include "benchmarking.h"
#include "assignment.h"

using u32 = unsigned int;

void run_4_kernels(u32 * results, const u32 * const data1, const u32 * const data2,
const u32 n_blocks, const u32 block_size, cudaStream_t & stream) {
    add_matrices<<<n_blocks, block_size, 0, stream>>>(results, data1, data2);
    subtract_matrices<<<n_blocks, block_size, 0, stream>>>(results, data1, data2);
    multiply_matrices<<<n_blocks, block_size, 0, stream>>>(results, data1, data2);
    modulo_matrices<<<n_blocks, block_size, 0, stream>>>(results, data1, data2);
	cudaStreamSynchronize(stream);
}

void run_kernels(u32 * results, const u32 * const data1, const u32 * const data2,
const u32 n_blocks, const u32 block_size, const size_t array_size) {
	// Create cuda stream
	cudaStream_t stream1; 
  	cudaStreamCreate(&stream1);

	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start); 
  	cudaEventCreate(&stop);
    float elapsedTime;

    // Allocate device memory
    u32 * device_results;
    cudaHostAlloc((void **)&device_results, array_size * sizeof(u32), cudaHostAllocDefault);
    u32 * arr1;
    cudaHostAlloc((void **)&arr1, array_size * sizeof(u32), cudaHostAllocDefault);
    u32 * arr2;
    cudaHostAlloc((void **)&arr2, array_size * sizeof(u32), cudaHostAllocDefault);

	// Start event
	cudaEventRecord(start, stream1);
    TIC();

    // Copy data1 memory to GPU memory
    cudaMemcpyAsync(arr1, data1,
        array_size * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(arr2, data2,
        array_size * sizeof(u32), cudaMemcpyHostToDevice);

    run_4_kernels(device_results, arr1, arr2, n_blocks, block_size, stream1);

    // Flush a message through std out while the GPU works
    std::cout << "I, the CPU, am printing this while the GPU does work..." << std::endl;

	cudaMemcpyAsync(results, device_results,
        array_size * sizeof(u32), cudaMemcpyDeviceToHost);

    // Push the stop event onto the kernel launch queue after copying the data out
    cudaEventRecord(stop, stream1);

    /*
    We must now synchronize with steam1 since this is asynchronous
    unlike stream0 which is synchronous by default
    (host waits for kernel execution to end)
    */
	cudaStreamSynchronize(stream1); // Wait for stream1 to complete queued actions
  	cudaEventSynchronize(stop); // Wait for "stop" to reach the front of the kernel queue
    
    std::cout << "CPU waited " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    
    // This should be roughly the same as tic/toc
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Event took " << elapsedTime*1000 << " microseconds in total\n";

    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(device_results);
	cudaStreamDestroy(stream1);
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
        std::cout << "Usage: " << argv[0] << " [block size] [number of threads per block] [array size" << std::endl;
		return 0;
    }

    // Allocate host memory ones vectors for 2 separate runs
    std::vector<u32> ones(N_INTS, 1);
    std::vector<u32> twos(N_INTS, 2);

    // Allocate two host destination vectors
    std::vector<u32> dest(N_INTS);

    // Run shared memory 4 kernels
    TIC();
    run_kernels(dest.data(), ones.data(), twos.data(), n_blocks, block_size, N_INTS);
    std::cout << "Stream/event kernel took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    print_vector(dest);
}
