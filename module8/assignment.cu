#include <vector>
#include <iostream>
#include "benchmarking.h"
#include "assignment.h"
#include <cufft.h>
#include <cublas_v2.h>

using c32 = std::complex<float>;
using c64 = std::complex<double>;
using u32 = unsigned int;

void run_fft(c32 * results, const u32 array_num_elements) {
	// Create FFT
    cufftHandle plan;

    // Set FFT plan
    cufftPlan1d(&plan, array_num_elements, CUFFT_C2C, 1);

    
    // Create cuda stream
	cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // Set FFT to stream1
    cufftSetStream(plan, stream1);

	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start); 
  	cudaEventCreate(&stop);
    float elapsedTime = -1;

    // Allocate device memory for FFT
    cufftComplex * device_results;
    cudaMallocManaged((void **)&device_results, array_num_elements * sizeof(c32), cudaMemAttachGlobal);

	// Start event
	cudaEventRecord(start, stream1);
    TIC();

    // Copy data1 memory to GPU memory
    cudaMemcpyAsync(device_results, results,
        array_num_elements * sizeof(c32), cudaMemcpyHostToDevice);

    cufftExecC2C(plan, device_results, device_results, CUFFT_FORWARD);

    // Flush a message through std out while the GPU works
    std::cout << "I, the CPU, am printing this while the GPU does an FFT..." << std::endl;

	cudaMemcpyAsync(results, device_results,
        array_num_elements * sizeof(c32), cudaMemcpyDeviceToHost);

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

    // Cleanup
    cudaFree(device_results);
	cudaStreamDestroy(stream1);
    cufftDestroy(plan);
}

void run_matrix_mult(c32 * results, const c32 * const data1, const c32 * const data2, const u32 x_dim, const u32 y_dim) {
    cublasHandle_t handle;

    // Create cuda stream
	cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // Create blas
    cublasCreate(&handle);
    cublasSetStream(handle, stream1);

	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start); 
  	cudaEventCreate(&stop);
    float elapsedTime = -1;

    // Allocate
    cuComplex * device_results;
    cudaMallocManaged((void **)&device_results, x_dim * y_dim * sizeof(c32), cudaMemAttachGlobal);
    cuComplex * arr1;
    cudaMallocManaged((void **)&arr1, x_dim * y_dim * sizeof(c32), cudaMemAttachGlobal);
    cuComplex * arr2;
    cudaMallocManaged((void **)&arr2, x_dim * y_dim * sizeof(c32), cudaMemAttachGlobal);

	// Start event
	cudaEventRecord(start, stream1);
    TIC();
    
    // Copy the host matrix to device
    cublasSetMatrixAsync(x_dim, y_dim, sizeof(c32), data1, x_dim, arr1, y_dim, stream1);
    cublasSetMatrixAsync(x_dim, y_dim, sizeof(c32), data2, x_dim, arr2, y_dim, stream1);
    
    // Execute
    cuComplex alpha{1,0};
    cuComplex beta{1,0};
    cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, x_dim, y_dim, x_dim, &alpha, arr1, x_dim, arr2, y_dim, &beta, device_results, x_dim);

    // Flush a message through std out while the GPU works
    std::cout << "I, the CPU, am printing this while the GPU does a matrix multiplication..." << std::endl;

    // Retrieve the device matrix back to the host
	cublasGetMatrixAsync(x_dim, y_dim, sizeof(c32), device_results, x_dim, results, y_dim, stream1);

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


    // Clean up
    cublasDestroy(handle);
	cudaStreamDestroy(stream1);
    cudaFree(device_results);
    cudaFree(arr1);
    cudaFree(arr2);
}

int main(int argc, char * argv[]) {
    // Parse command line
	unsigned int N_COMPLEX = 0;
    
    if (argc == 2) {
		N_COMPLEX = std::stol(std::string(argv[1]));
    }
    else {
        std::cout << "Usage: " << argv[0] << " [block size] [number of threads per block] [array size" << std::endl;
		return 0;
    }

    // "Receive" a buffer of complex samples of an "unknown signal"
    float initial_phase = 0.0f;
	float angular_frequency = PI;
	std::vector<c32> rx(N_COMPLEX);

	for (int i = 0; i < rx.size(); ++i) {
		rx[i] = std::polar(1.0f, i * angular_frequency + initial_phase);
	}

	// Create 2d array identity matrix
	std::vector<std::vector<c32>> i_matrix(N_COMPLEX, std::vector<c32> (N_COMPLEX, 0));
	
	for (int i = 0; i<N_COMPLEX; ++i) {
		i_matrix[i][i] = c32{1.0f, 0.0f};
	}
	
	// Create 2d array of twos
	std::vector<std::vector<c32>> twos(N_COMPLEX, std::vector<c32> (N_COMPLEX, {0, 2}));

    std::vector<std::vector<c32>> results(N_COMPLEX, std::vector<c32> (N_COMPLEX));

    // Run FFT
    TIC();
    run_fft(rx.data(), N_COMPLEX);
    std::cout << "Stream/event kernel took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    print_complex_vector(rx);

    // Run matrix mult
    TIC();
    run_matrix_mult(results.begin()->data(), twos.begin()->data(), i_matrix.begin()->data(), N_COMPLEX, N_COMPLEX);
    std::cout << "Stream/event kernel took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    print_complex_vector(results.front());    
}
