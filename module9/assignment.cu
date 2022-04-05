#include <vector>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <nvgraph.h>
#include <npp.h>
#include <npps.h>
#include "stdlib.h"
#include "inttypes.h"
#include "stdio.h"
#include "nvgraph.h"
#include "benchmarking.h"
#include "assignment.h"

using u32 = unsigned int;
#define check( a ) \
{\
nvgraphStatus_t status = (a);\
if ( (status) != NVGRAPH_STATUS_SUCCESS) {\
printf("ERROR : %d in %s : %d\n", status, __FILE__ , __LINE__ );\
exit(0);\
}\
}

template<typename T>
void run_thrust_operations(thrust::host_vector<T> & result, const thrust::host_vector<T> & x,
	const thrust::host_vector<T> & y) {

	// Copy to device
	thrust::device_vector<T> d1 = x;
	thrust::device_vector<T> d2 = y;

	// Add
	thrust::transform(x.begin(), x.end(), y.begin(), result.begin(),
                  thrust::plus<T>());
	print_thrust_vector(result);
	// Subtract
	thrust::transform(x.begin(), x.end(), y.begin(), result.begin(),
                  thrust::minus<T>());
	print_thrust_vector(result);
	// Multiply
	thrust::transform(x.begin(), x.end(), y.begin(), result.begin(),
                  thrust::multiplies<T>());
	print_thrust_vector(result);
	// Divide
	thrust::transform(x.begin(), x.end(), y.begin(), result.begin(),
                  thrust::modulus<T>());
	print_thrust_vector(result);
}

void run_npp(const Npp32f * result, const Npp32f * input, u32 array_length) {
	Npp32f * res;
	Npp32f * input_buf;
	Npp8u * scratch_buf;

	// Allocate device mem
	cudaMalloc((void **)(&res), sizeof(res));
	cudaMalloc((void **)(&input_buf), sizeof(Npp32f) * array_length);
	cudaMemcpy(&input_buf, input, sizeof(Npp32f) * array_length, cudaMemcpyHostToDevice);
	cudaMemset(res, 0, sizeof(res));

	// Allocate scratch buffer
	int nBufferSize;
	nppsMeanGetBufferSize_32f(array_length, &nBufferSize);
	cudaMalloc((void **)(&scratch_buf), nBufferSize);

	// Perform mean
	nppsMean_32f(input, array_length, res, scratch_buf);

	// Copy mean to host
	cudaMemcpy(&result, res, sizeof(Npp32f), cudaMemcpyDeviceToHost);

	// Clean up
	cudaFree(scratch_buf);
	cudaFree(input_buf);
	cudaFree(res);
}

void run_nvGRAPH() {
	// Modified example SSSP for C++ operation
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCSRTopology32I_t CSR_input;
	nvgraphCreate(&handle);
	nvgraphCreateGraphDescr(handle, &graph);
	
	const size_t n = 6, nnz = 8;
	int source_offsets[] = {0, 0, 1, 2, 4, 6, 8};
	int destination_indices[] = {0, 1, 1, 2, 2, 3, 3, 4};
	CSR_input = new(nvgraphCSRTopology32I_st);

	CSR_input->nvertices = n;
	CSR_input->nedges = nnz;
	CSR_input->source_offsets = source_offsets;
	CSR_input->destination_indices = destination_indices;

	nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32);

	uint64_t trcount = 0;
	nvgraphTriangleCount(handle, graph, &trcount);

	std::cout << "Counted " << trcount << " triangles\n";

	delete (CSR_input);
	nvgraphDestroyGraphDescr(handle, graph);
	nvgraphDestroy(handle);
}

int main(int argc, char * argv[]) {
    // Parse command line
	unsigned int N_INTS = 0;
    
    if (argc == 2) {
		N_INTS = std::stol(std::string(argv[1]));
    }
    else {
        std::cout << "Usage: " << argv[0] << " [block size] [number of threads per block] [array size" << std::endl;
		return 0;
    }

	///////////////////////////////////////////////////////////////////////////
	// Thrust
	thrust::host_vector<int> ones(N_INTS, 1);
	thrust::host_vector<int> twos(N_INTS, 2);
	thrust::host_vector<int> result(N_INTS);

	TIC();
	run_thrust_operations(result, ones, twos);
	std::cout << "Thrust operations took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
	
	///////////////////////////////////////////////////////////////////////////
	// NPP
	std::vector<Npp32f> npp_twos(N_INTS, 1);
	Npp32f npp_result;

	TIC();
	run_npp(&npp_result, npp_twos.data(), npp_twos.size());
	std::cout << "Result is " << npp_result << ". NPP operations took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;

	///////////////////////////////////////////////////////////////////////////
	// nvGRAPH
	TIC();
	run_nvGRAPH();
	std::cout << "nvGRAPH operations took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;

	return 0;
}