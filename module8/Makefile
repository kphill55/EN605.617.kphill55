all:
	nvcc assignment.cu -o assignment.exe -lcufft -lcublas -std=c++11

clean:
	rm assignment.exe

test:
	./assignment.exe 32
