all:
	nvcc assignment.cc -o assignment.exe -lOpenCL -std=c++11

test:
	./assignment.exe zeros.b64
	./assignment.exe declaration.b64

clean:
	rm assignment.exe
