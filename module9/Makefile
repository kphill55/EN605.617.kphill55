all:
	nvcc assignment.cu -o assignment.exe -lnpps -lnvgraph -std=c++11

clean:
	rm assignment.exe

test:
	./assignment.exe 32
	./assignment.exe 64
