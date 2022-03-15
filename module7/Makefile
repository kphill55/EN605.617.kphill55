all:
	nvcc assignment.cu -o assignment.exe -std=c++11

clean:
	rm assignment.exe

test:
	./assignment.exe 256 256 1024
	./assignment.exe 512 512 1024
