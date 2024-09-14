.PHONY: all kernels dataset clean

all: clean kernels dataset

kernels:
	./generate_kernels.sh

dataset:
	python3 generate_npueval.py

clean:
	rm -f kernels/*/kernel.json
