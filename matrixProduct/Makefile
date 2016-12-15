CXX = nvcc
TARGET = mm

all: second_cuda.cu
        $(CXX) $< -o $(TARGET)

.PHONY: clean run

clean:
        rm -f $(TARGET)

run:
        ./$(TARGET)
~                   