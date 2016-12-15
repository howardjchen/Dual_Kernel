CXX = nvcc
TARGET = cnnConvLayer

all: cnnConvLayer.cu
	$(CXX) $< -o $(TARGET)

.PHONY: clean run

clean:
	rm -f $(TARGET) 

run:
	./$(TARGET)
