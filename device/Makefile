CXX = nvcc
TARGET = device

all: device.cu
	$(CXX) $< -o $(TARGET)

.PHONY: clean run

clean:
	rm -f $(TARGET) 

run:
	./$(TARGET)
