CXX = nvcc
TARGET = ip
FLAGS = -lineinfo

all: ip.cu
	$(CXX) $< -o $(TARGET) $(FLAGS)

.PHONY: clean run

clean:
	rm -f $(TARGET) 

run:
	./$(TARGET)
