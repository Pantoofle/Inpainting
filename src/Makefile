INCLUDES=-I ../CImg/
LD_LIBS =-lm -lpthread -lX11
CXX_FLAGS= -g -O2 -Wall -Wpedantic -fopenmp

TARGET=inpaint.out
SIZE?=11

CXX=g++


all: $(TARGET)

$(TARGET): exemplar_based.o main.o
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(LD_LIBS) $^ -o $(TARGET)
	
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $< -c -o $@

run: $(TARGET)
	./$(TARGET) ../img/input.jpg ../img/msk.jpg ../img/res.jpg $(SIZE)
	feh ../img/res.jpg

vg: $(TARGET)
	valgrind ./$(TARGET) ../img/input.jpg ../img/msk.jpg ../img/res.jpg $(SIZE)

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(TARGET)
