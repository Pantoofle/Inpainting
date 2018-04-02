INCLUDES=-I ../CImg/
LD_LIBS =-lm -lpthread -lX11
CXX_FLAGS= -g -Wall -Wpedantic

TARGET=inpaint.out

CXX=clang++


all: $(TARGET)

$(TARGET): exemplar_based.o main.o
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(LD_LIBS) $^ -o $(TARGET)
	
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $< -c -o $@

run: $(TARGET)
	./$(TARGET) ../img/img1.jpg ../img/msk1.jpg ../img/res1.jpg
	feh ../img/res1.jpg


.PHONY: clean
clean:
	rm -f *.o
	rm -f $(TARGET)
