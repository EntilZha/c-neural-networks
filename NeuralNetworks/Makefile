CC = g++-4.8

TARGETS=multilayer

all: $(TARGETS)

neural: main.cpp
	 g++-4.8 -g -Ofast -o neural main.cpp -fopenmp -lblas -lcblas

multilayer: multilayer.cpp
	 g++-4.8 -g -Ofast -o multilayer multilayer.cpp -fopenmp -lblas -lcblas

clean:
	rm -f *.o $(TARGETS)
