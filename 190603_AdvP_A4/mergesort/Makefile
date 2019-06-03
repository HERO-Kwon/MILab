CXX=nvcc
CFLAG=-g --compiler-options="-Wall"
INCLUDE=-I/usr/include -I/usr/local/cuda/include
LIB=-lglut -lGL -lGLU

SRC=$(wildcard *.cu)
HDR=$(wildcard *.h)
OBJ=$(SRC:.cu=.o)
EXE=mergesort

.PHONY: default clean

default: $(EXE)

%.o: %.cu $(HDR)
	$(CXX) $(CFLAG) $(INCLUDE) -dc -o $@ -c $<

$(EXE): $(OBJ)
	$(CXX) $(CFLAG) $(INCLUDE) -o $@ $^ $(LIB)

clean:
	rm -f $(EXE) $(OBJ)
