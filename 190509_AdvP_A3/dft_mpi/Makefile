CXX=mpic++
CFLAGS=-g -Wall

SRC=$(wildcard *.cc)
HDR=$(wildcard *.h)
OBJ=$(SRC:.cc=.o)
EXE=dft_mpi

.PHONY: default clean

default: $(EXE)

%.o: %.cc $(HDR)
	$(CXX) $(CFLAGS) -o $@ -c $<

$(EXE): $(OBJ)
	$(CXX) $(CFLAGS) -o $@ $(OBJ)

clean:
	rm -f $(EXE) $(OBJ)
