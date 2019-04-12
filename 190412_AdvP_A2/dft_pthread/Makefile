CXX=g++
CFLAGS=-g -Wall

SRC=$(wildcard *.cc)
HDR=$(wildcard *.h)
OBJ=$(SRC:.cc=.o)
EXE=dft_pthread

.PHONY: default clean

default: $(EXE)

%.o: %.cc $(HDR)
	$(CXX) $(CFLAGS) -o $@ -c $<

$(EXE): $(OBJ)
	$(CXX) $(CFLAGS) -o $@ $(OBJ) -pthread

clean:
	rm -f $(EXE) $(OBJ)
