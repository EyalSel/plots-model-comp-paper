
uname := $(shell uname)
ifeq ($(uname),Linux)
  CC := g++
else
  CC := clang++
endif

all:
	$(CC) -I/usr/include/boost -g -O3 -Wextra --std=c++11 simulation.cpp -o simulation.o -lboost_system -lboost_program_options -lcnpy -lz

clean:
	rm -rf simulation.o*
