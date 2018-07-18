
uname := $(shell uname)
ifeq ($(uname),Linux)
  CC := g++
else
  CC := clang++
endif

all: clean reactive

reactive:
	$(CC) -I/usr/include/boost -g -O3 -Wextra --std=c++11 reactive_component.cpp -o reactive_component.o -pthread

simulation:
	$(CC) -I/usr/include/boost -g -O3 -Wextra --std=c++11 simulation.cpp -o simulation.o -lboost_system -lboost_program_options -lcnpy -lz

clean:
	rm -rf simulation.o*
	rm -rf reactive_component.o*
