#include <stdio.h>
#include <string>
#include <queue>
#include <fstream>

typedef unsigned long clock_time;

class Query
{
private:
  clock_time source_time;
  clock_time sink_time;
public:
  Query(clock_time src): source_time(src) {};
  ~Query(){};
  void sink(clock_time snk){
    sink_time = snk;
  }
};

class Node
{
public:
  const std::string name;
  std::queue<Query> arrival_queue;
  Node(std::string name): name(name), arrival_queue(std::queue<Query>()) {};
  ~Node(){};
  
};

float* read_deltas_file(std::string file_name){
  // first count how many lines in file
  std::ifstream(file_name);
  if(!myfile) {
    printf("Error opening output file\n");
    std::system("exit");
    return nullptr;
  }
  std::string line;
  int number_of_lines = 0;
  while (std::getline(myfile, line)) {
    ++number_of_lines;
  }
  printf("Number of lines is %d\n", number_of_lines);
  // then allocate array
  float* result = new float[number_of_lines];
  // then read lines into array and convert to float
}

int main(int argc, char const *argv[])
{
  printf("argc = %d\n", argc);
  read_deltas_file(argv[0]);
  printf("Hello\n");
  Query q = Query(2);
  return 0;
}