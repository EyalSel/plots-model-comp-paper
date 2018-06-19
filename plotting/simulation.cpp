#include <stdio.h>
#include <string>
#include <queue>

typedef unsigned long time;

class Query
{
private:
  const time source_time;
  const time sink_time;
public:
  Query(time src): source_time(src) {};
  ~Query(){};
  void sink(time snk){
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

int main(int argc, char const *argv[])
{
  printf("Hello\n");
  return 0;
}