#include <iostream>
#include <string>
#include <queue>
#include <fstream>
#include <assert.h>
#include <vector>

using namespace std;

typedef unsigned long clock_time;

class Query
{
private:
  clock_time source_time;
  clock_time sink_time;
public:
  const int id;
  Query(clock_time src, int id): id(id), source_time(src) {};
  ~Query(){};
  void sink(clock_time snk){
    sink_time = snk;
  }
};

class Node
{
private:
  vector<Node*> children;
  vector<Node*> parents;
public:
  const string name;
  
  Node(string name): name(name) {
    printf("Node %s constructor\n", name.c_str());
  };

  void then(Node* node) {
    children.push_back(node);
    node -> parents.push_back(node);
  }

  virtual void arrival(Query* [] queries, int num_queries);

  void send(Query*[] queries, int num_queries) {
    for(vector<Node*>::iterator it = children.begin(); it != children.end(); ++it) {
      (*it) -> arrival(queries, num_queries);
    }
  }

  ~Node(){};
  
};

class QueuedNode : public Node
{
protected:
  std::queue<Query> arrival_queue;

public:
  
  QueuedNode(string name): Node(name), arrival_queue(queue<Query>()){
    printf("QueuedNode %s constructor\n", name.c_str());
  };

  ~QueuedNode(){};

  void arrival(Query* queries, int num_queries) {

  }
  
};

class SourceNode : public Node
{
private:
  pair<float*, int> deltas;
  int next_query_index;
public:
  SourceNode(string name, pair<float*, int> deltas): 
    Node(name), deltas(deltas), next_query_index(0) {
    printf("SourceNode %s constructor\n", name.c_str());
  };
  ~SourceNode();

  float send_next() {
    if (next_query_index == deltas.second) {
      return -1
    }
    send()
    next_query_index++;
    return deltas.first[next_query_index-1];
  }
  
};


pair<float*, int> read_deltas_file(string file_name){
  // first count how many lines in file
  ifstream myfile(file_name);
  if(!myfile) {
    printf("Error opening output file\n");
    system("exit");
    return pair<float*, int>(nullptr, 0);
  }
  string line;
  int number_of_lines = 0;
  while (getline(myfile, line)) {
    ++number_of_lines;
  }
  printf("Number of lines is %d\n", number_of_lines);
  // then allocate array
  float* result = new float[number_of_lines];
  // then read lines into array and convert to float
  myfile.clear();
  myfile.seekg(0, myfile.beg);
  for(int i = 0; i < number_of_lines; i++) {
    assert (getline(myfile, line));
    result[i] = stof(line);
  }
  return pair<float*, int>(result, number_of_lines);
}


int main(int argc, char const *argv[])
{
  printf("argc = %d\n", argc);
  if (argc < 2) {
    return 0;
  }
  pair<float*, int> result = read_deltas_file(argv[1]);
  float sum = 0;
  for(int i = 0; i < result.second; i++) {
    sum+=result.first[i];
  }
  float mean = sum / result.second;
  printf("Mean: %f\n", mean);
  QueuedNode queuedNode("hello");
  return 0;
}