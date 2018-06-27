#include <iostream>
#include <string>
#include <queue>
#include <fstream>
#include <assert.h>
#include <vector>
#include <unordered_map>
#include <functional>
#include <boost/program_options.hpp>
#include "cnpy.h"

using namespace std;

typedef unsigned long clock_time;

class Query {
private:
  clock_time source_time;
  clock_time sink_time;
public:
  const int id;
  Query(int id, clock_time src): id(id) {
    source_time = src;
  };
  ~Query(){};
  void sink(clock_time snk){
    sink_time = snk;
  }
};

// Forward declaration. The class can be found below the Scheduler class below.
class Node;

typedef tuple<Node*, clock_time, int> event;

class Scheduler {
private:
  clock_time time_now;
  // returns true if a is less than b. Because the priority queue
  // takes items out by larger first, returning true means that
  // a should be taken AFTER b
  static bool compare(event a, event b) {
    return get<1>(a) > get<1>(b);
  }
  priority_queue < event, vector<event> , function<bool(event,event)> > scheduled;
public:
  Scheduler(): scheduled(compare) {
    time_now = 0;
  };

  void schedule(Node* node, clock_time in_how_long, int identifier = 0) {
    // event is not a function, it's a typedef of a tuple (see above)
    event new_event (node, time_now + in_how_long, identifier);
    scheduled.push(new_event);
  }

  // implemented after Node declaration in order to compile
  void run();
};

class Node {
protected:
  Scheduler* scheduler;
private:
  vector<Node*> children;
  vector<Node*> parents;
  int num_parents;
  int num_children;
public:
  const string name;
  Node(string name, Scheduler* scheduler): name(name), scheduler(scheduler) {
    printf("Node %s constructor\n", name.c_str());
  };

  void then(Node* node) {
    children.push_back(node);
    node -> parents.push_back(node);
    node -> num_parents++;
    num_children++;
  }

  // The arrival method guarantees read-only access to the queries array
  virtual void arrival(Query** queries, int num_queries, clock_time time_now);

  void send(Query** queries, int num_queries, clock_time time_now) {
    for(vector<Node*>::iterator it = children.begin(); it != children.end(); ++it) {
      (*it) -> arrival(queries, num_queries, time_now);
    }
  }

  // To be called by the scheduler
  virtual void to_schedule(clock_time time_now, int identifier);

  int get_num_parents(){
    return num_parents;
  }

  int get_num_children(){
    return num_children;
  }
};

void Scheduler::run() {
  while (!scheduled.empty()) {
    event next_event = scheduled.top();
    scheduled.pop();
    get<0>(next_event) -> to_schedule(get<1>(next_event), get<2>(next_event));
  }
}



class SourceNode : public Node
{
private:
  pair<float*, int> deltas;
  int next_query_index;
  vector<Query> query_array;
public:
  SourceNode(pair<float*, int> deltas, Scheduler* scheduler): 
    Node("source", scheduler), deltas(deltas) {
    next_query_index = 0;
    query_array.reserve(deltas.second);
    printf("SourceNode %s constructor\n", name.c_str());
  };
  ~SourceNode() {
    delete deltas.first;
  };

  void to_schedule(clock_time time_now, int /* unused identifier */) {
    send_next(time_now);
  }

  void send_next(clock_time time_now) {
    if (next_query_index == deltas.second) {
      printf("Source node finished sending all the queries!\n");
    }
    query_array.push_back(Query(time_now, next_query_index));
    Query* pointer = (query_array.data() + next_query_index);
    send(&pointer, 1, time_now);
    next_query_index++;
    scheduler -> schedule(this, deltas.first[next_query_index-1]);
  }
};

class QueuedNode : public Node
{
protected:
  std::queue<Query*> arrival_queue;

public:
  
  QueuedNode(string name, Scheduler* scheduler): Node(name, scheduler){
    printf("QueuedNode %s constructor\n", name.c_str());
  };

  void arrival(Query** queries, int num_queries, clock_time /* unused time_now */) {
    for(int i = 0; i < num_queries; i++){
      arrival_queue.push(queries[i]);
    }
  }
  
};

class BatchedNode:public QueuedNode
{
private:
  // A vector converted from a (flattened) numpy array of shape N * 3, 
  // where each entry represents profiler results for the model.
  // the first column of the array represents the batch size
  // the second column of the array represents the p99 latency in ms
  // the third column of the array represents the throughput in qps
  vector<float> batchsize_p99lat_thru;
  // A vector of just the batchsizes, initialized in extract_batchsizes()
  // called in the constructor
  vector<unsigned int> batchsizes;

  void extract_batchsizes() {
    for (unsigned int i = 0; i < batchsize_p99lat_thru.size(); i+=3) {
      int value_from_array = (int)batchsize_p99lat_thru[i];
      assert (value_from_array > 0);
      batchsizes.push_back((unsigned int)value_from_array);
    }
  }

  // Given entry (the row in the original numpy array) return delay
  // calculated from throughput (so mean, not p99)
  float get_delay_for_entry(int entry_id) {
    return 1/batchsize_p99lat_thru[entry_id*3+2]*1000;
  }

  // Given a batchsize, returns the predicted latency by connecting a line
  // between the closes profiled batchsizes above and below the given batchsize   
  float latency_for_batchsize(int batchsize) {
    int num_batchsize_entries = batchsizes.size();
    int lowest_batchsize = batchsizes[0];
    int highest_batchsize = batchsizes.end()[-1];
    assert (batchsize <= highest_batchsize && batchsize >= lowest_batchsize);
    unsigned int index = 0;
    for (; index < batchsizes.size(); ++index) {
      int batchsize_entry = batchsizes[index];
      if (batchsize == batchsize_entry) {
        return get_delay_for_entry(index);
      }
      if (batchsize_entry > batchsize) {
        break;
      }
    }
    int batchsize_below = batchsizes[index-1];
    int batchsize_above = batchsizes[index];
    float batchsize_below_latency = get_delay_for_entry(index-1);
    float batchsize_above_latency = get_delay_for_entry(index);
    float rise = batchsize_above_latency - batchsize_below_latency;
    float run  = batchsize_above - batchsize_below;
    float slope = rise/run;
    float delta = batchsize - batchsize_below;
    return batchsize_below_latency + delta * slope;
  }

  // a vector of vectors, each represents a queue for a replica.
  vector< vector<Query*> > replica_queues;

  // Replica takes from queue as much as it can. Must be called when there's something in queue
  // and replica is not processing something already
  void replica_take(int replica_index) {
    assert (!arrival_queue.empty() && replica_queues[replica_index].empty());
    int num_to_dequeue = min<int>(max_batchsize, arrival_queue.size());
    for (int j = 0; j < num_to_dequeue; j++) {
      Query* next_query = arrival_queue.front(); 
      replica_queues[replica_index].push_back(next_query);
      arrival_queue.pop();
    }
    scheduler -> schedule(this, latency_for_batchsize(num_to_dequeue), replica_index);
  }
public:
  const int max_batchsize;
  const int num_replicas;
  
  BatchedNode(string name, int max_batchsize, vector<float>batchsize_p99lat_thru, int num_replicas, Scheduler* scheduler)
    :QueuedNode(name,scheduler),max_batchsize(max_batchsize),num_replicas(num_replicas),batchsize_p99lat_thru(batchsize_p99lat_thru){
      extract_batchsizes();
      for (int i = 0; i < num_replicas; ++i) {
        replica_queues.push_back(vector<Query*>());
      }
  }

  void arrival(Query** queries, int num_queries, clock_time time_now) {
    Node::arrival(queries, num_queries, time_now);
    assert (!arrival_queue.empty());
    for(int i = 0; i < num_replicas; i++) {
      if (replica_queues[i].empty()) {
        replica_take(i);
        if (arrival_queue.empty()) {
          break;
        }
      }
    }
  }

  void finish_processing(int replica_index, clock_time time_now) {
    vector<Query*> replica_queue = replica_queues[replica_index];
    assert (replica_queue.size() > 0);
    // Create an array pointer to the vector's contents
    Query** array_pointer = &replica_queue[0];
    send(array_pointer, replica_queue.size(), time_now);
    replica_queue.clear();
    if (!arrival_queue.empty()) {
      replica_take(replica_index);
    }
  }

  void to_schedule(clock_time time_now, int identifier) {
    finish_processing(identifier, time_now);
  }
};

class JoinNode:Node
{
private:
    // A mapping from a query pointer to the number of times that query
    // has been received. When that number equals the number of parents
    // that this node has, that query can be sent on to the children
    unordered_map<Query*, int> queries_frequency;
public:
  JoinNode(string name, Scheduler* scheduler):Node(name, scheduler) {}

  void arrival(Query** queries, int num_queries, clock_time time_now) {
    for (int i = 0; i < num_queries; i++) {
      auto search_result = queries_frequency.find(queries[i]);
      if(search_result == queries_frequency.end()) { // not found
        queries_frequency.insert(make_pair(queries[i], 1));
      } else { // found
          if (search_result -> second < get_num_children()-1) {
            queries_frequency[queries[i]] = search_result -> second + 1;
          } else {
            queries_frequency.erase(queries[i]);
            send(&queries[i], 1, time_now);
          }
      }
    }
  }
};

class SinkNode:public Node
{
public:
  SinkNode(Scheduler* scheduler): Node("sink", scheduler){}

  virtual void arrival(Query** queries, int num_queries, clock_time time_now) {
    for (int i = 0; i < num_queries; ++i)
    {
      queries[i] -> sink(time_now);
    }
  }
};

pair<float*, int> read_deltas_file(string file_name){
  // first count how many lines in file
  ifstream myfile(file_name);
  if(!myfile) {
    printf("Error opening output file\n");
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

float mean(pair<float*, int> array) {
  float sum = 0;
  for(int i = 0; i < array.second; i++) {
    sum+=array.first[i];
  }
  return sum / array.second;
}

namespace po = boost::program_options;

int main(int argc, char const *argv[])
{
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 

  string deltas_file = "default";
  int irp = 0;
  int ibs = 0;
  string ibe = "default";
  int rrp = 0;
  int rbs = 0;
  string rbe = "default";
  int lrp = 0;
  int lbs = 0;
  string lbe = "default";
  int krp = 0;
  int kbs = 0;
  string kbe = "default";
  po::options_description desc("C++ simulation");
  desc.add_options()
      ("help", "produce help message")
      ("deltas", po::value<string>(&deltas_file)->required(), "Deltas file with ms difference between queries")
      ("irp", po::value<int>(&irp)->required(), "Inception replication factor")
      ("ibs", po::value<int>(&ibs)->required(), "Inception batchsize")
      ("ibe", po::value<string>(&ibe)->required(), "Inception behavior file")
      ("rrp", po::value<int>(&rrp)->required(), "ResNet replication factor")
      ("rbs", po::value<int>(&rbs)->required(), "ResNet batchsize")
      ("rbe", po::value<string>(&rbe)->required(), "ResNet behavior file")
      ("lrp", po::value<int>(&lrp)->required(), "LogReg replication factor")
      ("lbs", po::value<int>(&lbs)->required(), "LogReg batchsize")
      ("lbe", po::value<string>(&lbe)->required(), "LogReg behavior file")
      ("krp", po::value<int>(&krp)->required(), "Inception replication factor")
      ("kbs", po::value<int>(&kbs)->required(), "Inception batchsize")
      ("kbe", po::value<string>(&kbe)->required(), "Inception behavior file")
  ;
  try { 
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm); // can throw 

    if (vm.count("help")) {
      std::cout << desc << std::endl; 
      return SUCCESS; 
    } 

    po::notify(vm); // throws on error, so do after help in case 
                    // there are any problems 

    printf("%s\n", deltas_file.c_str());
  } catch(po::error& e) { 
    std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
    std::cerr << desc << std::endl; 
    return ERROR_IN_COMMAND_LINE; 
  }

  pair<float*, int> result = read_deltas_file(deltas_file);
  if (result.first != nullptr) {
    printf("Mean: %f\n", mean(result));
  }

  cnpy::NpyArray inception_behavior = cnpy::npy_load(ibe);
  cnpy::NpyArray resnet_behavior = cnpy::npy_load(rbe);
  cnpy::NpyArray logreg_behavior = cnpy::npy_load(lbe);
  cnpy::NpyArray ksvm_behavior = cnpy::npy_load(kbe);

  Scheduler scheduler();
  SourceNode source (result, &scheduler);
  BatchedNode inception ("Inception", ibs, inception_behavior, irp, &scheduler);
  BatchedNode resnet ("ResNet", rbs, resnet_behavior, rrp, &scheduler);
  BatchedNode logreg ("LogReg", lbs, logreg_behavior, lrp, &scheduler);
  BatchedNode ksvm ("KSVM", kbs, ksvm_behavior, krp, &scheduler);
  JoinNode join ("Join", &scheduler);
  SinkNode sink (&scheduler);
  source.then(inception);
  source.then(resnet);
  inception.then(logreg);
  resnet.then(ksvm);
  logreg.then(join);
  ksvm.then(join);
  join.then(sink);

  return 0;
}
