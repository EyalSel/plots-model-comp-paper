#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <assert.h>
#include <vector>
#include <unordered_map>
#include <functional>
#include <boost/program_options.hpp>
#include <exception>
#include <algorithm>
#include "cnpy.h"


// #define DEBUG 1

#ifdef DEBUG
#define D(x) x
#else 
#define D(x)
#endif

using namespace std;

// An assert macro that prints the values of both sides when assertion fails
// https://stackoverflow.com/questions/2193544
#define ASSERT(left,operator,right) { if(!((left) operator (right))){ std::cerr << "ASSERT FAILED: " << #left << #operator << #right << " @ " << __FILE__ << " (" << __LINE__ << "). " << #left << "=" << (left) << "; " << #right << "=" << (right) << std::endl; } }


typedef float clock_time;

class Query {
private:
  clock_time source_time;
  clock_time sink_time;
  bool sink_determined = false;
public:
  const int id;
  Query(int id, clock_time src): id(id) {
    source_time = src;
  };
  ~Query(){};
  void sink(clock_time snk){
    D(printf("%f:\tSink called for %d\n", snk, id);)
    sink_time = snk;
    sink_determined = true;
  }
  float end_to_end_time() {
    if(!sink_determined) {
      printf("Query id %d\n", id);
      throw logic_error("Tried to get end to end time before sink was determined");
    }
    return sink_time - source_time;
  }

};

class Node {
protected:
  // The arrival method guarantees read-only access to the queries array
  virtual void arrival(Query** queries, int num_queries, clock_time time_now) = 0;

  void send(Query** queries, int num_queries, clock_time time_now) {
    for(vector<Node*>::iterator it = children.begin(); it != children.end(); ++it) {
      (*it) -> arrival(queries, num_queries, time_now);
    }
  }
private:
  vector<Node*> children;
  vector<Node*> parents;
  int num_parents;
  int num_children;
public:
  const string name;
  Node(string name): name(name) {
    printf("Node %s constructor\n", name.c_str());
  };

  void then(Node* node) {
    children.push_back(node);
    node -> parents.push_back(node);
    node -> num_parents++;
    num_children++;
  }

  int get_num_parents(){
    return num_parents;
  }

  int get_num_children(){
    return num_children;
  }

};

// Forward declaration of Scheduler
class Scheduler;

class SchedulableNode : virtual public Node
{
protected: 
  Scheduler* scheduler;
public:
  SchedulableNode(string name, Scheduler* scheduler): Node(name), scheduler(scheduler) {
    printf("SchedulableNode %s constructor\n", name.c_str());
  }
  // To be called by the scheduler
  virtual void to_schedule(clock_time time_now, int identifier) = 0;
};

typedef tuple<SchedulableNode*, clock_time, int> event;

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

  void schedule(SchedulableNode* node, clock_time in_how_long, int identifier = 0) {
    // event is not a function, it's a typedef of a tuple (see above)
    event new_event (node, time_now + in_how_long, identifier);
    scheduled.push(new_event);
  }

  // implemented after Node declaration in order to compile
  void run() {
    while (!scheduled.empty()) {
      event next_event = scheduled.top();
      time_now = get<1>(next_event);
      scheduled.pop();
      get<0>(next_event) -> to_schedule(get<1>(next_event), get<2>(next_event));
    }
  }
};

class QueuedNode : virtual public Node
{
protected:
  std::queue<Query*> arrival_queue;
  
  void arrival(Query** queries, int num_queries, clock_time /* unused time_now */) {
    for(int i = 0; i < num_queries; i++){
      arrival_queue.push(queries[i]);
    }
  }
public:
  QueuedNode(string name): Node(name){
    printf("QueuedNode %s constructor\n", name.c_str());
  }; 
};


class SourceNode : public SchedulableNode
{
private:
  pair<float*, int> deltas;
  int next_query_index;
  vector<Query> query_array;

  void send_next(clock_time time_now) {
    if (next_query_index == deltas.second) {
      printf("Source node finished sending all the queries!\n");
      return;
    }
    D(printf("%f:\tSending %d\n", time_now, next_query_index);)
    query_array.push_back(Query(next_query_index, time_now));
    Query* pointer = (query_array.data() + next_query_index);
    send(&pointer, 1, time_now);
    scheduler -> schedule(this, deltas.first[next_query_index]);
    next_query_index++;
  }

public:
  SourceNode(pair<float*, int> deltas, Scheduler* scheduler): 
    SchedulableNode("source", scheduler), Node("source"), deltas(deltas) {
    next_query_index = 0;
    query_array.reserve(deltas.second);
    printf("SourceNode %s constructor with %d deltas\n", name.c_str(), deltas.second);
    scheduler -> schedule(this, 0);
  };
  ~SourceNode() {
    delete deltas.first;
  };

  void to_schedule (clock_time time_now, int /* unused identifier */) override {
    send_next(time_now);
  }

  void arrival(Query** /*queries*/, int /*num_queries*/, clock_time /*time_now*/) override {
    throw logic_error("Arrival function called for source node");
  }

  // To be called only after scheduler completes
  pair<float*, int> end_to_end_times() {
    float* result = new float[query_array.size()];
    for (unsigned int i = 0; i < query_array.size(); ++i) {
      result[i] = query_array[i].end_to_end_time();
    }
    return pair<float*, int>(result, query_array.size());
  }
  
};

class BatchedNode : public QueuedNode, public SchedulableNode
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
      ASSERT (value_from_array, >, 0);
      batchsizes.push_back((unsigned int)value_from_array);
    }
  }

  void print_behavior() {
    printf("%s model behavior\n", name.c_str());
    for (unsigned int i = 0; i < batchsize_p99lat_thru.size(); i+=3) {
      int batchsize = (int) batchsize_p99lat_thru[i];
      float p99_lat = batchsize_p99lat_thru[i+1];
      float thru = batchsize_p99lat_thru[i+2];
      printf("%d\t%f\t%f\n", batchsize, p99_lat, thru);
    }
    printf("\n");
  }

  // Given entry (the row in the original numpy array) return delay
  // calculated from throughput (so mean, not p99)
  float get_delay_for_entry(int entry_id) {
    return 1/batchsize_p99lat_thru[entry_id*3+2]*1000*batchsize_p99lat_thru[entry_id*3];
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
  // time_now argument is wrapped with the debugging symbol because it's only used during debugging
  void replica_take(int replica_index, clock_time D(time_now)) {
    assert (!arrival_queue.empty());
    assert (replica_queues[replica_index].empty());
    int num_to_dequeue = min<int>(max_batchsize, arrival_queue.size());
    for (int j = 0; j < num_to_dequeue; j++) {
      Query* next_query = arrival_queue.front(); 
      replica_queues[replica_index].push_back(next_query);
      D(printf("%f:\t%s replica#%d take %d\n", time_now, name.c_str(), replica_index, next_query->id);)
      arrival_queue.pop();
    }
    D(printf("%f:\t%s replica#%d a batch of %d\n", time_now, name.c_str(), replica_index, num_to_dequeue);)
    scheduler -> schedule(this, latency_for_batchsize(num_to_dequeue), replica_index);
  }
  void finish_processing(int replica_index, clock_time time_now) {
    D(printf("%f:\t%s replica#%d finished processing\n", time_now, name.c_str(), replica_index);)
    assert (replica_queues[replica_index].size() > 0);
    // Create an array pointer to the vector's contents
    Query** array_pointer = &replica_queues[replica_index][0];
    send(array_pointer, replica_queues[replica_index].size(), time_now);
    replica_queues[replica_index].clear();
    if (!arrival_queue.empty()) {
      replica_take(replica_index, time_now);
    }
  }
protected:
  void arrival(Query** queries, int num_queries, clock_time time_now) override {
    QueuedNode::arrival(queries, num_queries, time_now);
    D(printf("%f:\t%s queue size: %lu\n", time_now, name.c_str(), arrival_queue.size()));
    assert (!arrival_queue.empty());
    for(int i = 0; i < num_replicas; i++) {
      if (replica_queues[i].empty()) {
        replica_take(i, time_now);
        if (arrival_queue.empty()) {
          break;
        }
      } else {
        // printf("%s replica index %d has %d items\n", name.c_str(), i, (int)replica_queues[i].size());
      }
    }
  }

public:
  const int max_batchsize;
  const int num_replicas;
  
  BatchedNode(string name, int max_batchsize, pair<float*, int> model_behavior, int num_replicas, Scheduler* scheduler)
    :QueuedNode(name), SchedulableNode(name, scheduler), Node(name), max_batchsize(max_batchsize),num_replicas(num_replicas),batchsize_p99lat_thru(model_behavior.first, model_behavior.first+model_behavior.second){
      extract_batchsizes();
      print_behavior();
      for (int i = 0; i < num_replicas; ++i) {
        replica_queues.push_back(vector<Query*>());
      }
  }
  void to_schedule(clock_time time_now, int identifier) override {
    finish_processing(identifier, time_now);
  }
};

class JoinNode : public QueuedNode
{
private:
    // A mapping from a query pointer to the number of times that query
    // has been received. When that number equals the number of parents
    // that this node has, that query can be sent on to the children
    unordered_map<Query*, int> queries_frequency;
protected:
  void arrival(Query** queries, int num_queries, clock_time time_now) override {
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
public:
  JoinNode(string name):QueuedNode(name), Node(name) {}
};

class SinkNode : public QueuedNode
{
public:
  SinkNode(): QueuedNode("sink"), Node("sink"){}
protected:
  void arrival(Query** queries, int num_queries, clock_time time_now) override {
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

pair<float*,int> extract_cnpy(cnpy::NpyArray arr) {
  vector<size_t> the_shape = arr.shape;
  assert (arr.shape.size() == 2 && arr.shape[1] == 3);
  int size = arr.shape[0] * arr.shape[1];
  return pair<float*, int>(arr.data<float>(), size);
}

int main(int argc, char const *argv[])
{
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 

  string deltas_file = "default";
  int irf = 0;
  int ibs = 0;
  string ibe = "default";
  int rrf = 0;
  int rbs = 0;
  string rbe = "default";
  int lrf = 0;
  int lbs = 0;
  string lbe = "default";
  int krf = 0;
  int kbs = 0;
  string kbe = "default";
  po::options_description desc("C++ simulation");
  string result_file = "default";
  desc.add_options()
      ("help", "produce help message")
      ("deltas", po::value<string>(&deltas_file)->required(), "Deltas file with ms difference between queries")
      ("irf", po::value<int>(&irf)->required(), "Inception replication factor")
      ("ibs", po::value<int>(&ibs)->required(), "Inception batchsize")
      ("ibe", po::value<string>(&ibe)->required(), "Inception behavior file")
      ("rrf", po::value<int>(&rrf)->required(), "ResNet replication factor")
      ("rbs", po::value<int>(&rbs)->required(), "ResNet batchsize")
      ("rbe", po::value<string>(&rbe)->required(), "ResNet behavior file")
      ("lrf", po::value<int>(&lrf)->required(), "LogReg replication factor")
      ("lbs", po::value<int>(&lbs)->required(), "LogReg batchsize")
      ("lbe", po::value<string>(&lbe)->required(), "LogReg behavior file")
      ("krf", po::value<int>(&krf)->required(), "Inception replication factor")
      ("kbs", po::value<int>(&kbs)->required(), "Inception batchsize")
      ("kbe", po::value<string>(&kbe)->required(), "Inception behavior file")
      ("result_file", po::value<string>(&result_file)->required(), "File name where to print end to end times")
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

  Scheduler scheduler;
  SourceNode source (result, &scheduler);
  BatchedNode inception ("Inception", ibs, extract_cnpy(inception_behavior), irf, &scheduler);
  BatchedNode resnet ("ResNet", rbs, extract_cnpy(resnet_behavior), rrf, &scheduler);
  BatchedNode logreg ("LogReg", lbs, extract_cnpy(logreg_behavior), lrf, &scheduler);
  BatchedNode ksvm ("KSVM", kbs, extract_cnpy(ksvm_behavior), krf, &scheduler);
  JoinNode join ("Join");
  SinkNode sink;
  source.then(&inception);
  source.then(&resnet);
  inception.then(&logreg);
  resnet.then(&ksvm);
  logreg.then(&join);
  ksvm.then(&join);
  join.then(&sink);

  scheduler.run();

  pair<float*, int> end_to_end_times = source.end_to_end_times();
  ofstream myfile;
  myfile.open (result_file, ofstream::out | ofstream::trunc);
  for (int i = 0; i < end_to_end_times.second; ++i) {
    myfile << to_string(end_to_end_times.first[i]);
    myfile << "\n";
  }
  myfile.close();

  return 0;
}
