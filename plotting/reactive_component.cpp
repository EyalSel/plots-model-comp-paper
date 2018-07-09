#include <vector>
#include <map>
#include <thread>
#include "ctpl.h"
#include <future>
#include <iostream>
#include <fstream>

#define DEBUG 1

#ifdef DEBUG
#define D(x) x
#else 
#define D(x)
#endif

// An assert macro that prints the values of both sides when assertion fails
// https://stackoverflow.com/questions/2193544
#define ASSERT(left,operator,right) { if(!((left) operator (right))){ std::cerr << "ASSERT FAILED: " << #left << #operator << #right << " @ " << __FILE__ << " (" << __LINE__ << "). " << #left << "=" << (left) << "; " << #right << "=" << (right) << std::endl; } }

using namespace std;

typedef long time_unit;

/*
  This class keeps track of arrival behavior.
  Call add_timestamps to add more interarrival times to it. 
    Return code 0 means the arrival curve did not change.
    Return code 1 means that the arrival curve "increased" (i.e. represents a higher-throughput/burstier load)
    Return code -1 means that the arrival curve "decreased" (i.e. represents a lower-throughput/more-constant load)
  While "increasing" the arrival curve on unexpected throughput/burstiness increases makes sense, it is not clear how one should "decrease" the arrival curve
  As a simple heuristic, a sliding window of a given size is used. 
*/
class ArrivalTracker
{
private:
  // The entire history of interarrival times recorded.
  vector<time_unit> all_timestamps;
  ctpl::thread_pool pool;

  // x-values of arrival curve coordinates need to spread out logarithmically
  // A sorted mapping from x-values to a pair that consists of:
  // 1. The corresponding y-value on the arrival curve
  // 2. The start index of the window span with the highest value
  // 3. The start index of the window 
  map<time_unit, tuple<int, int> > queries_frequency;

  // function that takes a time_unit x-value, and a time_unit* array of interarrival times.
  // Returns the maximum number of queries that showed up in that interval
  // Notice that this function can be parallelized by cutting the timestamps in half, running this same function on both
  // halves separately, and then combining them by also finding the max of the intersection area
  static pair<int,int> arrival_curve_y_value(int id, time_unit x_value, time_unit* timestamps, int num_timestamps) {
    D(printf("Thread %d called arrival_curve_y_value for x_value %lu\n", id, x_value);)
    if (num_timestamps == 0) {
      D(printf("arrival_curve_y_value received a call with 0 timestamps!\n");)
      return 0;
    }
    int tail_head = 0; // 0 means tail, 1 means head
    int head_index = 0; // the timestamp of index head_index is less than or equal to the sliding time-range's higher end
    int tail_index = 0; // the timestamp of index head_index is less than or equal to the sliding time-range's lower end
    int contained_currently = 1; // at least a single timestamp must be contained in the sliding time-range
    while (head_index < num_timestamps) {
      if (timestamps[tail_index] + x_value > timestamps[head_index]) {
        head_index++;
      } else {
        break;
      }
    }
    if (head_index == num_timestamps) {
      // x_value is bigger than the total span of the timestamps array
      D(printf("Range of timestamps is smaller than the x_value!\n");)
      return num_timestamps;
    } else {
      // timestamps[head_index] - timestamps[tail_index] > x_value, so we take the head one index back to maintain invariant
      head_index--;
    }
    contained_currently = head_index - tail_index + 1;
    int index_of_biggest_window = 0;
    int most_seen_so_far = contained_currently;
    D(printf("Before while loop\n");)
    while (head_index < num_timestamps - 1) {
      time_unit distance_to_next_tail = 0;
      time_unit distance_to_next_head = 0;
      if (tail_head == 0) {
        time_unit head_time_position = timestamps[tail_index] + x_value;
        distance_to_next_tail = timestamps[tail_index+1] - timestamps[tail_index];
        distance_to_next_head = timestamps[head_index+1] - head_time_position;
      } else if (tail_head == 1) {
        time_unit tail_time_position = timestamps[head_index] - x_value;
        distance_to_next_tail = timestamps[tail_index+1] - tail_time_position;
        distance_to_next_head = timestamps[head_index+1] - timestamps[head_index];
      }
      if (distance_to_next_tail > distance_to_next_head) {
        tail_head = 1;
        head_index++;
        contained_currently++;
      } else if (distance_to_next_tail <= distance_to_next_head) {
        tail_head = 0;
        tail_index++;
        contained_currently--;
      }
      D(ASSERT(contained_currently, >=, head_index - tail_index))
      if (contained_currently > most_seen_so_far) {
        most_seen_so_far = contained_currently;
        index_of_biggest_window = tail_index+1;
      }
    }
    return make_pair(most_seen_so_far,index_of_biggest_window);
  } 

  // A thread-parallel version of the arrival_curve_y_value function
  vector<pair<time_unit,int> > arrival_curve_y_values(time_unit* x_values, int num_x_values, time_unit* timestamps, int num_timestamps) {
    D(printf("arrival_curve_y_values called with %d x_values\n", num_x_values);)
    vector<pair<time_unit, future<int>> > futures_vector;
    for (int i = 0; i < num_x_values; ++i) {
      // create threadpool, push arrival_curve_y_value with different x_value arguments, get futures back and place in vector
      D(printf("Pushing calculation for x_value %lu\n", x_values[i]);)
      future<int> future_result = pool.push(arrival_curve_y_value, x_values[i], timestamps, num_timestamps);
      futures_vector.push_back(make_pair(x_values[i], move(future_result)));
    }
    // loop through vector, call get to get the value of the future, return vector of x_values to y_values
    vector<pair<time_unit, int> > result;
    for (unsigned int i = 0; i < futures_vector.size(); ++i) {
      result.push_back(make_pair(futures_vector[i].first, futures_vector[i].second.get()));
    }
    return result;
  }

public:
  const time_unit window_size;
  const int number_of_parallel_threads;
  ArrivalTracker(time_unit window_size, int number_of_parallel_threads):
  window_size(window_size), number_of_parallel_threads(number_of_parallel_threads), pool(number_of_parallel_threads){
    // pick x values for the arrival curve
  };
  vector<pair<time_unit,int> > add_timestamps2(time_unit* timestamps, int num_timestamps) {
    // append to all_timestamps
    // figure out the maximum count (the y-value for each x-value) of the added piece = F
    // figure out the maximum count of the forefeited piece from the begining of the window = A
    // If F = max, need to rerun calculation on the entire new window span
    // if F < max and A > max, the arrival curve increased
    // if F < max and A <= max then nothing changed
    vector<time_unit> x_values;
    x_values.push_back(1000);
    x_values.push_back(4000);
    x_values.push_back(16000);
    x_values.push_back(64000);
    x_values.push_back(256000);
    x_values.push_back(1024000);
    return arrival_curve_y_values(x_values.data(), x_values.size(), timestamps, num_timestamps);
  }

  int add_timestamps(time_unit* timestamps, int num_timestamps) {
    // append to all_timestamps
    all_timestamps.push_back(timestamps, timestamps+num_timestamps);
    // figure out section that's forfeited
    int tail_index = 0;
    while (tail_index < all_timestamps.size()) {
      if (all_timestamps[tail_index] + window_size < all_timestamps.end()[-1]) {
        tail_index++;
      } else {
        break;
      }
    }
    if (tail_index == 0) {

    }

    // figure out the maximum count (the y-value for each x-value) of the added piece = F
    // figure out the maximum count of the forefeited piece from the begining of the window = A
    // If F = max, need to rerun calculation on the entire new window span
    // if F < max and A > max, the arrival curve increased
    // if F < max and A <= max then nothing changed
    return -2;
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

#include <chrono>

using namespace chrono;

int main(int argc, char const *argv[])
{
  int ms = duration_cast< microseconds >(system_clock::now().time_since_epoch()).count();
  printf("%d\n", ms);
  pair<float*, int> deltas = read_deltas_file("/data/ges/plots-model-comp-paper/experiments/cached_arrival_processes/241_0.1.deltas");
  vector<long> long_deltas (deltas.second);
  float sum = 0;
  for (int i = 0; i < deltas.second; ++i) {
    sum+=deltas.first[i];
    long_deltas[i] = long(sum*1000);
  }
  ArrivalTracker tracker (0, 4);
  auto result = tracker.add_timestamps2(long_deltas.data(), long_deltas.size());
  for(auto v:result) {
    printf("%lu -> %d\n", v.first, v.second);
  }
}
