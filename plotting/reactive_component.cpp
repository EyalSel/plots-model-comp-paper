#include <vector>
#include <queue>
#include <map>
#include <thread>
#include "ctpl.h"
#include <future>
#include <iostream>
#include <fstream>
#include <stdexcept>

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
  typedef long time_unit; // in microseconds, expects a chrono time since epoch timestamp

private:
  // The entire history of interarrival times recorded.
  ctpl::thread_pool pool;

  // x-values of arrival curve coordinates need to spread out logarithmically
  // A sorted mapping from x-values to a pair that consists of:
  // 1. The corresponding y-value on the arrival curve
  // 2. The start index of the window span with the highest value
  // 3. The start index of the last window
  typedef tuple<unsigned int, unsigned int, unsigned int> bookkeeping_entry;
  map<time_unit, bookkeeping_entry > queries_frequency;

  map<time_unit, future<bookkeeping_entry> > scheduled;

  // schedules a calculation to run, throws error if the given x_value has already been scheduled but not collected
  void schedule(time_unit x_value, unsigned int start_index, unsigned int end_index) {
    D(printf("Scheduling for x_value %lu from %d to %d\n", x_value, start_index, end_index);)
    if (scheduled.find(x_value) != scheduled.end()) {
      throw invalid_argument("Attempted to schedule an x_value "+to_string(x_value)+" that has already been scheduled!");
    }
    future<bookkeeping_entry> future_result = pool.push(arrival_curve_y_value, x_value, start_index, end_index, all_timestamps);
    scheduled[x_value] = move(future_result);
  }

  // Attempts to collect a result that has been scheduled. No guarantee about which result it is. May or may not block, depending
  // on whether the result picked to be collected has finished or not
  pair<time_unit, bookkeeping_entry> collect() {
    if(scheduled.empty()){
      throw logic_error("Tried to collect scheduled result, but nothing has been scheduled");
    }
    auto chosen_to_collect = scheduled.begin();
    time_unit x_value = chosen_to_collect->first;
    bookkeeping_entry retrieved_value = chosen_to_collect->second.get();
    scheduled.erase(x_value);
    return make_pair(x_value, retrieved_value);
  }

public:
  deque<time_unit> all_timestamps;
  const time_unit window_size;
  const int number_of_parallel_threads;
  ArrivalTracker(time_unit window_size, int number_of_parallel_threads):
  window_size(window_size), number_of_parallel_threads(number_of_parallel_threads), pool(number_of_parallel_threads){
    // pick x values for the arrival curve
    queries_frequency[1000] = make_tuple(0,0,0);
    queries_frequency[4000] = make_tuple(0,0,0);
    queries_frequency[16000] = make_tuple(0,0,0);
    queries_frequency[64000] = make_tuple(0,0,0);
    queries_frequency[256000] = make_tuple(0,0,0);
    queries_frequency[1024000] = make_tuple(0,0,0);
    queries_frequency[4096000] = make_tuple(0,0,0);
  };

  // function that takes a time_unit x-value, and a time_unit* array of interarrival times.
  // Returns the maximum number of queries that showed up in that interval
  // Notice that this function can be parallelized by cutting the timestamps in half, running this same function on both
  // halves separately, and then combining them by also finding the max of the intersection area
  static bookkeeping_entry 
  arrival_curve_y_value(int id, time_unit x_value, unsigned int start_index, unsigned int end_index, deque<time_unit>& all_timestamps) {
    D(printf("Thread %d called arrival_curve_y_value for x_value %lu\n", id, x_value);)
    if (start_index == end_index) {
      D(printf("arrival_curve_y_value received a call with 0 timestamps!\n");)
      return make_tuple(end_index-start_index+1, start_index, start_index);
    }
    int tail_head = 0; // 0 means tail, 1 means head
    unsigned int head_index = start_index; // the timestamp of index head_index is less than or equal to the sliding time-range's higher end
    unsigned int tail_index = start_index; // the timestamp of index head_index is less than or equal to the sliding time-range's lower end
    unsigned int contained_currently = 1; // at least a single timestamp must be contained in the sliding time-range
    while (head_index < end_index+1) {
      if (all_timestamps[tail_index] + x_value > all_timestamps[head_index]) {
        head_index++;
      } else {
        break;
      }
    }
    if (head_index == end_index+1) {
      // x_value is bigger than the total span of the timestamps array
      D(printf("Range of timestamps is smaller than the x_value!\n");)
      return make_tuple(end_index-start_index+1, start_index, start_index);
    } else {
      // timestamps[head_index] - timestamps[tail_index] > x_value, so we take the head one index back to maintain invariant
      head_index--;
    }
    contained_currently = head_index - tail_index + 1;
    unsigned int index_of_biggest_window = 0;
    unsigned int most_seen_so_far = contained_currently;
    while (head_index < end_index) {
      time_unit distance_to_next_tail = 0;
      time_unit distance_to_next_head = 0;
      if (tail_head == 0) {
        time_unit head_time_position = all_timestamps[tail_index] + x_value;
        distance_to_next_tail = all_timestamps[tail_index+1] - all_timestamps[tail_index];
        distance_to_next_head = all_timestamps[head_index+1] - head_time_position;
      } else if (tail_head == 1) {
        time_unit tail_time_position = all_timestamps[head_index] - x_value;
        distance_to_next_tail = all_timestamps[tail_index+1] - tail_time_position;
        distance_to_next_head = all_timestamps[head_index+1] - all_timestamps[head_index];
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
    return make_tuple(most_seen_so_far,index_of_biggest_window, tail_index+1);
  } 

  vector<pair<time_unit,bookkeeping_entry > > add_timestamps2(time_unit* timestamps, int num_timestamps) {
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
    for (int i = 0; i < num_timestamps; ++i) {
      all_timestamps.push_back(timestamps[i]);
    }
    vector< pair<time_unit, future<bookkeeping_entry> > > futures_vector;
    for (unsigned int i = 0; i < x_values.size(); ++i) {
      // create threadpool, push arrival_curve_y_value with different x_value arguments, get futures back and place in vector
      D(printf("Pushing calculation for x_value %lu\n", x_values[i]);)
      future<bookkeeping_entry> future_result = pool.push(arrival_curve_y_value, x_values[i], 0, all_timestamps.size()-1, all_timestamps);
      futures_vector.push_back(make_pair(x_values[i], move(future_result)));
    }
    vector<pair<time_unit, bookkeeping_entry > > result;
    for (unsigned int i = 0; i < futures_vector.size(); ++i) {
      result.push_back(make_pair(futures_vector[i].first, futures_vector[i].second.get()));
    }
    return result;
  }

  bool add_timestamps(time_unit* timestamps, int num_timestamps) {
    bool increased = false;
    // append to all_timestamps
    for (int i = 0; i < num_timestamps; ++i) {
      all_timestamps.push_back(timestamps[i]);
    }
    // Figure out maximum count of added piece (from the stored last tail_index to the new end)
    for(std::map<time_unit,bookkeeping_entry>::iterator iter = queries_frequency.begin();
        iter != queries_frequency.end(); ++iter) {
      schedule(iter->first, get<2>(iter->second), all_timestamps.size()-1);
    }
    for (unsigned int i = 0; i < queries_frequency.size(); ++i) {
      pair<time_unit, bookkeeping_entry> pair_result = collect();
      time_unit x_value = pair_result.first;
      bookkeeping_entry result = pair_result.second;
      bookkeeping_entry current = queries_frequency[x_value];
      // regardless of the count number, update the tail_index of the last window
      get<2>(current) = get<2>(result); // setting new last tail_index
      // If the count is higher, update the maximum count and the tail index of the maximum count
      if (get<0>(current) < get<0>(result)) {
        get<0>(current) = get<0>(result);
        get<1>(current) = get<1>(result);
        increased = true;
      }
    }
    // find tail index of window size
    unsigned int new_first_index = 0;
    unsigned int timestamps_number = all_timestamps.size();
    for (; new_first_index < timestamps_number; ++new_first_index) {
      if (all_timestamps[new_first_index] + window_size < all_timestamps[timestamps_number-1]) {
        new_first_index++;
      } else {
        break;
      }
    }
    D(printf("new_first_index found to be %d\n", new_first_index);)
    // if the tail isn't curtailed, no work left to be done
    if (new_first_index == 0) {
      return increased;
    }
    // curtail all indices below this tail index
    for (unsigned int i = 0; i < new_first_index; ++i) {
      all_timestamps.pop_front();
    }
    // update all indices to their new values in queries_frequency datastructure
    unsigned int num_scheduled = 0;
    for(std::map<time_unit,bookkeeping_entry>::iterator iter = queries_frequency.begin();
        iter != queries_frequency.end(); ++iter) {
      unsigned int current_max_window_tail_index = get<1>(iter->second);
      if (current_max_window_tail_index < new_first_index) {
        schedule(iter->first, 0, all_timestamps.size()-1); // stopped here
        num_scheduled++;
      } else {
        get<1>(iter->second) = get<1>(iter->second) - new_first_index;
      }
    }
    // if the tail_index of any maximum window is negative, rerun the arrival curve estimation on all x_values
    for (unsigned int i = 0; i < num_scheduled; ++i) {
      pair<time_unit, bookkeeping_entry> result = collect();
      queries_frequency[result.first] = result.second;
    }
    bool decreased = num_scheduled > 0;
    return increased || decreased;
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
  ArrivalTracker tracker (200000000, 4);
  auto result = tracker.add_timestamps(long_deltas.data(), long_deltas.size());
}
