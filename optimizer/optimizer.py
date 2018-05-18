import profiler
import itertools
import copy
import numpy as np
import logging
from multiprocessing import Process, Queue


logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


# logger.setLevel(logging.WARN)


def unwrap_get_arrival_curve_at_x(arg, **kwarg):
    return ArrivalHistory._get_arrival_curve_at_x(*arg, **kwarg)


class ArrivalHistory(object):

    def __init__(self, history):
        """
        Parameters
        ----------
        history : list(float)
            List of arrival times in milliseconds.

        """
        # deltas = np.diff(history)
        # clipped_deltas = np.clip(deltas, a_min=lower_bound_ms, a_max=None)
        # self.history = np.cumsum(deltas)
        self.history = history

    # Approximately compute the x point at which the service curve starts
    # exceeding the arrival curve
    def _get_max_x(self, service_latency, service_throughput):
        # If the average arrival rate is higher than the service_throughput, the
        # maximum point is infinity
        if np.mean(np.diff(self.history)) < 1. / service_throughput:
            # print("Service throughput lower than arrival rate!")
            return np.inf

        def get_service_curve_at_x(service_x_value):
            return 0 if service_x_value < service_latency else float(
                service_throughput) * (service_x_value - service_latency)
        # increase x_coordinate exponentially to find a coordinate where the
        # service curve definately exceeds the arrival curve
        # print("Initializing maximum x coordinate")
        x_coordinate = 1.
        service_y_point = get_service_curve_at_x(x_coordinate)
        _, arrival_y_point = self._get_arrival_curve_at_x(x_coordinate)
        while(service_y_point < arrival_y_point):
            x_coordinate = x_coordinate * 2.
            service_y_point = get_service_curve_at_x(x_coordinate)
            _, arrival_y_point = self._get_arrival_curve_at_x(x_coordinate)
        # Do a telescoping search to get the x_coordinate close to the intersection of the arrival
        # and service curve.
        # 1 is a magic number: We want to continue to telescope into the range of x values
        # [left_bound, right_bound] where the service and arrival curves are at most 1 queue length
        # away from each other
        # print("Telescoping on x coordinate")
        left_bound = x_coordinate / 2.
        right_bound = x_coordinate
        middle = (left_bound + right_bound) / 2.
        service_y_point = get_service_curve_at_x(middle)
        _, arrival_y_point = self._get_arrival_curve_at_x(middle)
        while(abs(service_y_point - arrival_y_point) > 1.):
            # print(left_bound, right_bound, middle, service_y_point, arrival_y_point)
            if np.isclose(middle, np.max(self.history)):
                # print("Setting maximum x coordinate to be maximum of history")
                break
            if arrival_y_point > service_y_point:
                left_bound = middle
            else:
                right_bound = middle
            middle = (left_bound + right_bound) / 2.
            service_y_point = get_service_curve_at_x(middle)
            _, arrival_y_point = self._get_arrival_curve_at_x(middle)
        return middle

    # Compute maximum vertical gap from arrival curve to service curve
    def _max_Q_given_arrival(self, arrival_x, arrival_y, service_latency, service_throughput):
        largest_gap = 0
        for arrival_x_point, arrival_y_point in zip(arrival_x, arrival_y):
            service_y_point = 0 if arrival_x_point < service_latency else (
                arrival_x_point - service_latency) * float(service_throughput)
            if arrival_y_point - service_y_point > largest_gap:
                largest_gap = arrival_y_point - service_y_point
        return largest_gap

    # Compute maximum horizontal gap from arrival curve to service curve
    def _max_response_time_given_arrival(
            self, arrival_x, arrival_y, service_latency, service_throughput):
        largest_gap = 0
        for arrival_x_point, arrival_y_point in zip(arrival_x, arrival_y):
            service_x_point = arrival_y_point / service_throughput + service_latency
            if service_x_point - arrival_x_point > largest_gap:
                largest_gap = service_x_point - arrival_x_point
        return largest_gap

    # Compute the arrival curve's y values at a particular x value
    def _get_arrival_curve_at_x(self, arrival_x_value, queue=None):
        def get_smallest_delta_2(time_range, timestamps):
            head_index = 0  # the first index less than or equal to time_range's higher end
            tail_index = 0  # the first index less than or equal to time_range's lower end
            position = 'tail'  # start when time_range's lower end (tail) is aligned with a point
            # a single point (the first index) must be contained in the time_range
            contained_currently = 1
            # add the rest of the timepoints contained in time_range
            for i in range(tail_index + 1, len(timestamps)):
                if timestamps[i] <= timestamps[tail_index] + time_range:
                    contained_currently += 1
                    head_index += 1
                else:
                    break
            max_so_far = contained_currently
            # this means time_range's higher end hasn't exceeded the very last timestamp
            while head_index < len(timestamps) - 1:
                if position == "tail":
                    head_time_position = timestamps[tail_index] + time_range
                    head_delta = timestamps[head_index + 1] - head_time_position
                    assert head_delta > 0 or np.isclose(head_delta, 0)
                    tail_delta = timestamps[tail_index + 1] - timestamps[tail_index]
                elif position == "head":
                    tail_time_position = timestamps[head_index] - time_range
                    tail_delta = timestamps[tail_index + 1] - tail_time_position
                    assert tail_delta >= 0 or np.isclose(tail_delta, 0)
                    head_delta = timestamps[head_index + 1] - timestamps[head_index]
                if tail_delta < head_delta:
                    position = "tail"
                    tail_index = tail_index + 1  # head_index stays the same
                    contained_currently -= 1
                elif head_delta <= tail_delta:
                    position = "head"
                    head_index = head_index + 1  # tail_index stays the same
                    contained_currently += 1
                assert contained_currently >= head_index - tail_index
                if contained_currently > max_so_far:
                    max_so_far = contained_currently
            return max_so_far
        result = get_smallest_delta_2(arrival_x_value, self.history)
        # print("_get_arrival_curve_at_x("+str(arrival_x_value)+") = "+str(result))
        if queue:
            queue.put((arrival_x_value, result))
        return (arrival_x_value, result)

    def get_max_Q_and_time(self, latency, throughput):

        max_x = self._get_max_x(latency, throughput)
        if max_x == np.inf:
            return (np.inf, np.inf)
        # Want to plot arrival curve at a granularity of about 200 points. More point means a
        # smoother and a marginally more accurate estimate plot of the arrival curve, but also
        # requires more computation time
        arrival_x = np.linspace(1, max_x, 200)
        chunk_size = 40
        # arrival_y = []
        arrival_xy_map = {}
        # This for loop is just to limit the degree of parallelism
        for chunk in range(len(arrival_x) // chunk_size):
            queue = Queue()
            procs = []
            for x in arrival_x[chunk*chunk_size : (chunk+1)*chunk_size]:
                p = Process(target=self._get_arrival_curve_at_x, args=(x, queue))
                p.start()
                procs.append(p)
            for p in procs:
                arr_x, arr_y = queue.get()
                arrival_xy_map[arr_x] = arr_y
                # arrival_y.append(queue.get())
                # p.join()
        sorted_arr_xy_pairs = sorted(list(arrival_xy_map.items()))
        arrival_x, arrival_y = zip(*sorted_arr_xy_pairs)
        for i in range(len(arrival_x)):
            assert arrival_y[i] == arrival_xy_map[arrival_x[i]]

        return (self._max_Q_given_arrival(arrival_x, arrival_y, latency, throughput),
                self._max_response_time_given_arrival(arrival_x, arrival_y, latency, throughput))

class GreedyOptimizer(object):

    def __init__(self, dag, scale_factors, node_profs):
        self.dag = dag
        self.scale_factors = scale_factors
        self.node_profs = node_profs

    def select_optimal_config(self,
                              cloud,
                              latency_constraint,
                              cost_constraint,
                              initial_config,
                              arrival_history=None):
        execution_history = {
            "cur_pipeline_config":[],
            "action": [],
            "throughput": [],
            "cost": []
            
        }
        arrival_history_obj = ArrivalHistory(arrival_history)
        cur_pipeline_config = initial_config
        if not profiler.is_valid_pipeline_config(cur_pipeline_config):
            logger.error("ERROR: provided invalid initial pipeline configuration")
        iteration = 0
        latency_slo_met = False
        last_action_response_time = np.inf
        while True:
            def try_upgrade_gpu(bottleneck, pipeline_config):
                """
                Returns either the new config or False if the GPU could not be upgraded
                """
                return self.node_profs[bottleneck].upgrade_gpu(
                    pipeline_config[bottleneck])

            def try_increase_batch_size(bottleneck, pipeline_config):
                """
                Returns either the new config or False if the batch_size could not be increased
                """
                return self.node_profs[bottleneck].increase_batch_size(
                    pipeline_config[bottleneck])

            def try_increase_replication_factor(bottleneck, pipeline_config):
                """
                Returns the new config with an increased replication factor
                """
                new_bottleneck_config = copy.deepcopy(pipeline_config[bottleneck])
                new_bottleneck_config.num_replicas += 1
                return new_bottleneck_config

            cur_estimated_perf, cur_bottleneck_node = \
                profiler.estimate_pipeline_performance_for_config(
                    self.dag, self.scale_factors, cur_pipeline_config, self.node_profs)

            cur_bottleneck_config = cur_pipeline_config[cur_bottleneck_node]
            cur_bottleneck_node_lat, cur_bottleneck_node_thru, cur_bottleneck_node_cost = \
                self.node_profs[cur_bottleneck_node].estimate_performance(cur_bottleneck_config)

            actions = {
				"gpu": try_upgrade_gpu,
                "batch_size": try_increase_batch_size,
                "replication_factor": try_increase_replication_factor
            }

            best_action = None
            best_action_thru = None
            best_action_config = None
            best_action_response_time = np.inf

            for action in actions:
                logger.info("\n\nEvaluating step {}".format(action))
                new_bottleneck_config = actions[action](cur_bottleneck_node, cur_pipeline_config)
                if new_bottleneck_config:
                    logger.info("\tOld config: {}\nNew config: {}".format(
                        cur_pipeline_config[cur_bottleneck_node], new_bottleneck_config))
                    copied_pipeline_config = copy.deepcopy(cur_pipeline_config)
                    copied_pipeline_config[cur_bottleneck_node] = new_bottleneck_config
                    result = profiler.estimate_pipeline_performance_for_config(
                        self.dag, self.scale_factors, copied_pipeline_config, self.node_profs)
                    if result is None:
                        continue
                    new_estimated_perf, new_bottleneck_node = result
                    logger.info("\tNew estimated perf: {}".format(new_estimated_perf))
                    # We'll never take this action if it violates the cost constraint
                    if new_estimated_perf["cost"] > cost_constraint:
                        logger.info("Cost {} higher than constraint {}, skipping this action".format(new_estimated_perf["cost"], cost_constraint))
                        continue
                    logger.info("\tDoing network calc")
                    netcalc_config = new_bottleneck_config
                    T_S = new_estimated_perf["latency"]
                    _, Q_waiting_time = arrival_history_obj.get_max_Q_and_time(0, #T_S * 1000.0,
                            new_estimated_perf["throughput"] / 1000.)

                    # converting time to seconds
                    T_Q = Q_waiting_time / 1000.0
                    response_time = T_Q + T_S
                    # assert T_Q >= T_S

                    logger.info("\tResponse time: {total}, T_s={ts}, T_q={tq}".format(total=response_time,
                        ts=T_S, tq=T_Q))
                    

                    if latency_slo_met:
                        latency_to_compare = response_time
                    else:
                        latency_to_compare = T_S
                    if (latency_to_compare <= latency_constraint and
                        new_estimated_perf["cost"] <= cost_constraint):
                            # and 2*T_S <= latency_constraint):
                        if new_estimated_perf["throughput"] < cur_estimated_perf["throughput"]:
                            logger.warning(
                                ("Uh oh: monotonicity violated:\n Old config: {}, Thru: {}"
                                 "\n New config: {}, Thru: {}").format(
                                     cur_pipeline_config[cur_bottleneck_node],
                                     cur_estimated_perf["throughput"],
                                     new_bottleneck_config,
                                     new_estimated_perf["throughput"]
                                 ))
                        if best_action is None or \
                                best_action_thru < new_estimated_perf["throughput"]:
                            best_action = action
                            best_action_thru = new_estimated_perf["throughput"]
                            best_action_config = new_bottleneck_config
                            best_action_response_time = response_time
                            logger.info("\tSetting best action response time to {}".format(best_action_response_time))
                else:
                    logger.info("\tAction configuration returned as None")

            # No more steps can be taken
            if best_action is None:
                logger.info("No more steps can be taken")
                break
            else:
                cur_pipeline_config[cur_bottleneck_node] = best_action_config
                logger.info(("Upgrading bottleneck node {bottleneck} to {new_config}."
                    "\nIncreased.\nNew config: {new_conf}").format(
                                 bottleneck=cur_bottleneck_node,
                                 new_config=best_action_config,
                                 new_conf=cur_pipeline_config))
                # Once latency_slo_met is set to True, it should never be set to False again
                if not latency_slo_met:
                    latency_slo_met = best_action_response_time <= latency_constraint
                    if latency_slo_met:
                        logger.info("LATENCY SLO MET\nConfig:{}".format(cur_pipeline_config))
                last_action_response_time = best_action_response_time
                logger.info("Finished iteration {}".format(iteration))
            iteration += 1

        # Finally, check that the selected profile meets the application constraints, in case
        # the user provided unsatisfiable application constraints
        cur_estimated_perf, _ = profiler.estimate_pipeline_performance_for_config(
            self.dag, self.scale_factors, cur_pipeline_config, self.node_profs)

        # last_action_response_time is the response time for the last upgrade action that
        # the optimizer took before terminating iteration
        if (last_action_response_time <= latency_constraint and
                cur_estimated_perf["cost"] <= cost_constraint):
            return cur_pipeline_config, cur_estimated_perf, last_action_response_time
        else:
            logger.info(("\n\nError: No configurations found that satisfy application constraints.\n"
                          "Latency constraint: {lat_const}, estimated response latency: {lat_est}\n"
                          "Cost constraint: {cost_const}, estimated cost: {cost_est}\nCURRENT CONFIG: {cur_config}").format(
                              lat_const=latency_constraint, lat_est=last_action_response_time,
                              cost_const=cost_constraint, cost_est=cur_estimated_perf["cost"],
                              cur_config=cur_pipeline_config))
            return False
