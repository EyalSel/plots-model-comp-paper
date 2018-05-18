import numpy as np
from itertools import count
from collections import defaultdict
from time import time



class Query(object):
    global_ids = count(0)
    
    def __init__(self, start_time):
        self.id = next(self.global_ids)
        self.start_time = start_time
        self.finish_time = np.inf

    def __str__(self):
        return "q({id}:{start_time}->{finish_time})".format(
            id = self.id, start_time=self.start_time,
            finish_time = self.finish_time) 
    
    def __repr__(self):
        return str(self)
    
    def time_remaining(self, current_time):
        return current_time - self.start_time


class Clock(object):
    def __init__(self):
        self.current_time = 0
        self.events = {}

    def now(self):
        return self.current_time

    def set_timer(self, node, time):
        self.events[node] = time

    def tick(self):
        if len(self.events) == 0:
            return False

        self.current_time = min(self.events.values())
        active_events = []
        for (ev, timer) in self.events.items():
            if timer <= self.current_time:
                active_events.append(ev)
        # TODO randomize active event order
        for ev in active_events:
            self.events.pop(ev)
            ev.tick()

        return True

    def run(self):
        last_time = time()
        while(self.tick()):
            if (time() - last_time) > 1:
                print(self.now())
                last_time = time()

class Node(object):
    def __init__(self, clock):
        self.clock = clock
        self.children = []
        self.parents = []

    def then(self, child):
        self.children.append(child)
        child.parents.append(self)

    def add_queries(self, queries):
        pass

class SourceNode(Node):
    def __init__(self, clock, queries):
        Node.__init__(self, clock)
        self.queries = sorted(queries, key = lambda x: x.start_time)
        self.clock.set_timer(self, self.queries[0].start_time)
        self.waterline = 0
    
    def tick(self):
        start = self.waterline 
        while (self.waterline < len(self.queries) and 
            self.queries[self.waterline].start_time <= self.clock.now()):
            self.waterline += 1
        # might want to randomize children
        for c in self.children:
            c.add_queries(self.queries[start:self.waterline])
        # schedule self again
        if self.waterline < len(self.queries):
            self.clock.set_timer(self, 
                self.queries[self.waterline].start_time)
    
class BatchedNode(Node):
    def __init__(self, clock, batch_fn, max_batch_size = 10):
        Node.__init__(self, clock)
        self.max_batch_size = max_batch_size
        self.in_queue = []
        self.current_batch = []
        self.batch_timer = -1
        self.batch_fn = batch_fn

    def batch_time(self, batch_size):
        return self.batch_fn(batch_size)

    def add_queries(self, queries):
        self.in_queue += queries
        self.start_new_batch()

    def start_new_batch(self):
        batch_size = min(self.max_batch_size, len(self.in_queue))
        if len(self.current_batch) == 0 and batch_size > 0:
            self.current_batch = self.in_queue[:batch_size]
            self.in_queue = self.in_queue[batch_size:]
            self.batch_timer = self.clock.now() + self.batch_time(batch_size)
            self.clock.set_timer(self, self.batch_timer)

    def tick(self):
        assert self.clock.now() >= self.batch_timer and len(self.current_batch) > 0
        for c in self.children:
            c.add_queries(self.current_batch)
        self.current_batch = []
        self.start_new_batch()


class JoinNode(Node):
    def __init__(self, clock):
        Node.__init__(self, clock)
        self.join_map = defaultdict(list)

    def add_queries(self, queries):
        for q in queries:
            self.join_map[q.id].append(q)
            # ASSUME only once delivery
            if len(self.join_map[q.id]) == len(self.parents):
                for c in self.children:
                    c.add_queries([q])

class IfNode(Node):
    def __init__(self, clock):
        Node.__init__(self, clock)
        self.child_prob = {}

    def then(self, child, prob):
        Node.then(self, child)
        self.child_prob[child] = prob

    def add_queries(self, queries):
        children = list(self.child_prob.keys())
        p = np.array(list(self.child_prob.values()))
        p = p / p.sum()
        for q in queries:
            c = np.random.choice(children, p=p)
            c.add_queries([q])


class SinkNode(Node):
    def __init__(self, clock):
        Node.__init__(self, clock)
        self.finished = []

    def add_queries(self, queries):
        for q in queries:
            q.finish_time = self.clock.now()
        self.finished += queries










        

