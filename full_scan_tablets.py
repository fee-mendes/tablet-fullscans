#!/usr/bin/env python3 

import json
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Set, DefaultDict
import threading
import time
import logging
from datetime import datetime
from queue import Queue, Empty
from collections import defaultdict
import concurrent.futures
import numpy as np
from datetime import datetime
import argparse

MIN_TOKEN = -2**63
MAX_TOKEN = 2**63 - 1

def nearest_base2_tablets(target_tablets_per_shard, total_shards, replication_factor):
    # Computes optimal tablet count -- assumes even shards/node
    t_optimal = (target_tablets_per_shard * total_shards) / replication_factor
    
    # Rounds to nearest base-2
    t_base2 = 2 ** math.ceil(math.log2(t_optimal))
    
    return t_base2

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)

@dataclass
class QueryMetrics:
    start_time: datetime
    end_time: datetime = None
    duration: float = 0.0

@dataclass
class MetricsCollector:
    # Track metrics per node-shard combination
    node_shard_requests: Dict[str, Dict[int, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    node_shard_latencies: Dict[str, Dict[int, List[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    total_requests: int = 0
    scheduled_requests: int = 0
    completed_requests: int = 0
    completion_lock: threading.Lock = field(default_factory=threading.Lock)

    def get_diagnostic_info(self) -> str:
        """Get diagnostic information about current state"""
        with self.completion_lock:
            return (f"\nDiagnostic Info:"
                    f"\nTotal Scheduled: {self.scheduled_requests}"
                    f"\nTotal Completed: {self.completed_requests}"
                    f"\nRemaining: {self.scheduled_requests - self.completed_requests}")

    def record_query_scheduled(self):
        """Record that a query has been scheduled"""
        with self.completion_lock:
            self.scheduled_requests += 1

    def record_query_start(self, node: str, shard_id: int) -> QueryMetrics:
        """Record the start of a query"""
        with self.completion_lock:
            self.node_shard_requests[node][shard_id] += 1
            self.total_requests += 1
        return QueryMetrics(start_time=datetime.now())

    def record_query_end(self, node: str, shard_id: int, metrics: QueryMetrics):
        """Record the end of a query"""
        metrics.end_time = datetime.now()
        metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
        with self.completion_lock:
            self.node_shard_latencies[node][shard_id].append(metrics.duration)
            self.completed_requests += 1

    def is_complete(self) -> bool:
        """Check if all scheduled queries have completed"""
        with self.completion_lock:
            return self.scheduled_requests > 0 and self.completed_requests >= self.scheduled_requests

    def get_completion_percentage(self) -> float:
        """Get the percentage of completed queries"""
        with self.completion_lock:
            if self.scheduled_requests == 0:
                return 0.0
            return (self.completed_requests / self.scheduled_requests) * 100

    def get_statistics(self) -> str:
        """Generate a complete statistics report"""
        report = []
        report.append("\nQuery Processing Statistics")
        report.append("=" * 30)

        # Global statistics
        report.append(f"\nTotal Requests Scheduled: {self.scheduled_requests}")
        report.append(f"Total Requests Completed: {self.completed_requests}")
        completion_percentage = self.get_completion_percentage()
        report.append(f"Completion Percentage: {completion_percentage:.1f}%")

        # Per-node statistics
        report.append("\nPer-Node Statistics:")
        report.append("-" * 20)
        for node in sorted(self.node_shard_requests.keys()):
            total_node_requests = sum(self.node_shard_requests[node].values())
            all_node_latencies = []
            for shard_latencies in self.node_shard_latencies[node].values():
                all_node_latencies.extend(shard_latencies)

            if all_node_latencies:
                avg_latency = np.mean(all_node_latencies)
                p99_latency = np.percentile(all_node_latencies, 99)
            else:
                avg_latency = p99_latency = 0

            report.append(f"\nNode {node}:")
            report.append(f"  Total Requests: {total_node_requests}")
            report.append(f"  Average Latency: {avg_latency:.3f}s")
            report.append(f"  P99 Latency: {p99_latency:.3f}s")

            # Per-shard statistics for this node
            report.append("\n  Per-Shard Breakdown:")
            for shard_id in sorted(self.node_shard_requests[node].keys()):
                requests = self.node_shard_requests[node][shard_id]
                latencies = self.node_shard_latencies[node][shard_id]

                if latencies:
                    avg_shard_latency = np.mean(latencies)
                    p99_shard_latency = np.percentile(latencies, 99)
                else:
                    avg_shard_latency = p99_shard_latency = 0

                report.append(f"    Shard {shard_id}:")
                report.append(f"      Requests: {requests}")
                report.append(f"      Average Latency: {avg_shard_latency:.3f}s")
                report.append(f"      P99 Latency: {p99_shard_latency:.3f}s")

        return "\n".join(report)

@dataclass
class QueryRange:
    start_token: int
    end_token: Optional[int]
    shard_id: int
    tablet_id: int
    subrange_id: int

@dataclass
class ShardReplica:
    shard_id: int
    replica: str
    inflight: int = 0
    queue: "Queue[QueryRange]" = field(default_factory=Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_load(self) -> int:
        return self.inflight + self.queue.qsize()

@dataclass
class QueryWithReplicas:
    query: QueryRange
    replicas: List[str]

@dataclass
class ShardScheduler:
    """Handles scheduling for a single shard"""
    shard_id: int
    max_inflight: int
    load_threshold: int
    pending_queries: Queue[QueryWithReplicas] = field(default_factory=Queue)
    replicas: Dict[str, ShardReplica] = field(default_factory=dict)
    scheduler_threads: List[threading.Thread] = field(default_factory=list)
    num_schedulers: int = 1
    shutdown: bool = False

    def start_scheduler(self, try_schedule_fn):
        """Start background scheduler thread for this shard"""
        for i in range(self.num_schedulers):
            thread = threading.Thread(
                 target=self._scheduler_loop,
                 args=(try_schedule_fn,),
                 name=f"Scheduler-Shard-{self.shard_id}"
            )
            thread.daemon = True
            thread.start()
            self.scheduler_threads.append(thread)

    def _scheduler_loop(self, try_schedule_fn):
        """Background thread that processes pending queries for this shard"""
        while not self.shutdown:
            try:
                query_with_replicas = self.pending_queries.get(timeout=1)
                if not try_schedule_fn(query_with_replicas, self):
                    self.pending_queries.put(query_with_replicas)
                    time.sleep(0.01)
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in shard {self.shard_id} scheduler loop: {e}")

class QueryScheduler:
    def __init__(self, shard_map: Dict, max_inflight: int = None):
        self.shard_map = shard_map
        self.max_inflight = max_inflight
        self.load_threshold = max_inflight * 2
        self.shard_schedulers: Dict[int, ShardScheduler] = {}
        self.active_workers = set()
        self.shutdown = False
        self.metrics = MetricsCollector()
        self.completion_event = threading.Event()
        self.total_requests = 0
        self.completed_requests = 0
        self.completion_lock = threading.Lock()
        self.init_schedulers()

    def init_schedulers(self):
        """Initialize per-shard schedulers"""
        for shard_id, tablets in self.shard_map.items():
            # Create scheduler for this shard
            num_schedulers = max(8, NODE_COUNT)
            scheduler = ShardScheduler(
                shard_id=shard_id,
                max_inflight=self.max_inflight,
                load_threshold=self.load_threshold,
                num_schedulers=num_schedulers
            )
            
            # Initialize replicas for this shard
            for tablet in tablets:
                for replica in tablet['replicas']:
                    scheduler.replicas[replica] = ShardReplica(
                        shard_id=shard_id,
                        replica=replica,
                        queue=Queue(), # Explicit new Queue for each replica
                        lock=threading.Lock()
                    )
            
            self.shard_schedulers[shard_id] = scheduler
            scheduler.start_scheduler(self._try_schedule_query)

    def get_queue_depths(self) -> str:
        """Get queue depths for all shard replicas"""
        depths = []
        for shard_id, scheduler in self.shard_schedulers.items():
            for replica, shard_replica in scheduler.replicas.items():
                with shard_replica.lock:
                    queue_size = shard_replica.queue.qsize()
                    inflight = shard_replica.inflight
                    depths.append(f"Shard {shard_id} Replica {replica}:"
                                f" Queue={queue_size}, Inflight={inflight}")
        return "\n".join(depths)

    def schedule_query(self, query: QueryRange, replicas: List[str]):
        """Schedule query to appropriate shard scheduler"""
        shard_scheduler = self.shard_schedulers[query.shard_id]
        shard_scheduler.pending_queries.put(QueryWithReplicas(query, replicas))
        self.metrics.record_query_scheduled()

    def _try_schedule_query(self, query_with_replicas: QueryWithReplicas, shard_scheduler: ShardScheduler) -> bool:
        """Attempt to schedule a query to a replica under load threshold"""
        query, replicas = query_with_replicas.query, query_with_replicas.replicas
        min_load = float('inf')
        chosen_replica = None
        
        # Find replica with minimum load that's under threshold
        for replica in replicas:
            shard_replica = shard_scheduler.replicas[replica]
            with shard_replica.lock:
                current_load = shard_replica.get_load()
                if current_load < min_load and current_load < shard_scheduler.load_threshold:
                    min_load = current_load
                    chosen_replica = replica
        
        if chosen_replica:
            # Schedule to chosen replica
            shard_scheduler.replicas[chosen_replica].queue.put(query)
            return True
        
        return False

    def process_single_query(self, query: QueryRange, shard_replica: ShardReplica, worker_id: str):
        try:
            query_metrics = self.metrics.record_query_start(shard_replica.replica, query.shard_id)
            process_time = 0.1 * random.random()  # Random processing time

            logging.debug(f"[{worker_id}] Starting query token > {query.start_token} AND token <= {query.end_token}")
            time.sleep(process_time)  # Simulate processing

            self.metrics.record_query_end(shard_replica.replica, query.shard_id, query_metrics)
            logging.debug(f"[{worker_id}] Completed query token > {query.start_token} AND token <= {query.end_token} in {process_time:.2f}s")

        except Exception as e:
            logging.error(f"Error in worker {worker_id}: {e}")
        finally:
            with shard_replica.lock:
                shard_replica.inflight -= 1

    def process_shard_replica(self, shard_id: int, replica: str):
        """Worker function processes queries for a specific shard-replica"""
        shard_scheduler = self.shard_schedulers[shard_id]
        shard_replica = shard_scheduler.replicas[replica]
        worker_id = f"Worker-{replica}-{shard_id}"
        self.active_workers.add(worker_id)
        
        logging.info(f"Started worker {worker_id}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_inflight) as executor:
            active_futures = []
            
            while not self.shutdown:
                # Clean up completed futures
                active_futures = [f for f in active_futures if not f.done()]
                
                # Submit new work if capacity available
                while len(active_futures) < self.max_inflight:
                    try:
                        query = shard_replica.queue.get_nowait()
                        with shard_replica.lock:
                            shard_replica.inflight += 1
                        
                        future = executor.submit(
                            self.process_single_query,
                            query,
                            shard_replica,
                            f"{worker_id}-{len(active_futures)}"
                        )
                        active_futures.append(future)
                    # If no work is available, check if we're done
                    except Empty:
                        if self.metrics.is_complete():
                            if not active_futures:
                                return
                        # Otherwise keep checking for new work
                        time.sleep(0.01)
                        break
                
                if len(active_futures) == self.max_inflight:
                    concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
        
        print(f"{worker_id} - Left loop and ready to remove worker")
        self.active_workers.remove(worker_id)
        logging.info(f"Stopped worker {worker_id}")

    def shutdown_schedulers(self):
        """Shutdown all shard schedulers"""
        self.shutdown = True
        for scheduler in self.shard_schedulers.values():
            scheduler.shutdown = True

class TabletMap:
    def __init__(self):
        self.nodes = list(map(chr, range(97, 97 + NODE_COUNT)))
        self.shard_map = self._generate_tablet_map()
        self.scheduler = QueryScheduler(self.shard_map, INFLIGHT_PER_SHARD_REPLICA)
        
    def _split_to_subranges(self, previous_token: int, last_token: int, splits: None, wrap=False) -> List[Union[List[int], int]]:
        if wrap:
            return [[previous_token, last_token]]  # Return as list of lists for consistency
        
        ranges = []
        start = previous_token
        step = int((last_token - previous_token) / splits)
        
        for i in range(splits):
            end = start + step if i < splits - 1 else last_token
            ranges.append([start, end])
            start = end
            
        return ranges

    def _generate_tablet_map(self) -> Dict:
        ring = MAX_TOKEN + (MIN_TOKEN * -1)
        step = int(ring / TOTAL_TABLETS)  # Now using TOTAL_TABLETS instead of SHARDS * TABLETS_PER_SHARD
        
        shard = {key: [] for key in range(SHARDS)}
        shard_number = 0
        previous_token = MIN_TOKEN
        tablets_assigned = 0
        
        while tablets_assigned < TOTAL_TABLETS:
            last_token = previous_token + step
            replicas = random.sample(self.nodes, REPLICATION_FACTOR)
            tablet = {
                'last_token': last_token,
                'replicas': replicas,
                'subranges': self._split_to_subranges(previous_token, last_token, SUBRANGE_SPLITS)
            }
            
            shard[shard_number].append(tablet)
            tablets_assigned += 1
            previous_token = last_token
            
            # Round-robin across shards
            shard_number = (shard_number + 1) % SHARDS
        
        # Handle the wrap-around case for the last tablet
        last_shard = (SHARDS - 1)
        shard[last_shard][-1]['subranges'].extend(
            self._split_to_subranges(last_token, MIN_TOKEN, SUBRANGE_SPLITS, wrap=True)
        )
        
        return shard

    def run_concurrent_queries(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=NODE_COUNT*SHARDS) as executor:
            futures = []
            
            # Start a worker for every node-shard combination
            for node in self.nodes:  # Loop through all nodes
                for shard_id in range(SHARDS):  # Loop through all shards
                    futures.append(executor.submit(
                        self.scheduler.process_shard_replica, shard_id, node
                    ))

            # Schedule all queries
            for shard_id, tablets in self.shard_map.items():
                for tablet_id, tablet in enumerate(tablets):
                    for subrange_id, subrange in enumerate(tablet['subranges']):
                        query = QueryRange(
                            start_token=subrange[0],
                            end_token=subrange[1],
                            shard_id=shard_id,
                            tablet_id=tablet_id,
                            subrange_id=subrange_id
                        )
                        self.scheduler.schedule_query(query, tablet['replicas'])

            # Poll for completion
            start_time = time.time()
            last_progress_time = start_time
            last_percentage = 0

            while True:
                if self.scheduler.metrics.is_complete():
                    print("\nAll queries completed successfully!")
                    break

                current_percentage = self.scheduler.metrics.get_completion_percentage()
                current_time = time.time()
                if current_percentage != last_percentage:
                    print(f"\rProgress: {current_percentage:.1f}%", end="")
                    last_percentage = current_percentage
                    last_progress_time = current_time
                elif current_time - last_progress_time > 10:
                    print("\nNo progress for 10 seconds. Dumping diagnosis:")
                    print(self.scheduler.metrics.get_diagnostic_info())
                    print("\nQueue depths:")
                    print(self.scheduler.get_queue_depths())
                    last_progress_time = current_time

                time.sleep(1)

            print(f"Target tablets per shard: {TARGET_TABLETS_PER_SHARD}")
            print(f"Actual tablets per shard: {TABLETS_PER_SHARD}")
            print(f"Total tablets: {TOTAL_TABLETS}")
            print(f"Total shards: {SHARDS * NODE_COUNT}")
            print(f"Replication factor: {REPLICATION_FACTOR}")
            print(self.scheduler.metrics.get_statistics())

            # Now safe to shutdown
            self.scheduler.shutdown = True
            concurrent.futures.wait(futures, timeout=10)

def main():
    tablet_map = TabletMap()
    print(f"\nStarting concurrent query processing with {INFLIGHT_PER_SHARD_REPLICA} inflight requests per shard-replica...")
    start_time = time.time()
    tablet_map.run_concurrent_queries()
    end_time = time.time()
    print("--- Started at: %s UTC ---" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)))
    print("--- Ended at: %s UTC ---" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end_time)))
    print("--- Runtime: %.2f seconds ---" % (end_time - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tablet-based full scan simulation')
    parser.add_argument('--splits', type=int, default=16,
                        help='Number of subranges to split each tablet into (default: 16)')
    parser.add_argument('--shards', type=int, default=2,
                        help='Number of shards per node (default: 2)')
    parser.add_argument('--node-count', type=int, default=3,
                        help='Number of nodes in the cluster (default: 3)')
    parser.add_argument('--replication-factor', type=int, default=3,
                        help='Replication factor for the table (default: 3)')
    parser.add_argument('--tablets-count', type=int, default=128,
                        help='Target number of tablets per shard (default: 128)')
    parser.add_argument('--inflight', type=int, default=2,
                        help='Number of concurrent inflight requests per shard (default: 2)')

    args = parser.parse_args()

    global SUBRANGE_SPLITS, SHARDS, NODE_COUNT, REPLICATION_FACTOR, TARGET_TABLETS_PER_SHARD, INFLIGHT_PER_SHARD_REPLICA
    global TOTAL_TABLETS, TABLETS_PER_SHARD

    SUBRANGE_SPLITS = args.splits
    SHARDS = args.shards
    NODE_COUNT = args.node_count
    REPLICATION_FACTOR = args.replication_factor
    TARGET_TABLETS_PER_SHARD = args.tablets_count
    INFLIGHT_PER_SHARD_REPLICA = args.inflight

    # Calculate actual number of tablets
    TOTAL_TABLETS = int(nearest_base2_tablets(TARGET_TABLETS_PER_SHARD, SHARDS * NODE_COUNT, REPLICATION_FACTOR))
    TABLETS_PER_SHARD = (REPLICATION_FACTOR / NODE_COUNT) * TOTAL_TABLETS / SHARDS

    main()
