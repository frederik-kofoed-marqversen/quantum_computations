from numpy.random import Generator, default_rng
from sequence_class import Stage, LogicalDistillationSequence
import numpy as np
from bisect import insort

import logging
logger = logging.getLogger(__name__)


class ActiveStage:
    def __init__(self, stage: Stage, p_fail: float, K_in: int):
        self.n = stage.n
        self.k = stage.k
        self.distil_steps = stage.get_physical_depth()
        self.p = p_fail
        self.K_in = K_in
        
        self.qubit_size = stage.qubit_size
        
        self.process_timers = []
        self.output_buffer = 0
    
    def init(self, n: int=1) -> None:
        # Init processes
        self.process_timers += [self.distil_steps] * n

    def step(self, rng: Generator) -> None:
        # Progress processes
        new_process_timers = []
        for t in self.process_timers:
            if t <= 0: # Handle finished processes
                if rng.random() > self.p:
                    self.output_buffer += 1
            else: # Progress processesstep_time
                new_process_timers.append(t - 1)
        self.process_timers = new_process_timers
    
    def memory_usage(self) -> int:
        idle = self.output_buffer * self.k * self.K_in
        active = len(self.process_timers) * self.n * self.K_in
        return (active + idle) * self.qubit_size
    
    def active_processes(self) -> int:
        return len(self.process_timers)

class Simulator():
    def __init__(
        self, 
        space: int, 
        input_rate: float, 
        dist_seq: LogicalDistillationSequence, 
        rng_seed: int=42
    ):
        if space < dist_seq.min_memory_req:
            raise ValueError("Insurficient memory for given distillation sequence")
        
        self.M = space
        self.rng = default_rng(rng_seed)
        # Input rate measured in units of local_gate_rate
        self.input_rate = dist_seq.distillation_rate(space, input_rate) / dist_seq.encoding_rate
        self.stages: list[ActiveStage] = []
        self.K = dist_seq.K
        self.Ns = []
        self.dMs = []
        
        self.output = 0
        self.input_stage = ActiveStage.__new__(ActiveStage)
        self.input_stage.output_buffer = 0
        
        self.E = 1
        K = 1
        s = 0
        for stage, p_fail in zip(dist_seq.stages, dist_seq.stage_p_fail):
            # Add stage to simulation
            self.stages.append(ActiveStage(stage, p_fail, K))
            # Compute stage memory increase
            dM = K * stage.n * (stage.qubit_size - s)
            self.dMs.append(dM)
            s = stage.qubit_size
            # Compute stage "quotas"
            N = self.input_rate * stage.get_physical_depth() * self.E / stage.n
            self.Ns.append(N)
            # Update values for next stage
            self.E *= (1 - p_fail) * stage.k / stage.n
            K *= stage.k
    
    def memory_usage(self) -> int:
        return sum(stage.memory_usage() for stage in self.stages)
    
    def init_processes(self):
        # Sort stages by gap
        gaps = [N - s.active_processes() for N, s in zip(self.Ns, self.stages)]
        candidates = zip(self.stages, gaps, range(len(gaps)))
        # candidates = [item for item in candidates if item[1] > 0]  # Retain only positive gaps
        candidates = sorted(candidates, key=lambda item: item[1])
        
        # Schedule new processes according to quota rule
        available_memory = self.M - self.memory_usage()
        while candidates:
            stage, gap, i = candidates.pop()
            prev = self.stages[i - 1] if i > 0 else self.input_stage
            next_n = self.stages[i + 1].n if i < len(self.stages)-1 else 1

            if available_memory < self.dMs[i]:
                # if i!=0:
                #     print("Hello")
                # Not enough memory to initialise new process
                continue
            if prev.output_buffer < stage.n:
                # Not enough inputs available. Try next candidate
                continue
            if stage.output_buffer >= next_n:
                # Buffer already full
                continue

            # Initialise process
            prev.output_buffer -= stage.n
            stage.init(1)
            available_memory -= self.dMs[i]
            gap -= 1
            # Insert stage to be checked again
            insort(candidates, (stage, gap, i), key=lambda item: item[1])
            # # Only add more processes if quota not reached
            # if gap >= 0:
            #     insort(candidates, (stage, gap, i), key=lambda item: item[1])

    def step(self):
        # Schedule new processes
        self.init_processes()
        
        # Advance all stages by one local time unit
        for stage in self.stages:
            stage.step(self.rng)
        
        # Receive qubits
        self.input_stage.output_buffer += self.input_rate

        # Take care of final output
        self.output += self.stages[-1].output_buffer
        self.stages[-1].output_buffer = 0

    def run(self, steps: int, collect_data: bool=False, printing: bool=False) -> None | dict:
        # Efficient run without data collection
        if not collect_data:
            for _ in range(steps):
                self.step()
            return None
        
        # Otherwise collect data while running
        start = self.output
        ms = []
        for _ in range(steps):
            self.step()
            ms.append(self.memory_usage())
        if max(ms) > self.M:
            raise RuntimeError("Something went wrong.")
        mem = np.mean(ms)
        outputs = (self.output - start) * self.K
        rate = outputs / steps
        overhead = steps * self.input_rate / outputs if outputs > 0 else np.inf

        if printing:
            print("Input per output qubit (Overhead):", overhead)
            print("Output per time step:", rate)
            print("Mean memory consumption:", mem, "; (max, min) =", (max(ms), min(ms)))
        
        return {"rate": rate, "avg_memory": mem, "max_memory": max(ms)}
    
    def estimate_rate(self) -> float:
        logger.warning("Current implementation of `estimate_rate` can get loop-stuck.")
        elapsed_time = 0
        logger.info("Starting loop 1")
        while self.output < 100:
            self.step()
            elapsed_time += 1
        elapsed_time = 0
        start = self.output
        outputs = 0
        logger.info("Starting loop 2")
        while self.output < 1100:
            self.step()
            elapsed_time += 1
            outputs = (self.output - start) * self.K
        rate = outputs / elapsed_time
        return rate

