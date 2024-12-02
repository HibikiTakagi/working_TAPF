#!/bin/bash

#kernprof -l -v -o ./dataR/dataRA/perf/testperftime.lprof testing_script.py 
#kernprof -l -v -o ./dataR/dataRA/perf/testperftime.lprof savetraining.py 
kernprof -l -v -o ./dataR/dataRA/perf/testperftime.lprof cbs_script.py
python3.10 -m line_profiler ./dataR/dataRA/perf/testperftime.lprof > ./dataR/dataRA/perf/testperftime100.txt
#python3 -m kernprof -l -v -o ./dataR/dataRA/perf/testperftime.lprof swarm_tello_testing.py
#python3 -m line_profiler ./dataR/dataRA/perf/testperftime.lprof > ./dataR/dataRA/perf/testperftime100.txt