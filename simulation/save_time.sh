#!/bin/bash
#(time python3.10 savetraining.py) 2> ./dataR/dataRA/savetime.txt
#(time python3.10 cbs_script.py) 2> ./dataR/dataRA/savetime.txt
#(time python3.10 beam_script.py) 2> ./dataR/dataRA/savetime.txt
(time python3.10 cache_beam_script.py) 2> ./dataR/dataRA/savetime.txt
#(time python3.10 cache_cbs_script.py) 2> ./dataR/dataRA/savetime.txt