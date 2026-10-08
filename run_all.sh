#!/bin/bash
cd "$(dirname "$0")"
for w in main ablation pareto robust sens; do python3 -W ignore exp_main.py $w > ./$w.log 2>&1; done
