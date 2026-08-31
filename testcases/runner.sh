#!/bin/bash
s=$1; i=$2
sc=$(/Users/ball103/DoSay/.venv/bin/python /Users/ball103/DoSay/algorithm/experiment/anneal.py \
    --version Anneal_V01b --seed "$s" --instance "$i" --iters 3000 --neighbor worst --quiet 2>/dev/null)
echo "$s $sc"
