#!/bin/bash

mkdir -p outputs

echo "FKEM"
python benchmarker_cut_bins.py confs/config_fkem.yml FKEM 200 full 2 7
python benchmarker_cut_bins.py confs/config_fkem.yml FKEM 200 half 2 7
python benchmarker_cut_bins.py confs/config_fkem.yml FKEM 200 quarter 2 7

echo "MATTER"
python benchmarker_cut_bins.py confs/config_matter.yml MATTER 200 full 2 7
python benchmarker_cut_bins.py confs/config_matter.yml MATTER 200 half 2 7
python benchmarker_cut_bins.py confs/config_matter.yml MATTER 200 quarter 2 7

echo "Levin"
python benchmarker_cut_bins.py confs/config_levin.yml Levin 200 full 2 7
python benchmarker_cut_bins.py confs/config_levin.yml Levin 200 half 2 7
python benchmarker_cut_bins.py confs/config_levin.yml Levin 200 quarter 2 7
