#!/bin/sh
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J ddpm_eval
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 02:00
#BSUB -o adlcv_ex_5/jobs/outputs/ddpm_eval_%J.out
#BSUB -e adlcv_ex_5/jobs/outputs/ddpm_eval_%J.err

cd ~/projects/adlcv

uv sync

uv run -m adlcv_ex_5.ddpm_eval
