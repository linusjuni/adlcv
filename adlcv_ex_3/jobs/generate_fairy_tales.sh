#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J generate_fairy_tales
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 00:30
#BSUB -o jobs/outputs/generate_fairy_tales_%J.out
#BSUB -e jobs/outputs/generate_fairy_tales_%J.err

cd ~/projects/adlcv
mkdir -p jobs/outputs

uv sync

PROMPT="Once upon a time, in a land far away, there lived"

uv run python -m adlcv_ex_3.test --strategy greedy --prompt "$PROMPT" --output /tmp/greedy.txt
uv run python -m adlcv_ex_3.test --strategy sampling --temperature 0.8 --prompt "$PROMPT" --output /tmp/sampling.txt

cat > fairy_tales.md << EOF
# Fairy Tales

**Prompt:** $PROMPT

---

## Greedy Decoding

$(cat /tmp/greedy.txt)

---

## Sampling (temperature=0.8)

$(cat /tmp/sampling.txt)
EOF

echo "Written to fairy_tales.md"
