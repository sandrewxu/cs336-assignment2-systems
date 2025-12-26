mkdir -p profiles

# for CONTEXT in 128 256 512 1024; do
#     for MODEL in small medium large xl 2.7B; do
#         nsys profile --trace=cuda,nvtx \
#         --python-backtrace=cuda \
#         -o profiles/${MODEL}_${CONTEXT} \
#         python cs336_systems/benchmark.py --model $MODEL \
#         --context-length $CONTEXT
#     done
# done

echo "=== FORWARD ONLY (compile) ==="
python cs336_systems/benchmark.py --benchmark-steps 5 --forward-only --compile
echo "\n\n===FORWARD ONLY (no compile) ==="
python cs336_systems/benchmark.py --benchmark-steps 5 --forward-only
echo "\n\n=== FULL STEP (compile) ==="
python cs336_systems/benchmark.py --benchmark-steps 5 --compile
echo "\n\n=== FULL STEP (no compile) ==="
python cs336_systems/benchmark.py --benchmark-steps 5
