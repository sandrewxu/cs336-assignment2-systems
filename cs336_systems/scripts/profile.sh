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

for CONTEXT in 256; do
    for MODEL in small medium large xl 2.7B; do
        nsys profile --trace=cuda,nvtx \
        --python-backtrace=cuda \
        -o profiles/mixed_precision/${MODEL}_${CONTEXT}_mixed_precision \
        python cs336_systems/benchmark.py --model $MODEL \
        --context-length $CONTEXT \
        --benchmark-steps 1 \
        --mixed-precision
    done
done
