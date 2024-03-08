epochs=100
learning_rate=0.001
use_gpu=True


python -u main.py \
    --epochs $epochs \
    --lr $learning_rate \
    --device "cuda"> results/Debug/initial_mmnist_results.log
