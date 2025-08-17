srun --gres=gpu:nvidia:2 --cpus-per-task=16 --mem=256G python maker.py
srun --gres=gpu:nvidia:2 --cpus-per-task=16 --mem=256G python tester.py