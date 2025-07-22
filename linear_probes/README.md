
Compute linear probes per layer for bert model:

```bash
CUDA_VISIBLE_DEVICES=0 python probing_bert.py --dataset_name diversevul --characteristic cyclomatic_complexity --executions 1
```

For --characteristic you can use either cyclomatic_complexity or halstead_difficulty