

To execute, use the following commands:
where :

--dataset_name: Name of the dataset (default: "primevul").
--reduce: Flag to reduce the model (default: "0").
--prune_layers: Number of layers to prune (default: "0").
--samples_per_class: Number of samples per class (default: "5000").
--random_pruning: Flag for random pruning (default: "0").

```bash
python bert_finetuning.py --dataset_name primevul --reduce 1 --prune_layers 15 --samples_per_class 5000 --random_pruning 0
```


```bash
python gemma_finetuning.py --dataset_name primevul --reduce 1 --prune_layers 5 --samples_per_class 5000 --random_pruning 0
```