import os
import pickle
import uuid

# Third-party imports for data handling and machine learning
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import random
from transformers import TrainingArguments, EarlyStoppingCallback
import trl
import torch
import argparse
# Hugging Face's Transformers and datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import Dataset, DatasetDict, load_dataset

import wandb

from utils_mine.utils import (
    read_dataset,
    diversevul_prepocessing_by_cwe_same_size,
    primevul_prepocessing_by_cwe_same_size,
    process_big_vul
)

unique_id = str(uuid.uuid4())


checkpoint_dir = "./runs/"
os.makedirs(checkpoint_dir, exist_ok=True)



class PrintAccuracyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if "eval_accuracy" in kwargs["metrics"]:
            print(
                f"Accuracy after epoch {state.epoch}: {kwargs['metrics']['eval_accuracy']:.4f}"
            )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def preprocess_function(examples):
    return tokenizer(examples["func"], truncation=True)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Finetune gemma 2b model.")
    parser.add_argument('--dataset_name', type=str, default="primevul", help='Name of the dataset bigvul, diversevul or primevul')
    parser.add_argument('--reduce', type=str, default="0", help='0 means dont reduce the model. 1 means reduce the model')
    parser.add_argument('--prune_layers', type=str, default="0", help='Number of layers to prune')
    parser.add_argument('--samples_per_class', type=str, default="5000", help='Number of samples per class')
    parser.add_argument('--random_pruning', type=str, default="0", help='Random pruning. 0 means no, 1 means yes')
    parser.add_argument('--rank', type=str, default="256", help='set rank for GALORE')

    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    for ronda in [4,5,6]:

        initial_learning_rate = 5e-5
        warmup_steps = 1000
        batch_size = 16
        epochs = 10
        num_classes = 11
        random_pruning = int(args.random_pruning)
        max_tokens = 1024
        number_of_samples = int(args.samples_per_class)
        model_name = (
            "google/gemma-2b"
        )
        architecture = "gemma"
        layers = 18
        reduced_layers = int(args.prune_layers)
        reduce = int(args.reduce)
        rank = int(args.rank)

        dataset_to_read = args.dataset_name

        dataset = read_dataset(dataset_to_read)

            
        dataset = dataset.dropna()
        if "primevul" in dataset_to_read:
            string_cwe ="CWE-119 CWE-20 CWE-416 CWE-362 CWE-476 CWE-190 CWE-787 CWE-125 CWE-79 CWE-78"
        elif "diverse" in dataset_to_read:
            string_cwe ="CWE-416 CWE-125 CWE-20 CWE-119 CWE-787 CWE-476 CWE-190 CWE-362 CWE-22 CWE-78"
        elif "bigvul" in dataset_to_read:
            string_cwe ="CWE-119 CWE-20 CWE-416 CWE-125 CWE-362 CWE-476 CWE-190 CWE-787 CWE-79 CWE-269"
                
        cwe_list = string_cwe.split()
        
        if dataset_to_read == "primevul":
            train_data, val_data, y_train, y_val, label_to_encoded, label_encoder = primevul_prepocessing_by_cwe_same_size(
                    dataset, "func", max_tokens, cwe_list, 10, number_of_samples, ronda
                )
        elif dataset_to_read == "diversevul":

            train_data, val_data, y_train, y_val, label_to_encoded, label_encoder = diversevul_prepocessing_by_cwe_same_size(
                dataset, "func", max_tokens, cwe_list, 10,number_of_samples, ronda
            )

        elif dataset_to_read == "bigvul":
                train_data, val_data, y_train, y_val, label_to_encoded, label_encoder = process_big_vul(ronda,1024,cwe_list,number_of_samples)

        

        list_codes_train = train_data["func"].tolist()

        list_codes_val = val_data["func"].tolist()

      

        train_dataset = Dataset.from_dict(
            {"func": list_codes_train, "label": y_train}
        )

        test_dataset = Dataset.from_dict({"func": list_codes_val, "label": y_val})
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})


        tokenizer = AutoTokenizer.from_pretrained(
            model_name, max_seq_length=max_tokens
        )

        processed_dataset = dataset_dict.map(
            preprocess_function, batched=True
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes,torch_dtype=torch.float16
        ).to(0)

        print("---- initial ----")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        non_trainable_params = total_params - trainable_params

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")

        if reduce:
            if "gemma" in model_name:
                if random_pruning:
                    random_indices = random.sample(range(len(model.model.layers)), reduced_layers)
                    random_indices.sort()
                    random_layers = torch.nn.ModuleList([model.model.layers[i] for i in random_indices])
                    model.model.layers = random_layers
                    model.config.update({'num_hidden_layers': len(random_indices)})
                    model.config.use_cache = False  
                else:
                    new_layers = torch.nn.ModuleList(
                        list(model.model.layers.children())[:reduced_layers]
                    )

                    model.model.layers = new_layers
                    model.config.update({'num_hidden_layers': len(random_indices)})
                   
            print(model)

        if rank == 1024:
            scale = 2
        else:
            scale = 0.1

        directory_name = (
            f"{model_name.replace('/','')}_lr-{initial_learning_rate}_epochs-{epochs}_batch-{batch_size}_"
            f"classes-{num_classes}_samples-{number_of_samples}_maxtokens-{max_tokens}_"
            f"layers-galore-{layers}_reduced-{reduced_layers}_cwe_id_{ronda}_primevul_real_{rank}_rank_{dataset_to_read}_{scale}"
        )

        name_save = f"./runs/{dataset_to_read}/{directory_name}"

        training_args = TrainingArguments(
            output_dir=name_save,
            
            learning_rate=initial_learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            bf16=True,
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=500,
            report_to="wandb",
            metric_for_best_model="accuracy",
            gradient_checkpointing = True,
            optim = "galore_adamw_8bit_layerwise",
        optim_target_modules = ["attn", "mlp"],
        optim_args=f"rank={rank}, update_proj_gap=100, scale={scale}",
            
        )


        trainer = trl.SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["test"],
            tokenizer=tokenizer,
            dataset_text_field="text",
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            max_seq_length=1024,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3),PrintAccuracyCallback()],
        )

        trainer.train()

        test_dataset = processed_dataset["test"]
        eval_results = trainer.evaluate(test_dataset)

        # Print evaluation metrics
        print("Evaluation Metrics:")
        print(eval_results)
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = test_dataset["label"]

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix:")
        print(cm)

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(
            f"{name_save}/confusion_matrix.png"
        )  # Save the figure as a PNG file

        # Compute and print classification report
        report = classification_report(true_labels, predicted_labels)
        print("Classification Report:")
        print(report)

        # Compute and print F1 score
        f1 = f1_score(true_labels, predicted_labels, average="weighted")
        print("Weighted F1 Score:", f1)
        with open(f"{name_save}/true_labels.pkl", "wb") as file:
            pickle.dump(true_labels, file)
        with open(f"{name_save}/test_samples.pkl", "wb") as file:
            pickle.dump(test_dataset, file)

        
