import os
import pickle
import uuid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    TrainingArguments, Trainer, TrainerCallback, AutoConfig
)
from datasets import Dataset, DatasetDict, load_dataset

import wandb
import random
from utils_mine.utils import (
    read_dataset,
    diversevul_prepocessing_by_cwe_same_size,
    primevul_prepocessing_by_cwe_same_size,
    process_big_vul
)
from collections import Counter
from random import sample 
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

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Finetune bert large model.")
    parser.add_argument('--dataset_name', type=str, default="primevul", help='Name of the dataset bigvul, diversevul or primevul')
    parser.add_argument('--reduce', type=str, default="0", help='0 means dont reduce the model. 1 means reduce the model')
    parser.add_argument('--prune_layers', type=str, default="0", help='Number of layers to prune')
    parser.add_argument('--samples_per_class', type=str, default="5000", help='Number of samples per class')
    parser.add_argument('--random_pruning', type=str, default="0", help='Random pruning. 0 means no, 1 means yes')
    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    
    for rounds in [0]:
        
        initial_learning_rate = 2e-5
        warmup_steps = 1000
        batch_size = 16
        epochs = 10
        num_classes = 11
        max_tokens = 512
        model_name = 'bert-large-uncased' 
        architecture = 'bert-large'
        layers = 24
        reduced_layers = int(args.prune_layers)
        batch_size = 16
        reduce = int(args.reduce)
        number_of_samples = int(args.samples_per_class)
        seed = rounds
        dataset_to_read = args.dataset_name
        dataset = read_dataset(dataset_to_read)
        random_subset = int(args.random_pruning)
        
        if not "bigvul" in dataset_to_read:
            edataset = dataset.dropna()
        cwe_list = dataset['cwe'].tolist()
        if "primevul" in dataset_to_read:
            string_cwe ="CWE-119 CWE-20 CWE-416 CWE-362 CWE-476 CWE-190 CWE-787 CWE-125 CWE-79 CWE-78"
        elif "diverse" in dataset_to_read:
            string_cwe ="CWE-416 CWE-125 CWE-20 CWE-119 CWE-787 CWE-476 CWE-190 CWE-362 CWE-22 CWE-78"
        elif "bigvul" in dataset_to_read:
            string_cwe ="CWE-119 CWE-20 CWE-416 CWE-125 CWE-362 CWE-476 CWE-190 CWE-787 CWE-79 CWE-269"
        cwe_list = string_cwe.split()

        if dataset_to_read == "primevul":
            train_data, val_data, y_train, y_val, label_to_encoded, label_encoder = primevul_prepocessing_by_cwe_same_size(
                dataset, "func", max_tokens, cwe_list, 10, number_of_samples, rounds
            )
        elif dataset_to_read == "diversevul":
            train_data, val_data, y_train, y_val, label_to_encoded, label_encoder = diversevul_prepocessing_by_cwe_same_size(
            dataset, "func", max_tokens, cwe_list, 10, number_of_samples, rounds
        )

        elif dataset_to_read == "bigvul":
            train_data, val_data, y_train, y_val, label_to_encoded, label_encoder = process_big_vul(1,512,cwe_list,number_of_samples)

        list_codes_train = train_data["func"].tolist()
        list_codes_val = val_data["func"].tolist()

        train_dataset = Dataset.from_dict({"func": list_codes_train, "label": y_train})
        test_dataset = Dataset.from_dict({"func": list_codes_val, "label": y_val})
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        tokenizer = AutoTokenizer.from_pretrained(model_name)                    
        processed_dataset = dataset_dict.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)

        if reduce:
            classifier = model.classifier
            config.num_hidden_layers = reduced_layers
            if random_subset:
                
                random_indices = random.sample(range(len(model.bert.encoder.layer)), reduced_layers)
                random_indices.sort()
                random_layers = torch.nn.ModuleList([model.bert.encoder.layer[i] for i in random_indices])
                model.bert.encoder.layer = random_layers
            else:
                print(reduced_layers)
                model.bert.encoder.layer = model.bert.encoder.layer[:reduced_layers]
            
            model.classifier = classifier
            
            print("Final model")
            print(model)
            


        directory_name = (
        f"{model_name.replace('/','')}_lr-{initial_learning_rate}_epochs-{epochs}_batch-{batch_size}_"
        f"classes-{num_classes}_samples-{number_of_samples}_maxtokens-{max_tokens}_"
        f"layers-{layers}_reduced-reducedprune-{reduced_layers}_cwe_id-{unique_id}_{rounds}_{dataset_to_read}_")
        
        name_save = (
            f"./runs/{dataset_to_read}/{model_name}/{directory_name}"
        )

        training_args = TrainingArguments(
            output_dir=name_save,
            learning_rate=initial_learning_rate,
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            metric_for_best_model="accuracy",
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=500,
            report_to="none", 
            gradient_accumulation_steps=8, 
        )
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,  
            callbacks=[PrintAccuracyCallback()],  
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f"./{name_save}/confusion_matrix.png")  # Save the figure as a PNG file

        # Compute and print classification report
        report = classification_report(true_labels, predicted_labels)
        print("Classification Report:")
        print(report)

        # Compute and print F1 score
        f1 = f1_score(true_labels, predicted_labels, average="weighted")
        print("Weighted F1 Score:", f1)
        with open(f"./{name_save}/true_labels.pkl", "wb") as file:
            pickle.dump(true_labels, file)
        with open(f"./{name_save}/predicted_labels.pkl", "wb") as file:
            pickle.dump(predicted_labels, file)
        with open(f"./{name_save}/report.pkl", "wb") as file:
            pickle.dump(report, file)
        with open(f"./{name_save}/test_samples.pkl", "wb") as file:
            pickle.dump(test_dataset, file)



        
