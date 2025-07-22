import matplotlib.pyplot as plt
import torch
from colorama import Fore, Style
from utils_probes import *
import numpy as np

import matplotlib.pyplot as plt
import gzip
import pickle
import argparse
print(torch.cuda.is_available())
device = torch.device("cuda")



def train_linear_probes(
    num_epochs,
    train_representations,
    train_labels,
    device,
    classifier,
    criterion,
    optimizer,
    batch_size=128,
):
    num_total = len(train_representations)

    tensors = [torch.tensor(inner_list,dtype=torch.float32) for inner_list in train_representations]
    train_representations = torch.stack(tensors).to(device)
    train_labels_tensor = torch.tensor(train_labels).to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_correct = 0.0
        for batch in range(0, num_total, batch_size):
            end = batch + batch_size
            batch_repr = train_representations[batch:end]

            
            batch_labels = train_labels_tensor[batch:end]


            optimizer.zero_grad()

            out = classifier(batch_repr)
            pred = out.max(1)[1]
            equal_values = pred.eq(batch_labels).sum().item()
            num_correct += equal_values

            loss = criterion(out, batch_labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()


        # Optionally print or log epoch-wise average loss and accuracy
        #print(f'Epoch {epoch+1}: Loss: {total_loss / num_total:.4f}, Accuracy: {num_correct / num_total:.4f}')


    return total_loss / num_total, num_correct / num_total

def evaluate_linear_probes(
    test_representations, test_labels, device, classifier, criterion, batch_size=128
):
    # Convert list of lists to tensor and move to the specified device at once
    test_representations = torch.stack([torch.tensor(inner_list,dtype=torch.float32) for inner_list in test_representations]).to(device)
    test_labels_tensor = torch.tensor(test_labels).to(device)
    classifier.to(device)

    num_correct = 0.0
    num_total = len(test_labels)
    total_loss = 0.0

    with torch.no_grad():
        for batch in range(0, num_total, batch_size):
            end = batch + batch_size
            batch_repr = test_representations[batch:end]
            batch_labels = test_labels_tensor[batch:end]

            out = classifier(batch_repr)
            pred = out.max(1)[1]

            num_correct += pred.eq(batch_labels).sum().item()

            loss = criterion(out, batch_labels)
            total_loss += loss.item() * len(batch_labels)  # Weight the loss by the batch size

    # Normalize the total loss and calculate accuracy
    return total_loss / num_total, num_correct / num_total


def process_extract_representations(sentences):
    all_sentences_last_representations = []
    for sentence in sentences:
    
            input_ids = tokenizer.encode(
                sentence, add_special_tokens=False, return_tensors="pt"
            ).to(device)
     

            with torch.no_grad():
                outputs = model(input_ids)
                hidden_states = outputs.hidden_states
            last_representations = [
                layer_hidden_state[0, -1].cpu().numpy()
                for layer_hidden_state in hidden_states[0:]
            ]

            all_sentences_last_representations.append(last_representations)
 
    all_sentences_last_representations = [
        np.stack(reps) for reps in all_sentences_last_representations
    ]
    return all_sentences_last_representations



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some dataset.")
    parser.add_argument('--dataset_name', type=str, default="primevul", help='Name of the dataset bigvul, diversevul or primevul')
    parser.add_argument('--characteristic', type=str, default="cyclomatic_complexity", help='cyclomatic_complexity or halstead_difficulty')
    parser.add_argument('--executions', type=str, default="1", help='Number of executions')
    args = parser.parse_args()

    dataset_name = args.dataset_name


    if 'primevul' in dataset_name:
        cwe_list = ["CWE-476", "CWE-416","CWE-20"," ","CWE-787","CWE-125","CWE-79","CWE-119","CWE-190","CWE-78","CWE-362"]
    elif 'diverse' in dataset_name:
        cwe_list = ["[]","CWE-190", "CWE-119", "CWE-416", "CWE-78", "CWE-22", "CWE-787", "CWE-476", "CWE-20", "CWE-362", "CWE-125"]
    elif 'bigvul' in dataset_name:
        cwe_list = [" ","CWE-476","CWE-362","CWE-416","CWE-20","CWE-787","CWE-269","CWE-125","CWE-79","CWE-119","CWE-190"]
    else:
        print("Wrong dataset")
        exit()

    total_time_per_layer_lp = []

    for cw in cwe_list:
        
            quantization = 1
            cwe = cw
            print(f"Computing for CWE: {cw}")
            model_name = "bert-large-uncased"
            charasteristic = args.characteristic

            train_set, test_set = pre_process(cwe, charasteristic, dataset_name)
            X_train_orig = train_set["Code"].tolist()
        

            y_train = train_set[charasteristic].tolist()
            X_test_orig = test_set["Code"].tolist()
            y_test = test_set[charasteristic].tolist()
            num_labels = len(set(y_train))
            
     

            if 'halstead_difficulty' in charasteristic:
                y_train = np.array(y_train) 
                y_test = np.array(y_test) 
            else:
                y_train = np.array(y_train) - 1
                y_test = np.array(y_test) - 1

            model, tokenizer, sep, emb_dim = get_model_and_tokenizer(
                model_name, device, random_weights=False, quantization=0
            )
            model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True,
            ).to(0)


            classifier, criterion, optimizer = build_classifier(emb_dim, num_labels, device)

            plot_all_graphics = []
        
            for trial_num in range(int(args.executions)):
                
                X_train = process_extract_representations(X_train_orig)
                X_test = process_extract_representations(X_test_orig)
              

                X_train = np.array(X_train)
                X_test = np.array(X_test)
                num_layers = len(X_train[0])
                emb_dim = len(X_train[0][0])
                num_epochs = 100

                unique_elements_train, counts_train = np.unique(y_train, return_counts=True)
                unique_elements_test, counts_test = np.unique(y_test, return_counts=True)

                _, counts_train = np.unique(y_train, return_counts=True)
                min_class_size = counts_train.min()

                classes = np.unique(y_train)
                X_train_balanced = []
                y_train_balanced = []

                for cls in classes:
                    
                    class_indices = np.where(y_train == cls)[0]

                    selected_indices = np.random.choice(
                        class_indices, min_class_size, replace=False
                    )
      
                    X_train_balanced.append(X_train[selected_indices])
                    y_train_balanced.append(y_train[selected_indices])

                # Convert lists to numpy arrays
                X_train_balanced = np.vstack(X_train_balanced)
                y_train_balanced = np.concatenate(y_train_balanced)

                _, counts_test = np.unique(y_test, return_counts=True)
                min_class_size_test = counts_test.min()

                classes_test = np.unique(y_test)
                X_test_balanced = []
                y_test_balanced = []

                for cls in classes_test:
                    class_indices_test = np.where(y_test == cls)[0]

                    selected_indices_test = np.random.choice(
                        class_indices_test, min_class_size_test, replace=False
                    )

                    X_test_balanced.append(X_test[selected_indices_test])
                    y_test_balanced.append(y_test[selected_indices_test])

                X_test_balanced = np.vstack(X_test_balanced)
                y_test_balanced = np.concatenate(y_test_balanced)
                unique_elements_train, counts_train = np.unique(
                    y_train_balanced, return_counts=True
                )
                unique_elements_test, counts_test = np.unique(
                    y_test_balanced, return_counts=True
                )

                num_trials = 1
                num_layers = len(
                    X_train_balanced[0]
                )  # Assuming the layer dimension is the second one
                per_layer_acc = np.zeros(
                    (num_trials, num_layers)
                )  # To store accuracy for each trial and layer

                for trial in [0]:
                    for l in range(num_layers):
                        print(f"Layer {l}")

                        X_train_layer = X_train_balanced[:, l, :]
                        X_test_layer = X_test_balanced[:, l, :]

                        # Build the classifier
                        classifier, criterion, optimizer = build_classifier(
                            emb_dim, num_labels, device
                        )

                        # Train the classifier
                        train_loss, train_accuracy = train_linear_probes(
                            100,
                            X_train_layer,
                            y_train_balanced,
                            device,
                            classifier,
                            criterion,
                            optimizer,
                            batch_size=512,
                        )
                    

                        # Evaluate the classifier
                        test_loss, test_accuracy = evaluate_linear_probes(
                            X_test_layer,
                            y_test_balanced,
                            device,
                            classifier,
                            criterion,
                            batch_size=512,
                        )
                    
                        # Store the test accuracy
                        per_layer_acc[trial, l] = test_accuracy
        


 

                avg_per_layer_acc = per_layer_acc.mean(axis=0)

                plot_all_graphics.append(avg_per_layer_acc)

            avg_per_layer_acc = np.array(plot_all_graphics).mean(axis=0)

            print(f"Accuracy per layer for CWE {cw} {avg_per_layer_acc}" )

    
            



