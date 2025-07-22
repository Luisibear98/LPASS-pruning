
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset, concatenate_datasets
import wandb
from datasets import Dataset
import evaluate
from sklearn.preprocessing import LabelEncoder
import math
from collections import Counter


def read_dataset(dataset):

    if dataset == 'diversevul':
        csv_name = '../data/diversevul/diversevul_output.csv'
    elif dataset == 'primevul':
        csv_name = '../data/processed/bigvul_metadata.csv'
    elif dataset == 'bigvul':
        csv_name = '../data/big-vul/big-vul_metadata.csv'
    
    return pd.read_csv(csv_name)


def compute_len_funcs(df, key):
    functions = df[key].tolist()
    len_functions = []
    for fun in functions:

        try:
            len_functions.append(len(fun))
        except:
            len_functions.append(0)
    df['len_func'] = len_functions
    return df

def filter_dataset(df, max_tokens):
    df_filtered = df[df['len_func'] <= max_tokens]
    df_filtered = df_filtered[df_filtered['len_func'] > 0]
    return df_filtered

def split_diversevul(df):
    cwe_list = df['cwe'].tolist()
    cwe_or_not = []
    for cwe in cwe_list:
        try:
            if 'CWE' in cwe:
                cwe_or_not.append(1)
            else: 
                cwe_or_not.append(0)
        except:

            cwe_or_not.append(0)

    df['cwe_presence'] = cwe_or_not
    df_with_cwe = df[df['cwe_presence'] == 1]
    df_without_cwe = df[df['cwe_presence'] == 0]

    return df_with_cwe, df_without_cwe


def split_data(dataset, target_column='cwe_presence', train_ratio=0.8, val_ratio=0.15, test_ratio=0.5, random_state=1):

    # Separate the DataFrame into two based on the target column
    positive_class_data = dataset[dataset[target_column] == 1]
    negative_class_data = dataset[dataset[target_column] == 0]

    # Shuffle each subset independently
    positive_class_data = positive_class_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    negative_class_data = negative_class_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split each subset into training, validation, and testing sets
    pos_train, pos_temp = train_test_split(positive_class_data, test_size=(1 - train_ratio), random_state=random_state)
    pos_val, pos_test = train_test_split(pos_temp, test_size=(test_ratio / (test_ratio + val_ratio)) if (test_ratio + val_ratio) > 0 else 0, random_state=random_state)

    neg_train, neg_temp = train_test_split(negative_class_data, test_size=(1 - train_ratio), random_state=random_state)
    neg_val, neg_test = train_test_split(neg_temp, test_size=(test_ratio / (test_ratio + val_ratio)) if (test_ratio + val_ratio) > 0 else 0, random_state=random_state)

    # Concatenate the splits to form the final sets
    train_data = pd.concat([pos_train, neg_train], ignore_index=True)
    val_data = pd.concat([pos_val, neg_val], ignore_index=True)
    test_data = pd.concat([pos_test, neg_test], ignore_index=True)

    # Shuffle final sets
    train_data = train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_data, val_data, test_data






def balanced_splitted(dataframe):

    X = dataframe.drop('real_label', axis=1)  # Features
    y = dataframe['real_label']  # Labels
    label_to_encoded = {}
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    for label, encoded_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"Original Label: '{label}' is encoded as {encoded_label}")

        label_to_encoded[label] = encoded_label
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
    return X_train, X_val, y_train, y_val,label_to_encoded, label_encoder



def process_big_vul(seed,max_len, cwe_list,to_get):


    dataset = load_dataset("benjis/bigvul")
    filtered_dataset = dataset.filter(lambda example: len(example['func_after']) <= max_len)
    whole_dataset =  concatenate_datasets([filtered_dataset['train'], filtered_dataset['validation'], filtered_dataset['test']]) #concatenate_datasets([filtered_dataset['train'], filtered_dataset['validation'], filtered_dataset['test']])
    df = pd.DataFrame(whole_dataset)
    df_none = df[df['CWE ID'].isnull()]

    df_vul = df[df['CWE ID'].notna()]

    vulnerable_with_cwe_code = df_vul['func_after'].tolist()
    vulnerable_with_cwe_id = df_vul['CWE ID'].tolist()
    
    no_vulnerable_with_cwe_code = df_none['func_after'].tolist()
    novulnerable_with_no_cwe_id = df_none['CWE ID'].tolist()

    vulnerable_with_cwe_code_filtered = []
    vulnerable_with_cwe_id_filtered = []

    counter = 0
    sample_to_get = to_get
    seed = seed
    vulnerable_samples = pd.DataFrame()
    for cw in cwe_list:
        partial_cw = df_vul[df_vul['CWE ID'] == cw]
        max_len_of_label = len(partial_cw)
        print(f'{cw} there are: {max_len_of_label}')
        if max_len_of_label >= sample_to_get:
            filtered_by_cwe_first = partial_cw.sample(n = sample_to_get, random_state = seed ) 
        else:
            filtered_by_cwe_first = partial_cw.sample(n = max_len_of_label, random_state = seed ) 
        vulnerable_samples = pd.concat([vulnerable_samples,filtered_by_cwe_first], ignore_index=True)
    
    
    dataset_without_cwe = df_none.sample(n = sample_to_get, random_state = seed ) 
    list_codes_train = dataset_without_cwe["func_after"].tolist()
    list_codes_val = vulnerable_samples["func_after"].tolist()


    dataset = pd.concat([vulnerable_samples,dataset_without_cwe], ignore_index=True)

    dataset_shuffled = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    dataset_shuffled = dataset_shuffled.drop_duplicates()
    X = dataset_shuffled.drop('func_before', axis=1)  # Features
    y = dataset_shuffled['CWE ID']  # Labels
    label_to_encoded = {}
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    threshold = sample_to_get
    for label, encoded_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"Original Label: '{label}' is encoded as {encoded_label}")

        label_to_encoded[label] = encoded_label
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    list_codes_train = X_train["func_after"].tolist()
    list_codes_val = X_val["func_after"].tolist()

    
    y_train_array = np.array(y_train) 
    unique, counts = np.unique(y_train_array, return_counts=True)
    class_counts = dict(zip(unique, counts))


    undersampled_classes = {class_label: count for class_label, count in class_counts.items() if count < threshold}
    oversampled_X_train = X_train.copy()
    oversampled_y_train = list(y_train) 
    for class_label, count in undersampled_classes.items():

        class_indices = np.where(y_train_array == class_label)[0]        
        samples_to_add = threshold - count
        samples_indices_to_duplicate = np.random.choice(class_indices, size=samples_to_add, replace=True)

        oversampled_X_train = pd.concat([oversampled_X_train, X_train.iloc[samples_indices_to_duplicate]], ignore_index=True)
        oversampled_y_train.extend(y_train_array[samples_indices_to_duplicate].tolist())


    shuffled_indices = np.random.permutation(len(oversampled_y_train))
    X_train = oversampled_X_train.iloc[shuffled_indices]
    y_train = [oversampled_y_train[i] for i in shuffled_indices]
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)

    return X_train, X_val, y_train, y_val, label_to_encoded, label_encoder



def primevul_prepocessing_by_cwe_same_size(dataset, key_db, max_tokens, cwe_list, top_to_keep, number_per_class, seed):
    dataset = compute_len_funcs(dataset, key_db)

    dataset = filter_dataset(dataset, max_tokens)

    dataset_with_cwe, dataset_without_cwe = split_diversevul(dataset)
 
    balance = 1
    list_new_labels = []

    dict_target_cwe = {}

    for cwe in cwe_list:
        dict_target_cwe[cwe] = 0

    list_present_cwe = dataset_with_cwe['cwe'].tolist()
    for cwe in list_present_cwe:
        real_cwe = cwe.replace("'",'').replace('[','').replace(']','')
        real_cwe_splitted = real_cwe.split(',')
        if len(real_cwe_splitted) > 1:
            list_new_labels.append(real_cwe_splitted[0])
            if real_cwe_splitted[0] in cwe_list:
                dict_target_cwe[real_cwe_splitted[0]] = dict_target_cwe[real_cwe_splitted[0]] + 1

        else:
            list_new_labels.append(real_cwe)
            if real_cwe in cwe_list:
                dict_target_cwe[real_cwe] = dict_target_cwe[real_cwe] + 1
    
    sorted_dict = dict(sorted(dict_target_cwe.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)

    keys_to_keep = []
    count_to_keep = []
    
    for key, value in sorted_dict.items():

        keys_to_keep.append(key)
        count_to_keep.append(value)

    dataset_with_cwe['real_label'] = list_new_labels
    candidates = keys_to_keep[:top_to_keep]
    get_samples = 0
    repeat_to_fill = 0

    if 'undersampling' in balance_mode:
        get_samples = count_to_keep[:top_to_keep][-1]

    elif 'oversampling' in balance_mode:
        get_samples = count_to_keep[:top_to_keep][0]
        repeat_to_fill = 0


    dataset_with_cwe = dataset_with_cwe[dataset_with_cwe['real_label'].isin(candidates)]
    dataset_without_cwe['real_label'] = ['None'] * len(dataset_without_cwe)

    if get_samples == 0:
        size_of_cwe = math.floor(len(dataset_with_cwe)/top_to_keep)
    else:
        size_of_cwe = get_samples
        
    vulnerable_samples = pd.DataFrame()
    for cwe in candidates:
    
        filtered_by_cwe = dataset_with_cwe[dataset_with_cwe['real_label'] == cwe]
        max_len_of_label = len(filtered_by_cwe)

        if max_len_of_label < number_per_class:
            additional_to_extract = number_per_class - max_len_of_label
         
            filtered_by_cwe_first = filtered_by_cwe.sample(n = max_len_of_label, random_state = seed )

        else:
            filtered_by_cwe_first = filtered_by_cwe.sample(n = number_per_class, random_state = seed ) 

        vulnerable_samples = pd.concat([vulnerable_samples,filtered_by_cwe_first], ignore_index=True)
    

    if len(dataset_without_cwe) < number_per_class:
        dataset_without_cwe = dataset_without_cwe.sample(n = len(dataset_without_cwe), random_state = seed ) 
    else:
        dataset_without_cwe = dataset_without_cwe.sample(n = number_per_class, random_state = seed ) 
    dataset = pd.concat([vulnerable_samples,dataset_without_cwe], ignore_index=True)


    dataset_shuffled = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    X_train, X_val, y_train, y_val, label_to_encoded, label_encoder = balanced_splitted(dataset_shuffled)

    threshold = number_per_class  # Set the threshold for the minimum number of samples per class

    # Convert y_train to a numpy array if it's not already, to use np.unique
    y_train_array = np.array(y_train)  # Assuming y_train is a list initially
    unique, counts = np.unique(y_train_array, return_counts=True)
    class_counts = dict(zip(unique, counts))


    # Find classes with fewer than the threshold number of samples
    undersampled_classes = {class_label: count for class_label, count in class_counts.items() if count < threshold}

    # Create a copy of X_train and y_train for oversampling
    oversampled_X_train = X_train.copy()
    oversampled_y_train = list(y_train)  # Convert to list if it's not already

    for class_label, count in undersampled_classes.items():
        # Find indices of the current undersampled class
        class_indices = np.where(y_train_array == class_label)[0]
        
        # Calculate the number of samples to replicate
        samples_to_add = threshold - count
        
        # Randomly choose samples to duplicate
        samples_indices_to_duplicate = np.random.choice(class_indices, size=samples_to_add, replace=True)
        
        # Append duplicated samples to the training data
        oversampled_X_train = pd.concat([oversampled_X_train, X_train.iloc[samples_indices_to_duplicate]], ignore_index=True)
        oversampled_y_train.extend(y_train_array[samples_indices_to_duplicate].tolist())

    # Optionally shuffle the dataset to mix the oversampled data
    shuffled_indices = np.random.permutation(len(oversampled_y_train))
    X_train = oversampled_X_train.iloc[shuffled_indices]
    y_train = [oversampled_y_train[i] for i in shuffled_indices]

    # Count the occurrences of each class in the training and validation sets
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)

    return X_train, X_val, y_train, y_val, label_to_encoded, label_encoder


def diversevul_prepocessing_by_cwe_same_size(dataset, key_db, max_tokens, cwe_list, top_to_keep, number_per_class, seed):
    dataset = compute_len_funcs(dataset, key_db)
    dataset = filter_dataset(dataset, max_tokens)
    dataset_with_cwe, dataset_without_cwe = split_diversevul(dataset)
    balance = 1
    list_new_labels = []

    dict_target_cwe = {}

    for cwe in cwe_list:
        dict_target_cwe[cwe] = 0

    list_present_cwe = dataset_with_cwe['cwe'].tolist()

    for cwe in list_present_cwe:
        real_cwe = cwe.replace("'",'').replace('[','').replace(']','')
        real_cwe_splitted = real_cwe.split(',')
        if len(real_cwe_splitted) > 1:
            list_new_labels.append(real_cwe_splitted[0])
            if real_cwe_splitted[0] in cwe_list:
                dict_target_cwe[real_cwe_splitted[0]] = dict_target_cwe[real_cwe_splitted[0]] + 1

        else:
            list_new_labels.append(real_cwe)
            if real_cwe in cwe_list:
                dict_target_cwe[real_cwe] = dict_target_cwe[real_cwe] + 1

    sorted_dict = dict(sorted(dict_target_cwe.items(), key=lambda item: item[1], reverse=True))

    keys_to_keep = []
    count_to_keep = []
    
    for key, value in sorted_dict.items():

        keys_to_keep.append(key)
        count_to_keep.append(value)

    dataset_with_cwe['real_label'] = list_new_labels
    candidates = keys_to_keep[:top_to_keep]
    get_samples = 0
    repeat_to_fill = 0


   

    dataset_with_cwe = dataset_with_cwe[dataset_with_cwe['real_label'].isin(candidates)]
    dataset_without_cwe['real_label'] = ['None'] * len(dataset_without_cwe)
  

        


    vulnerable_samples = pd.DataFrame()

    for cwe in candidates:
    
        filtered_by_cwe = dataset_with_cwe[dataset_with_cwe['real_label'] == cwe]
        max_len_of_label = len(filtered_by_cwe)
        if max_len_of_label < number_per_class:
            additional_to_extract = number_per_class - max_len_of_label

            filtered_by_cwe_first = filtered_by_cwe.sample(n = max_len_of_label, random_state = seed ) 
        else:
            filtered_by_cwe_first = filtered_by_cwe.sample(n = number_per_class, random_state = seed ) 

        vulnerable_samples = pd.concat([vulnerable_samples,filtered_by_cwe_first], ignore_index=True)




    dataset_without_cwe = dataset_without_cwe.sample(n = number_per_class, random_state = seed ) 
    dataset = pd.concat([vulnerable_samples,dataset_without_cwe], ignore_index=True)


    dataset_shuffled = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    X_train, X_val, y_train, y_val, label_to_encoded, label_encoder = balanced_splitted(dataset_shuffled)

    threshold = number_per_class  # Set the threshold for the minimum number of samples per class

    # Convert y_train to a numpy array if it's not already, to use np.unique
    y_train_array = np.array(y_train)  # Assuming y_train is a list initially
    unique, counts = np.unique(y_train_array, return_counts=True)
    class_counts = dict(zip(unique, counts))

  
    # Find classes with fewer than the threshold number of samples
    undersampled_classes = {class_label: count for class_label, count in class_counts.items() if count < threshold}

    # Create a copy of X_train and y_train for oversampling
    oversampled_X_train = X_train.copy()
    oversampled_y_train = list(y_train)  # Convert to list if it's not already

    for class_label, count in undersampled_classes.items():
        # Find indices of the current undersampled class
        class_indices = np.where(y_train_array == class_label)[0]
        
        # Calculate the number of samples to replicate
        samples_to_add = threshold - count
        
        # Randomly choose samples to duplicate
        samples_indices_to_duplicate = np.random.choice(class_indices, size=samples_to_add, replace=True)
        
        # Append duplicated samples to the training data
        oversampled_X_train = pd.concat([oversampled_X_train, X_train.iloc[samples_indices_to_duplicate]], ignore_index=True)
        oversampled_y_train.extend(y_train_array[samples_indices_to_duplicate].tolist())

    # Optionally shuffle the dataset to mix the oversampled data
    shuffled_indices = np.random.permutation(len(oversampled_y_train))
    X_train = oversampled_X_train.iloc[shuffled_indices]
    y_train = [oversampled_y_train[i] for i in shuffled_indices]

    # Count the occurrences of each class in the training and validation sets
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)


    return X_train, X_val, y_train, y_val, label_to_encoded, label_encoder




def gemma_preprocessing_function(dataset):

                
        text = dataset['func']

        return tokenizer(text, truncation=True, max_length=2048,)
    

