import lizard
import pandas as pd
import os
import subprocess
import json
import tempfile
from tqdm import tqdm
from multiprocessing import Pool
import sys
import re
import argparse
from datasets import load_dataset, concatenate_datasets
import jsonlines
import pandas as pd


'''
Code in charge of processing the code complexity.
'''

def test_multiMetrics(code):
    loc = []
    loc.append(sum(1 for x in code if x == '\n'))
    print(loc)

def process_json(c_file_path,test_function):
    with open(c_file_path, 'w') as c_file:
        c_file.write(test_function)
    cmd = f"multimetric {c_file_path}"
    output = subprocess.check_output(cmd, shell=True, text=True)
    json_object = json.loads(output)
    return json_object

def remove_comments_and_empty_lines(code):
    # Remove comments (both // and /* */)
    code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # Remove empty lines
    code = re.sub(r'\n\s*\n', '\n', code)
    
    return code

def remove_imports(code):
    # Remove import statements
    code = re.sub(r'^#include\s+<.*?>\s*\n', '', code, flags=re.MULTILINE)
    return code

def read_jsonl_to_dataframe(file_path):
    funcs = []
    target = []
    cwes = []

    # Open the JSONL file in read mode
    with jsonlines.open(file_path) as reader:
        # Iterate over each line (which corresponds to a JSON object)
        for obj in reader:
            # Process each JSON object as needed
            if 'cwe' in obj.keys():
                try:     
                    funcs.append(obj['func'])
                    target.append(obj['target'])
                    cwes.append(obj['cwe'])
                except:
                    continue

    # Create a DataFrame from the extracted data
    df = pd.DataFrame()
    df['func'] = funcs
    df['target'] = target
    df['cwes'] = cwes
    return df


def main(path):
    df = pd.read_csv(path)  
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some dataset.")
    parser.add_argument('--dataset_name', type=str, default="primevul", help='Name of the dataset')
    parser.add_argument('--quarter', type=str, default="1", help='process in quarters use 1,2,3 and 4')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if "bigvul" in dataset_name:
        dataset = load_dataset("benjis/bigvul")
        whole_dataset =  concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        df = whole_dataset.to_pandas()
        df = df[df['lang'] == 'C']
        target_list = df['vul'].tolist()
        cwe_list = df['CWE ID'].tolist()
        df['func'] = df['func_after']


    elif "diverse" in dataset_name:
        path = '../data/diversevul/diversevul_output.csv'
        df = main(path)
        print(df.keys())
        target_list = df['target'].tolist()
        cwe_list = df['cwe'].tolist()
        

    else:
        train_file_path = '../data/primevul/primevul_train.jsonl'
        test_file_path = '../data/primevul/primevul_test.jsonl'
        valid_file_path = '../data/primevul/primevul_valid.jsonl'
        train_df = read_jsonl_to_dataframe(train_file_path)

        # Read testing data into a DataFrame
        test_df = read_jsonl_to_dataframe(test_file_path)

        # Read validation data into a DataFrame
        valid_df = read_jsonl_to_dataframe(valid_file_path)

        # Concatenate training, testing, and validation data into a single DataFrame
        df = pd.concat([train_df, test_df, valid_df], ignore_index=True)


        target_list = df['target'].tolist()
        cwe_list = df['cwes'].tolist()


    functions = df['func'].tolist()
    filtered_functions = []
    for func in functions:
        try:
            fun = remove_comments_and_empty_lines(func)
            fun = remove_imports(fun)
    
            filtered_functions.append(fun)
            
                
        except:
            continue

    # Define a function to process a specific quarter
    def process_quarter(quarter_num, filtered_functions, target_list, cwe_list,dataset_name ):

        code_info = []
        
        quarter_size = int(len(filtered_functions) * 0.25)
        start_index = (quarter_num - 1) * quarter_size
        end_index = quarter_num * quarter_size
        functions_list = filtered_functions[start_index:end_index]
        target_list = target_list[start_index:end_index]

        cwe_list = cwe_list[start_index:end_index]


        funcs = []
        target = []
        project = []
        commit_id = []
        cwe = []
        part = 0
        position = 0
        for test_function in tqdm(functions_list, desc=f"Processing quarter {quarter_num}"):
            c_file_path = os.path.join('./', f"testtraissn_{quarter_num}.c")
            json_object = process_json(c_file_path,test_function)
           
            code_info.append(json_object['overall'])
            target.append(target_list[position])

            cwe.append(cwe_list[position])
            funcs.append(functions_list[position])

            position += 1

            if len(code_info) == 1000:

                new_df = pd.DataFrame()
                new_df['code_metadata'] = code_info
                new_df['func'] = funcs
                new_df['target'] = target

                new_df['cwe'] = cwe
                new_df.to_csv(f'../data/splitted/{dataset_name}/{dataset_name}_filtered_all_sizes_with_metadata_{quarter_num}_{part}.csv', index=False)
                part += 1
                code_info = []
                funcs = []
                target = []

                cwe = []
        new_df = pd.DataFrame()
        new_df['code_metadata'] = code_info
        new_df['func'] = funcs
        new_df['target'] = target

        new_df['cwe'] = cwe
        
        new_df.to_csv(f'../data/splitted/{dataset_name}/{dataset_name}_vul_filtered_all_sizes_with_metadata_{quarter_num}_final.csv', index=False)
        
   


    process_quarter(int(args.quarter), filtered_functions, target_list, cwe_list, dataset_name)
