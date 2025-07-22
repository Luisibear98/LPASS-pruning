

import torch
import os
import numpy as np
from transformers import  GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, BitsAndBytesConfig
from transformers import BertModel, BertConfig

import sys
import pandas as pd
import json
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
UD_EN_PREF = "en-ud-"


from sklearn.utils import shuffle
import pandas as pd
import ast
import json
from sklearn.model_selection import StratifiedShuffleSplit

# Convert your lists to a format suitable for splitting (e.g., numpy arrays or a DataFrame)
import numpy as np
import pandas as pd

from collections import Counter

def get_data( frac=1.0, field_to_predict='',cwe='', dataset_name=''):
    if "prime" in dataset_name:
        dataset_name = '../data/processed/primevul_meta.csv'
    if "bigvul" in dataset_name:
        dataset_name = '../data/processed/bigvul_metadata.csv'
    if "diverse" in dataset_name:
        dataset_name = '../data/processed/diverse_metadata.csv'
    csv_dataframe = pd.read_csv(dataset_name)

    csv_dataframe = shuffle(csv_dataframe)



    lista_cwes = csv_dataframe['cwe'].tolist()
    list_of_codes = csv_dataframe['func'].tolist()
    list_of_codes_metadata = csv_dataframe['code_metadata'].tolist()
    counter = 0
    lista_cwes = [' ' if x is np.nan else x for x in lista_cwes]



    # Initialize filtered lists
    lista_cwes_filtered = []
    list_of_codes_filtered = []
    list_of_codes_metadata_filtered = []
    
    # Filter the lists
    for cwes, code, metadata in zip(lista_cwes, list_of_codes, list_of_codes_metadata):
        try:
            if cwe in cwes:
                
                    dict_obj = ast.literal_eval(metadata)
                    if 'cyclomatic_complexity' in field_to_predict:
                        cyclomatic_complexity = dict_obj[field_to_predict]
                        if cyclomatic_complexity > 0 and cyclomatic_complexity <= 5 and len(code) < 512 and len(code) > 10:
                            lista_cwes_filtered.append(cwe)
                            list_of_codes_filtered.append(code)
                            list_of_codes_metadata_filtered.append(cyclomatic_complexity)
                    elif 'halstead_difficulty' in field_to_predict:
                        halstead = dict_obj[field_to_predict]
                        if halstead <= 30 and len(code) < 512 and len(code) > 10:
                            lista_cwes_filtered.append(cwe)
                            list_of_codes_filtered.append(code)
                            list_of_codes_metadata_filtered.append(halstead)
        except:     
                
            continue

    
    
    counter_0 = 0
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_4 = 0
    counter_5 = 0
    filtered_codes = []
    pointer = 0
    if 'halstead' in field_to_predict:
        categorized = []
        for value in list_of_codes_metadata_filtered:
            # 1-5,6-10,11-15,16-20,21-25,25-30
            if value <= 5 and counter_0 <= 1000:
                categorized.append(0)
                counter_0 += 1
                filtered_codes.append(list_of_codes_filtered[pointer])
            elif value >= 6 and value <= 10 and counter_1 <= 1000:
                categorized.append(1)
                counter_1 += 1
                filtered_codes.append(list_of_codes_filtered[pointer])
            elif value >= 11 and value <= 15 and counter_2 <= 1000:
                categorized.append(2)
                counter_2 += 1
                filtered_codes.append(list_of_codes_filtered[pointer])
            elif value >= 16 and value <= 20 and counter_3 <= 1000:
                categorized.append(3)
                counter_3 += 1
                filtered_codes.append(list_of_codes_filtered[pointer])
            elif value >= 21 and value <= 25 and counter_4 <= 1000:
                categorized.append(4)
                counter_4 += 1
                filtered_codes.append(list_of_codes_filtered[pointer])
            elif counter_5 <= 1000: 
                categorized.append(5)
                counter_5 += 1
                filtered_codes.append(list_of_codes_filtered[pointer])
            pointer += 1

        list_of_codes_metadata_filtered = categorized
        list_of_codes_filtered = filtered_codes







    
    counts = Counter(list_of_codes_metadata_filtered)



    data = pd.DataFrame({

    'Code': list_of_codes_filtered,
    field_to_predict: list_of_codes_metadata_filtered
})


    data[field_to_predict] = pd.Categorical(data[field_to_predict])


    split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)


    for train_index, test_index in split.split(data, data[field_to_predict]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]



    # Returning the training and testing data sets
    return train_set, test_set
  



def pre_process(cwe, charasteristic, dataset_name):
    frac = 0.75
    train_set, test_set = get_data(frac, charasteristic,cwe, dataset_name)


    print("Number of training samples:", len(train_set))
    print("Number of test samples:", len(test_set))
    return train_set, test_set




    
def get_model_and_tokenizer(model_name, device, random_weights=False, quantization=0):

    model_name = model_name


    if model_name.startswith('xlnet'):
        model = XLNetModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        sep = u'▁'
        emb_dim = 1024 if "large" in model_name else 768        
    elif model_name.startswith('gpt2'):
        model = GPT2Model.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        sep = 'Ġ'
        sizes = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600}
        emb_dim = sizes[model_name]
    elif model_name.startswith('xlm'):
        model = XLMModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = XLMTokenizer.from_pretrained(model_name)
        sep = '</w>'
    elif model_name.startswith('bert'):
        

        model = BertModel.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained(model_name)

        sep = '##'
        emb_dim = 1024 if "large" in model_name else 768
    elif model_name.startswith('distilbert'):
        model = DistilBertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        sep = '##'    
        emb_dim = 768
    elif model_name.startswith('roberta')  or 'codebert' in model_name or 'language-id'  in model_name:

        model_name = 'microsoft/codebert-base'  
        tokenizer = RobertaTokenizer.from_pretrained(model_name)

        model = RobertaModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        sep = 'Ġ'        
        emb_dim = 1024 if "large" in model_name else 768
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    if random_weights:
        print('Randomizing weights')
        model.init_weights()

    return model, tokenizer, sep, emb_dim
# this follows the HuggingFace API for pytorch-transformers
def get_sentence_repr(sentence, model, tokenizer, sep, model_name, device):
    """
    Get representations for one sentence
    """
    
    with torch.no_grad():
        ids = tokenizer.encode(sentence,add_special_tokens=False)


        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: list of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) (hidden_states at output of each layer plus initial embedding outputs)
        all_hidden_states = model(input_ids)[-1]
        # convert to format required for contexteval: numpy array of shape (num_layers, sequence_length, representation_dim)
        all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states]
        all_hidden_states = np.array(all_hidden_states)

    #For each word, take the representation of its last sub-word
    
    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert len(segmented_tokens) == all_hidden_states.shape[1], 'incompatible tokens and states'
    mask = np.full(len(segmented_tokens), False)

    if model_name.startswith('gpt2') or model_name.startswith('xlnet') or model_name.startswith('roberta'):

        for i in range(len(segmented_tokens)-1):
            if segmented_tokens[i+1].startswith(sep):
           
                mask[i] = True
        
        mask[-1] = True

    elif model_name.startswith('xlm'):
        # if current token is a new word, take it
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].endswith(sep):
                mask[i] = True
        mask[-1] = True
    elif model_name.startswith('codebert') or model_name.startswith('bert') or model_name.startswith('distilbert'):
        # if next token is not a continuation, take current token's representation
        for i in range(len(segmented_tokens)-1):
            if not segmented_tokens[i+1].startswith(sep):
                mask[i] = True
        mask[-1] = True
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    all_hidden_states = all_hidden_states[:, mask]


    # all_hidden_states = torch.tensor(all_hidden_states).to(device)

    return all_hidden_states



class Classifier(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
            
        self.linear = torch.nn.Linear(input_dim, output_dim)
            
    def forward(self, input):

        output = self.linear(input)

   
        return output



def build_classifier(emb_dim, num_labels, device='cpu'):

    classifier = Classifier(emb_dim, num_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters())

    return classifier, criterion, optimizer



def get_sentence_repr_codes(sentence, model, tokenizer, sep, model_name, device):
    """
    Get representations for one sentence
    """

    with torch.no_grad():

        ids = tokenizer.encode(sentence,add_special_tokens=False)



        input_ids = torch.tensor([ids]).to(device)

        # Hugging Face format: list of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) (hidden_states at output of each layer plus initial embedding outputs)
        all_hidden_states = model(input_ids)[-1]


        # convert to format required for contexteval: numpy array of shape (num_layers, sequence_length, representation_dim)
        all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states][1:len(all_hidden_states)]

        all_hidden_states = np.array(all_hidden_states)

        


    
    last_representations = []
    for layer in all_hidden_states:
        last_representations.append(layer[-1])
  
    return last_representations


def process_labels(number_models_layers, sentece_representations, y_labels):
    per_layer_representation = []
    per_layer_label = []

    for _ in range(number_models_layers):
        per_layer_representation.append([])
        per_layer_label.append([])


    for i in range(len(y_labels)):
        label = y_labels[i]
        intermediate_representation = sentece_representations[i]

        for j in range(number_models_layers):
            partial_list_representation = per_layer_representation[j]
            partial_list_label = per_layer_label[j]

            partial_list_representation.append(intermediate_representation[j])
            partial_list_label.append(label)

            per_layer_representation[j] = partial_list_representation
            per_layer_label[j] = partial_list_label

    return per_layer_representation, per_layer_label



