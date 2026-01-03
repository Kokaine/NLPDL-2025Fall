# hw2_huggingface/dataHelper.py

import os
import json
import random
from typing import List
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, Value

## * You can add more helper functions or modify function arguments if needed


def restaurant(sep_token: str):
    '''Load the ABSA restaurant dataset.'''
    
    ## TODO: the ABSA restaurant dataset.

    data_dir = os.path.join("datasets", "SemEval14-res")

    LABEL_MAP = {'positive': 0, 'neutral': 1, 'negative': 2}

    splits = {}
    for split in ['train', 'test']:
        file_path = os.path.join(data_dir, f'{split}.json')
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        text_list = [f"{item['term']} {sep_token} {item['sentence']}" for item in data.values()]
        label_list = [LABEL_MAP[item['polarity']] for item in data.values()]

        splits[split] = Dataset.from_dict({'text': text_list, 'label': label_list})

    return DatasetDict(splits)


def laptop(sep_token: str):
    '''Load the ABSA laptop dataset.'''
    
    ## TODO: the ABSA laptop dataset.
    data_dir = os.path.join("datasets", "SemEval14-laptop")

    LABEL_MAP = {'positive': 0, 'neutral': 1, 'negative': 2}

    splits = {}
    for split in ['train', 'test']:
        file_path = os.path.join(data_dir, f'{split}.json')
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        text_list = [f"{item['term']} {sep_token} {item['sentence']}" for item in data.values()]
        label_list = [LABEL_MAP[item['polarity']] for item in data.values()]

        splits[split] = Dataset.from_dict({'text': text_list, 'label': label_list})

    return DatasetDict(splits)


def acl(sep_token: str):
    '''Load the ACL-ARC dataset.'''
    
    ## TODO: the ACL-ARC dataset.
    data_dir = os.path.join("datasets", "acl_sup")

    LABEL_MAP = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2, 
                 'Motivation': 3, 'Extends': 4, 'Background': 5}
    
    splits = {}
    for split in ['train', 'test']:
        file_path = os.path.join(data_dir, f'{split}.jsonl')
        text_list = []
        label_list = []
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                text_list.append(item['text'])
                label_list.append(LABEL_MAP[item['label']])
        
        splits[split] = Dataset.from_dict({'text': text_list, 'label': label_list})

    return DatasetDict(splits)


def agnews(sep_token: str):
    '''Load the AGNews dataset (test set onlys).'''
    
    ## TODO: the AGNews dataset.
    
    dataset = load_dataset("ag_news", split="test")

    new_features = dataset.features.copy()
    new_features['label'] = Value('int64')
    dataset = dataset.cast(new_features)

    datasetDict = dataset.train_test_split(test_size=0.1, seed=2025)

    return datasetDict


def get_fs(dataset_name: str, sep_token: str, sample_size: int):
    '''
    Get few-shot dataset. Call this function inside `get_dataset` if needed.
    dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''

    ## TODO: your code for preparing the few-shot dataset

    base_name = dataset_name.replace('_fs', '')
    
    dataset = get_dataset(base_name, sep_token)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    seed = 2025

    train_fs = train_dataset.shuffle(seed=seed).select(range(sample_size))
    test_fs = test_dataset.shuffle(seed=seed).select(range(sample_size))

    return DatasetDict({
        'train': train_fs, 
        'test': test_fs
    })

## ! DO NOT change the function name or arguments
def get_dataset(dataset_name: str | List[str], sep_token: str) -> DatasetDict:
    '''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
    dataset = None

    ## TODO: your code for preparing the dataset

    if isinstance(dataset_name, str):
        ## TODO: implement for single dataset and few-shot dataset
        if dataset_name.endswith('_fs'):
            dataset = get_fs(dataset_name, sep_token, sample_size=32)
        if dataset_name.startswith('restaurant'):
            dataset = restaurant(sep_token)
        elif dataset_name.startswith('laptop'):
            dataset = laptop(sep_token)
        elif dataset_name.startswith('acl'):
            dataset = acl(sep_token)
        elif dataset_name.startswith('agnews'):
            dataset = agnews(sep_token)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported yet.")



    elif isinstance(dataset_name, List):
        ## TODO: implement for aggregation
        dataset_list = [get_dataset(name, sep_token) for name in dataset_name]
        return  aggregate_datasets(dataset_list, dataset_name)

    else:
        raise ValueError("Unsupported dataset format.")

    return dataset


def aggregate_datasets(dataset_list, dataset_names):
    '''
    Aggregate multiple datasets into one dataset.
    '''

    label_offsets = {
        'restaurant_sup': 3, 'laptop_sup': 3, 'acl_sup': 6, 'agnews_sup': 4,
        'restaurant_fs': 3,  'laptop_fs': 3,  'acl_fs': 6,  'agnews_fs': 4
    }
    current_offset = 0
    train_datasets = []
    test_datasets = []
    
    for i, datasetDict in enumerate(dataset_list):
        dataset_name = dataset_names[i]

        if current_offset > 0:
            datasetDict = datasetDict.map(
                lambda x, offset=current_offset: {'label': x['label'] + offset}
            )

        train_datasets.append(datasetDict['train'])
        test_datasets.append(datasetDict['test'])

        if dataset_name in label_offsets:
            current_offset += label_offsets[dataset_name]

    aggregated_dataset = DatasetDict({
        'train': concatenate_datasets(train_datasets), 
        'test': concatenate_datasets(test_datasets)
    })

    return aggregated_dataset