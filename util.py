from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset
import torch 

class Seq2SQLDataset(Dataset):
    def __init__(self, split, dataset_name, tokenizer, max_length, ignore_id = -100):
        super().__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_id = ignore_id
        if self.dataset_name == 'wikisql':
            self.dataset = load_dataset('wikisql', split = self.split)
        elif self.dataset_name =='spider':
            assert self.split!='test', "Seq2SQL model only have train and validation datasets"
            self.dataset = load_dataset('spider', split = self.split)
        else:
            assert "We only have wikisql and spider datasets for Seq2SQL model"
    def __len__(self):
        return self.dataset.num_rows
    def __getitem__(self, index):
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({"additional_special_tokens": ['</pet>']})
        if self.dataset_name == 'wikisql':
            question = self.dataset['question'][index]
            sql = self.dataset['sql'][index]['human_readable']
            if self.split == 'train':
                input_text = question + ' ' + '</pet>' + ' ' +sql
                input_ids = self.tokenizer(input_text, max_length = self.max_length, padding="max_length", return_tensors = 'pt', truncation = True)['input_ids'].squeeze(0)
            
                labels = input_ids.clone()
                pet_where = self.tokenizer.encode('</pet>')[1]
                pet_index = int(torch.where(labels==pet_where)[0])
                labels[0: pet_index+1] = self.ignore_id
                labels[labels==self.tokenizer.pad_token_id]= self.ignore_id
                a = {'input_ids': input_ids, 'labels': labels}
                return a
            else:
                question += '</pet>'
                input_ids= self.tokenizer(question, return_tensors = 'pt', truncation = True)['input_ids'].squeeze(0)
                labels = self.tokenizer(sql, return_tensors = 'pt')['input_ids'].squeeze(0)[1: ]
                a = {'input_ids': input_ids, 'labels': labels}
                return a
                
        elif self.dataset_name == 'spider':

            question = self.dataset['question'][index]
            sql = self.dataset['query'][index]

            if self.split == 'train':
                input_text = question + ' ' + '</pet>' + ' ' +sql
                input_ids = self.tokenizer(input_text, max_length = self.max_length, padding="max_length", return_tensors = 'pt', truncation = True)['input_ids'].squeeze(0)
            
                labels = input_ids.clone()
                pet_where = self.tokenizer.encode('</pet>')[1]
                pet_index = int(torch.where(labels==pet_where)[0])
                labels[0: pet_index+1] = self.ignore_id
                labels[labels==self.tokenizer.pad_token_id]= self.ignore_id
                a = {'input_ids': input_ids, 'labels': labels}
                return a
            else:
                question += '</pet>'
                input_ids= self.tokenizer(question, return_tensors = 'pt', truncation = True)['input_ids'].squeeze(0)
                labels = self.tokenizer(sql, return_tensors = 'pt')['input_ids'].squeeze(0)[1: ]
                a = {'input_ids': input_ids, 'labels': labels}
                return a
            
    def table(self, index):
        if self.dataset_name == 'wikisql':
            table = self.dataset['table'][index]
            df = pd.DataFrame(columns=table['header'])
            row_list = table['rows']
            for i in row_list:
                df.loc[len(df.index)] = i
            return df
        elif self.dataset_name == 'spider':
            assert "Spider dataset doesn't have tables"

# tokenizer =LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-hf",token="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")