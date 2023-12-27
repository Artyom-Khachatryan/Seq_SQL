from datasets import load_dataset
import sqlite3
import pandas as pd
from torch.utils.data import Dataset
import torch 
import random
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
            self.list_dataset = []
            for i in self.dataset:
                self.list_dataset.append(i)
        elif self.dataset_name =='spider':
            assert self.split!='test', "Seq2SQL model only have train and validation datasets"
            self.dataset = load_dataset('spider', split = self.split)
            self.list_dataset = []
            for i in self.dataset:
                if self.split == 'train':
                    conn = sqlite3.connect(f"spid/database/{i['db_id']}/{i['db_id']}.sqlite")
                if self.split == 'validation':
                    conn = sqlite3.connect(f"spid/test_database/{i['db_id']}/{i['db_id']}.sqlite")

                cursor = conn.cursor()

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                
                table_name = cursor.fetchall()
                table_list = []

                for table in table_name:
                    #table_list.append(table[0])
                    cursor.execute(f"PRAGMA table_info({table[0]})")
                    columns = cursor.fetchall()
                    column_list = []
                    for column in columns:
                        column_list.append(column[1])
                    table_list.append({'table': table[0], 'columns': column_list})
                    
                i['table'] = table_list
                i.pop('question_toks')
                i.pop('query_toks')
                i.pop('query_toks_no_value')
                self.list_dataset.append(i)

                conn.close()
        else:
            assert "We only have wikisql and spider datasets for Seq2SQL model"
    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, index):
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({"additional_special_tokens": ['</pet>']})
        if self.dataset_name == 'wikisql':
            question = self.list_dataset[index]['question']
            sql = self.list_dataset[index]['sql']['human_readable']
            
            instruction = f"I have following columns in my SQL table: {self.list_dataset[index]['table']['header']}, which have rows like this {self.list_dataset[index]['table']['rows'][0]}, table_name=table. Generate SQL query for this question: " + f'{question}'
            
            input_text = instruction + '</pet>' + sql + '</s>'
            
            input_ids = self.tokenizer(input_text, max_length = self.max_length, padding="max_length", return_tensors = 'pt', truncation = True)['input_ids'].squeeze(0)
            labels = input_ids.clone()
            pet_where = self.tokenizer.encode('</pet>')[1]
            pet_index = int(torch.where(labels==pet_where)[0])
            labels[0: pet_index+1] = self.ignore_id
            labels[labels==self.tokenizer.pad_token_id]= self.ignore_id
            a = {'input_ids': input_ids, 'labels': labels}
            return a
            
                
        elif self.dataset_name == 'spider':

            question = self.dataset[index]['question']
            sql = self.dataset[index]['query']

            if self.split == 'train':
                instruction = f"I have following SQL database: {self.list_dataset[index]['db_id']}s. Wher I have list of dictionarys, each of them I have two keys 'table' and 'columns'(table and corresponding columns): {self.list_dataset[index]['table']}.Generate SQL query for this question: {question}"
                input_text = instruction + '</pet>' + sql + '</s>'
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