from transformers import LlamaForCausalLM,LlamaTokenizerFast, AutoModelForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model
import transformers
from pandasql import sqldf
import torch
import json
import numpy as np
from nltk import edit_distance
from util import Seq2SQLDataset
from sconf import Config

#"meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained('vzgo/checkpoint-2000', token="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", load_in_8bit=True)
model.eval()
model.resize_token_embeddings(32002)

tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token="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")

class Custom_config(LlamaConfig):
    def get_config(self,config_path):
        return Config(config_path)
    
llamaconfig = Custom_config()
config = llamaconfig.get_config('config.yaml')

train_dataset = Seq2SQLDataset(
    dataset_name=config.dataset_name,
    max_length=config.max_length,
    tokenizer = tokenizer,
    split = 'train',
    )

val_dataset = Seq2SQLDataset(
    dataset_name=config.dataset_name,
    max_length=config.max_length,
    tokenizer = tokenizer,
    split = 'validation[:10%]',
    )
import warnings

# Filter out all warnings (not recommended for debugging)
warnings.filterwarnings("ignore")
n = 0
n_edit_distance = 0
error = 0
for i in val_dataset:
    
    print(n)
    input_ids = i['input_ids']

    q_id = int(torch.where(input_ids == 1139)[0])
    pad_id = int(torch.where(input_ids == 32000)[0][0])
    index = int(torch.where(input_ids == 32001)[0])
    sql = input_ids[index+1: pad_id-1]
    question  = input_ids[q_id+2: index]
    sql = tokenizer.decode(sql)
    sql = sql.replace('table', 'df')
    print(sql)
    question = tokenizer.decode(question)

    input_ids = torch.tensor(input_ids[:index+1]).unsqueeze(0)
    output = model.generate(input_ids, temperature=1, max_length = 650)[0]
    
    df = val_dataset.table(n)
    output = output[index+1: -1]
    output = tokenizer.decode(output)
    output = output.replace('table', 'df')
    if 'COUNT' in output:
        o_l = output.split(' FROM ')
        select = o_l[0][0: 13] + '('
        o_l[0] = o_l[0][13: ]
        o_l[0] = f"[{o_l[0]}]"+ ")"
        output = select + o_l[0] + ' FROM ' +  o_l[1]
    elif ('MAX' or 'MIN') in output:
        o_l = output.split(' FROM ')
        select = o_l[0][0: 11] + '('
        o_l[0] = o_l[0][11: ]
        o_l[0] = f"[{o_l[0]}]" + ")"
        output = select + o_l[0] + ' FROM ' +  o_l[1]
    elif 'AVERAGE' in output:
        o_l = output.split(' FROM ')
        select = o_l[0][0: 15] + '('
        o_l[0] = o_l[0][15: ]
        o_l[0] = f"[{o_l[0]}]" +  + ")"
        output = select + o_l[0] + ' FROM ' +  o_l[1]
    else :
        o_l = output.split(' FROM ')
        select = o_l[0][0: 7]
        o_l[0] = o_l[0][7: ]
        o_l[0] = f"[{o_l[0]}]"
        output = select + o_l[0] + ' FROM ' +  o_l[1]
    out_l = output.split(' = ')
    if len(out_l) == 2:
        column_name = out_l[0].split('WHERE ')[1]
        if ('<' or '=' or '<=' or  '>=' or ')') not in column_name:
            row_name = out_l[1]
            ed_l = []
            if df[column_name].dtype == object:
                for i in df[column_name].items():
                    
                        ed_l.append(edit_distance(i[1], row_name))
                ind = np.argmin(ed_l)
                out_l[1] = f"'{df[column_name][ind]}'"
                o_1 = out_l[0].split('WHERE ')
                c_name = f"[{o_1[1]}]"
                
                o_1 = o_1[0] + "WHERE " + c_name

                output = o_1 + ' = ' + out_l[1]
        else:
             pass
    try:
        sqldf(output)
    except:
         error+=1
    n_edit_distance += (1-edit_distance(output, sql)/max(len(sql), len(output)))/168
    print(error)
    print(10*'*')
    print(output)
    n+=1
k = 0
d = {} 
for i in val_dataset:
    print(k)
    input_ids = i['input_ids']
    index = int(torch.where(input_ids == 32001)[0])
    input_ids = torch.tensor(input_ids[:index+1]).unsqueeze(0)
    output = model.generate(input_ids, temperature=1, max_length = 650)[0]

    output = output[index+1: -1]
    output = tokenizer.decode(output)
    d[f'{k}'] = output
    k+=1
    if k ==134:
        break
    
with open("val_jsons_sp1.json", "w") as outfile: 
    json.dump(d, outfile)