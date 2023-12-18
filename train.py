from transformers import LlamaForCausalLM,LlamaTokenizerFast, AutoModelForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model
import transformers
import torch
from util import Seq2SQLDataset
from sconf import Config

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")

tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-hf",token="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")

class Custom_config(LlamaConfig):
    def get_config(self,config_path):
        return Config(config_path)
    
llamaconfig = Custom_config()
config = llamaconfig.get_config('config.yaml')

train_dataset = Seq2SQLDataset(
    dataset_name='spider',
    max_length=config.max_length,
    tokenizer = tokenizer,
    split = 'train',
    )

val_dataset = Seq2SQLDataset(
    dataset_name=config.dataset_name,
    max_length=config.max_length,
    tokenizer = tokenizer,
    split = 'validation',
    )
model.resize_token_embeddings(32002)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            print(name)
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

l = find_all_linear_names(model)
l= [
"model.layers.29.mlp.down_proj",
"model.layers.30.self_attn.q_proj",
"model.layers.30.self_attn.k_proj"]
loraconfig = LoraConfig(
    r=16,
    lora_alpha=16,
    #lora_dropout=0.1,
    bias="none",
    target_modules=l,
    task_type="CAUSAL_LM"
)

get_peft_model(model, loraconfig)
# for _, i in model.named_parameters():
#     if _ == 'model.layers.25.mlp.up_proj.weight':
#         i.requires_grad = True
#     else:
#         i.requires_grad = False
param = 0
for _, i in model.named_parameters():
    if i.requires_grad ==True:
        param += i.numel()
training_args = transformers.TrainingArguments(
    'checkpoint',
    use_cpu = True, 
    max_steps=config.max_steps,
    num_train_epochs=config.epoch,
    save_total_limit=1,
)

training_args.set_dataloader(train_batch_size=config.batch_size)
collator = transformers.DataCollatorForLanguageModeling(tokenizer = tokenizer,return_tensors="pt", mlm=False)
trainer = transformers.Trainer(model, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator = collator)

trainer.train()


