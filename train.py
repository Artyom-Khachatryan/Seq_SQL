from transformers import LlamaForCausalLM,LlamaTokenizerFast, AutoModelForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model
import transformers
import torch
from util import Seq2SQLDataset
from sconf import Config

#"meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained('vzgo/checkpoint-19000', token="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", load_in_8bit=True)
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
    split = 'validation',
    )


# def find_all_linear_names(model):
#     cls = torch.nn.Linear
#     lora_module_names = set()
#     multimodal_keywords = ['mm_projector']
#     for name, module in model.named_modules():
#         if any(mm_keyword in name for mm_keyword in multimodal_keywords):
#             continue
#         if isinstance(module, cls):
#             print(name)
#             lora_module_names.add(name)

#     if 'lm_head' in lora_module_names: # needed for 16-bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)

# l = find_all_linear_names(model)
l = [
 'model.layers.8.self_attn.k_proj',
 'model.layers.2.self_attn.q_proj',
 'model.layers.13.self_attn.v_proj',
 'model.layers.5.self_attn.o_proj',
 'model.layers.0.self_attn.k_proj',
 'model.layers.2.self_attn.o_proj',
 'model.layers.3.self_attn.v_proj',
 'model.layers.17.self_attn.q_proj',
 'model.layers.12.self_attn.q_proj',
 'model.layers.10.self_attn.v_proj',
 'model.layers.31.self_attn.v_proj',
 'model.layers.7.self_attn.v_proj',
 'model.layers.0.self_attn.q_proj',
 'model.layers.5.self_attn.v_proj',
 'model.layers.30.self_attn.o_proj',
 'model.layers.22.self_attn.o_proj',
 'model.layers.21.self_attn.o_proj',
 'model.layers.21.self_attn.k_proj',
 'model.layers.7.self_attn.k_proj',
 'model.layers.1.self_attn.k_proj',
 'model.layers.14.self_attn.q_proj',
 'model.layers.8.self_attn.o_proj',
 'model.layers.4.self_attn.v_proj',
 'model.layers.1.self_attn.q_proj',
 'model.layers.9.self_attn.q_proj',
 'model.layers.30.self_attn.v_proj',
 'model.layers.18.self_attn.k_proj',
 'model.layers.23.self_attn.v_proj',
 'model.layers.20.self_attn.k_proj',
 'model.layers.27.self_attn.k_proj',
 'model.layers.28.self_attn.v_proj',
 'model.layers.0.self_attn.v_proj',
 'model.layers.30.self_attn.q_proj',
 'model.layers.20.self_attn.q_proj',
 'model.layers.7.self_attn.q_proj',
 'model.layers.27.self_attn.q_proj',
 'model.layers.9.self_attn.k_proj',
 'model.layers.31.self_attn.o_proj',
 'model.layers.14.self_attn.o_proj',
 'model.layers.31.self_attn.k_proj',
 'model.layers.25.self_attn.k_proj',
 'model.layers.4.self_attn.q_proj',
 'model.layers.5.self_attn.k_proj',
 'model.layers.23.self_attn.o_proj',
 'model.layers.23.self_attn.k_proj',
 'model.layers.21.self_attn.v_proj',
 'model.layers.27.self_attn.o_proj',
 'model.layers.8.self_attn.q_proj',
 'model.layers.1.self_attn.o_proj',
  'model.layers.18.self_attn.v_proj',
 'model.layers.25.self_attn.o_proj',
 'model.layers.20.self_attn.v_proj',
 'model.layers.5.self_attn.q_proj',
 'model.layers.8.self_attn.v_proj',
 'model.layers.18.self_attn.q_proj',
 'model.layers.25.self_attn.v_proj',
 'model.layers.28.self_attn.o_proj',
 'model.layers.28.self_attn.k_proj',
 'model.layers.14.self_attn.v_proj',
 'model.layers.9.self_attn.o_proj',
 'model.layers.2.self_attn.k_proj',
 'model.layers.13.self_attn.k_proj',
 'model.layers.28.self_attn.q_proj',
 'model.layers.15.self_attn.q_proj',
 'model.layers.10.self_attn.q_proj',
 'model.layers.3.self_attn.q_proj',
 'model.layers.22.self_attn.q_proj',
 'model.layers.4.self_attn.o_proj',
 'model.layers.14.self_attn.k_proj',
 'model.layers.15.self_attn.v_proj',
 'model.layers.26.self_attn.k_proj',
 'model.layers.26.self_attn.o_proj',
 'model.layers.26.self_attn.v_proj',
 'model.layers.26.self_attn.k_proj',
 'model.layers.4.self_attn.k_proj',
 'model.layers.17.self_attn.k_proj',
 'model.layers.13.self_attn.o_proj',
 'model.layers.2.self_attn.v_proj',
 'model.layers.30.self_attn.k_proj',
 'model.layers.4.self_attn.k_proj',
 'model.layers.12.self_attn.k_proj',
 'model.layers.3.self_attn.o_proj',
 'model.layers.22.self_attn.v_proj',
 'model.layers.22.self_attn.k_proj',
 'model.layers.1.self_attn.v_proj',
 'model.layers.10.self_attn.k_proj',
 'model.layers.17.self_attn.o_proj',
 'model.layers.25.self_attn.q_proj',
 'model.layers.12.self_attn.o_proj',
 'model.layers.27.self_attn.v_proj',
 'model.layers.23.self_attn.q_proj',
 'model.layers.29.self_attn.q_proj',
 'model.layers.29.self_attn.o_proj',
 'model.layers.29.self_attn.v_proj',
 'model.layers.29.self_attn.q_proj',
 'model.layers.9.self_attn.v_proj',
 'model.layers.10.self_attn.o_proj',
 'model.layers.21.self_attn.q_proj',
 'model.layers.18.self_attn.o_proj',
 'model.layers.12.self_attn.v_proj',
 'model.layers.15.self_attn.o_proj',
 'model.layers.13.self_attn.q_proj',
 'model.layers.31.self_attn.q_proj',
 'model.layers.17.self_attn.v_proj',
 'model.layers.0.self_attn.o_proj',
 'model.layers.15.self_attn.k_proj',
 'model.layers.3.self_attn.k_proj',
 'model.layers.7.self_attn.o_proj',
 ]
loraconfig = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.15,
    bias="none",
    target_modules=l,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, loraconfig)
# model.forward(input_ids = train_dataset[0]['input_ids'].unsqueeze(0), labels = train_dataset[0]['labels'].unsqueeze(0))
# for _, i in model.named_parameters():
#     if _ == 'model.layers.25.mlp.up_proj.weight':
#         i.requires_grad = True
#     else:
#         i.requires_grad = False
param = 0
for _, i in model.named_parameters():
    if i.requires_grad ==True:
        param += i.numel()
print(param)
training_args = transformers.TrainingArguments(
    'checkpoint',
    fp16=True,
    num_train_epochs=config.epoch,
    save_total_limit=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    save_steps=1000,
    eval_steps=1500,
    do_train=True,
    do_eval = True,
    evaluation_strategy='steps',
    lr_scheduler_type = 'cosine',
    optim = 'adamw_bnb_8bit',
    logging_steps=200,
)
training_args.set_dataloader(train_batch_size=3, eval_batch_size=8)

collator = transformers.DataCollatorForLanguageModeling(tokenizer = tokenizer,return_tensors="pt", mlm=False)

trainer = transformers.Trainer(model, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator = collator, args = training_args)

trainer.train(resume_from_checkpoint= False)
# for _,i in model.named_parameters():
#     if i.requires_grad:
#         print(_, i)
#         break
# model.layers[3].self_attn.k_proj.state_dict()
# trainer.save_model('valod')

# model.save_state()

# torch.save(trainer, 'my_model')