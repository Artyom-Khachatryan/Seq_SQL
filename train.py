from transformers import LlamaForCausalLM, LlamaTokenizerFast
import transformers
from util import Seq2SQLDataset
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",token="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")

tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-hf",token="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")
train_dataset = Seq2SQLDataset(
    dataset_name='spider',
    max_length=100,
    tokenizer = tokenizer,
    split = 'train',
    )

val_dataset = Seq2SQLDataset(
    dataset_name='spider',
    max_length=100,
    tokenizer = tokenizer,
    split = 'validation',
    )
model.resize_token_embeddings(32002)
for _, i in model.named_parameters():
    if _ == 'model.layers.25.mlp.up_proj.weight':
        i.requires_grad = True
    else:
        i.requires_grad = False
for _, i in model.named_parameters():
    print(i.requires_grad)
training_args = transformers.TrainingArguments(
    'checkpoint',
    use_cpu = True, 
    max_steps=2,
    num_train_epochs=2,
    save_total_limit=1,
    per_device_train_batch_size=1
)

training_args.set_dataloader(train_batch_size=1)
collator = transformers.DataCollatorForLanguageModeling(tokenizer = tokenizer,return_tensors="pt", mlm=False)
trainer = transformers.Trainer(model, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator = collator)

trainer.train()


