from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokens = tokenizer.tokenize("We are very happy to participate at this hackathon.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("We are very happy to participate at this hackathon.")


print(f'Tokens: {tokens}')
print(f'Tokens IDs: {token_ids}')
print(f'Tokens IDs: {input_ids}')

X_train = ["We are very happy to participate at this hackathon.",
           "We hope you don't hate it."]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch, labels=torch.tensor([1, 0]))
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)


save_directory = "saved_model"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)


