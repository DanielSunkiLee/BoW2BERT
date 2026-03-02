from pathlib import Path
import torch
from src import load_dataset
from src import split_data, SentimentDataset
from src import load_model, load_tokenizer
from src import compute_metrics
from src import train_model
from torch.utils.data import Subset
#set up csv file path.
#
#
base_dir = Path(__file__).resolve().parent
ML_dir = base_dir.parent.parent

DATA_DIR = (
    ML_dir
    / "Representation_Learning"
    / "Representation-Learning-for-NLP"
    / "data"
)

CSV_PATH = DATA_DIR / "training.1600000.processed.noemoticon.csv"

#CSV file
#  ↓
#pandas DataFrame
#  ↓
#df["text"]  ──▶ tokenizer ──▶ input_ids + attention_mask
#df["label"] ─────────────────▶ labels


# load data
texts, labels = load_dataset(CSV_PATH)

# split
text_train, text_eval, y_train, y_val = split_data(texts, labels)

# model & tokenizer
tokenizer = load_tokenizer()
model = load_model()
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
# tokenize
train_enc = tokenizer(text_train, truncation=True, padding=True, max_length=128)
eval_enc = tokenizer(text_eval, truncation=True, padding=True, max_length=128)

train_dataset = SentimentDataset(train_enc, y_train)
eval_dataset = SentimentDataset(eval_enc, y_val)

small_train = Subset(train_dataset,range(1000))
small_eval =  Subset(eval_dataset,range(1000))

# train
trainer = train_model(model, small_train, small_eval, compute_metrics)
loss_function = torch.nn.CrossEntropyLoss()

'''
optimizer, train, eval_dataset, trainer = accelerator.prepare(
    optimizer, small_train, small_eval, trainer)

for batch in train :
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = trainer.model(**inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
'''

tokenizer.save_pretrained("./saved_model")

metrics = trainer.evaluate(eval_dataset=small_eval)
print(metrics)
