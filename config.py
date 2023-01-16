import transformers

DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 1
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
