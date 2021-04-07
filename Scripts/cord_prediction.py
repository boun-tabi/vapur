import pandas as pd
import numpy as np
import json
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import keras
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, AdamW, AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from tqdm import tqdm, trange

if torch.cuda.is_available():
    print('cuda')

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def to_ids(sentences, tokenizer):

  fc_id = 101
  e1_id = 1
  end_e1_id = 2
  e2_id = 3
  end_e2_id = 4
  fc_end = 102
  input_ids = []

  for example_sent in sentences:
    
    input_id = []  
    if example_sent.find('<e1>') < example_sent.find('<e2>'):
      
      first = example_sent[:example_sent.find('<e1>')]
      entity_1 =  example_sent[(example_sent.find('<e1>') + 4):example_sent.find('</e1>')]
      inner = example_sent[(example_sent.find('</e1>') + 5):example_sent.find('<e2>')]
      entity_2 = example_sent[(example_sent.find('<e2>') + 4):example_sent.find('</e2>')]
      last = example_sent[(example_sent.find('</e2>') + 5):]


      input_id = ([fc_id] + tokenizer.encode(first, add_special_tokens = False) + [e1_id] 
                  + tokenizer.encode(entity_1, add_special_tokens = False) + [end_e1_id] 
                  + tokenizer.encode(inner, add_special_tokens = False) + [e2_id] 
                  + tokenizer.encode(entity_2, add_special_tokens = False) + [end_e2_id]   
                  + tokenizer.encode(last, add_special_tokens = False) + [fc_end] )
     
    else:
      
      first = example_sent[:example_sent.find('<e2>')]
      entity_2 =  example_sent[(example_sent.find('<e2>') + 4):example_sent.find('</e2>')]
      inner = example_sent[(example_sent.find('</e2>') + 5):example_sent.find('<e1>')]
      entity_1 = example_sent[(example_sent.find('<e1>') + 4):example_sent.find('</e1>')]
      last = example_sent[(example_sent.find('</e1>') + 5):]

      input_id = ([fc_id] + tokenizer.encode(first, add_special_tokens = False) + [e2_id] 
                  + tokenizer.encode(entity_2, add_special_tokens = False) + [end_e2_id] 
                  + tokenizer.encode(inner, add_special_tokens = False) + [e1_id] 
                  + tokenizer.encode(entity_1, add_special_tokens = False) + [end_e1_id]   
                  + tokenizer.encode(last, add_special_tokens = False) + [fc_end] )
      
    input_ids.append(input_id)
    
  input_ids = pad_sequences(input_ids, maxlen = 200, dtype="long", truncating="post", padding="post")

  return input_ids

def create_attention(input_ids):
  # Create attention masks
  attention_masks = []
  # Create a mask of 1s for each token followed by 0s for padding
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
  
  return attention_masks

class Bert_Chemprot(nn.Module):
    def __init__(self):
        super(Bert_Chemprot, self).__init__()
        num_labels = 2
        self.net_bert = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x, attention):
      x, _ = self.net_bert(x, attention)

      x = x[:, 0, :]

      x = self.classifier(x)

      return x


def cord_prediction(pair_path = 'pairs_tagger_2.json', save_dir = "model_12092020_2_classes_bert.pt"):
	with open(pair_path, 'rb') as f:
    	df_base_only = pd.read_json(f)

	tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
	test_article_id = list(df_base_only['doc_id'])
	test_arg1 = list(df_base_only['e1_start'])
	test_arg2 = list(df_base_only['e2_start'])
	test_sentence = list(df_base_only['changed_sent'])
	test_sent_id = list(df_base_only['sent_id'])


	test_input_ids = to_ids(test_sentence, tokenizer)
	test_attention_masks = create_attention(test_input_ids)

	# Test Data
	test_arg1s = torch.tensor(test_arg1).to(device)
	test_arg2s = torch.tensor(test_arg2).to(device)
	test_sent_ids = torch.tensor(test_sent_id).to(device)

	test_inputs = torch.tensor(test_input_ids).to(device)
	test_masks = torch.tensor(test_attention_masks).to(device)

	# Create an iterator of our data with torch DataLoader 
	batch_size = 16
	test_data = TensorDataset(test_inputs, test_masks, test_arg1s, test_arg2s, test_sent_ids)
	test_dataloader = DataLoader(test_data, batch_size = batch_size)

	criterion = nn.CrossEntropyLoss()
	num_training_steps = 3000
	num_warmup_steps = 400
	max_grad_norm = 1.0
	warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

	optimizer = AdamW(bert_chemprot.parameters(), lr=0.00003,  correct_bias=False, weight_decay=0.1)  # To reproduce BertAdam specific behavior set correct_bias=False
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

	
	bert_chemprot = Bert_Chemprot().to(device)

	checkpoint = torch.load(save_dir)
	bert_chemprot.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loss = checkpoint['loss']


	## Prediction on test set
	# Put model in evaluation mode
	bert_chemprot.eval()
	test_preds = []
	test_arg1 = []
	test_arg2 = []
	test_article_id_ = []
	test_sent_ids = []

	count = 1
	# Predict 
	for step, batch in enumerate(test_dataloader): 
	  
	  b_input_ids, b_input_mask,  b_arg1, b_arg2, b_sent_id = batch
	  
	  with torch.no_grad():
	    # Forward pass
	    pred_labels = bert_chemprot.forward(b_input_ids, b_input_mask)
	    _, predicted = torch.max(pred_labels.data, 1)   
	    
	    test_preds = test_preds + predicted.tolist()
	    test_arg1 = test_arg1 + b_arg1.tolist()
	    test_arg2 = test_arg2 + b_arg2.tolist()
	    test_article_id_.extend(test_article_id[(count-1)*batch_size: (count)*batch_size ])
	    test_sent_ids = test_sent_ids + b_sent_id.tolist()
	    count = count + 1

	pred_test = pd.DataFrame({'Relation Label': np.array(test_preds).flatten(), 'doc_id': np.array(test_article_id_).flatten(), 'e1_start': np.array(test_arg1).flatten(), 'e2_start': np.array(test_arg2).flatten(), 'sent_id': np.array(test_sent_ids).flatten() })
	df_final = df_base_only.merge(pred_test, on = ['doc_id', 'e1_start', 'e2_start', 'sent_id'], how = 'left')
	
	"""
	# write output file
	df_final.to_csv('test_pred_30082020_2_classes_2.csv')
	with open('test_pred_30082020_2_classes_2.json', 'w') as handle:
	    df_final.to_json(handle)
	"""
	
	return df_final
