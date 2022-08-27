import os
import sys
import pandas as pd
import numpy as np 
import torch
import random

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from transformers import BertTokenizer, BertModel

tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")

class CustomDataset(Dataset):
    """
    리스트 형태의 입력을 받아 (input, target) 형태의 Dataset 생성
    """
    def __init__(self, input_data:list, target_data:list) -> None: 
        self.X = input_data #x = input_data
        self.Y = target_data # y = target

    def __len__(self):
        return len(self.Y) # len(y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index] # getitem 호출 시 tuple 형태로 x, y 반환


class CustomClassifier(nn.Module):
    """
    사전 학습(pre-trained)된 `BERT` 모델을 불러와 그 위에 1 hidden layer와 binary classifier layer를 쌓아 CustomClassifier 생성
    """
    def __init__(self, hidden_size: int, n_label: int): #type-hint

        super(CustomClassifier, self).__init__() # parent class에서 __init__활용하기 위해서 
    
        self.bert = BertModel.from_pretrained("klue/bert-base") # bert model instance
    
        dropout_rate = 0.1
        linear_layer_hidden_size = 32

        #Classifier 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, linear_layer_hidden_size), # input shape : hidden_size, output shape : 32(linear_layer_hidden_size)
            nn.ReLU(),
            nn.Dropout(p = dropout_rate), #dropout_rate : 0.1
            nn.Linear(linear_layer_hidden_size, n_label) #input shape : 32, output shape : 2(n_label)
        )
  
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
      
    # 생성해 둔 bert instance로 outputs 생성  
        outputs = self.bert(
        input_ids,
        attention_mask = attention_mask,
        token_type_ids = token_type_ids
        )
    
        #BERT 모델 마지막 layer의 cls 토큰에 접근
        cls_token_last_hidden_states = outputs["pooler_output"] # pooler_output에 마지막 layer의 cls 토큰 저장돼있음.  
        logits = self.classifier(cls_token_last_hidden_states)
    
        return logits

def set_device():
    """Set torch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    return device


def custom_collate_fn(batch):
    """
  한 배치 내 문장(input)들을 tokenizing 한 후 텐서로 변환함. 
  이때, dynamic padding (즉, 같은 배치 내 토큰의 개수가 동일할 수 있도록, 부족한 문장에 [PAD] 토큰을 추가하는 작업)을 적용.
  라벨(target)은 텐서화 시킴.
  (input, target) 튜플 형태를 반환.
    """
    global tokenizer_bert
  
    input_list, target_list = zip(*batch) #(input_list, target_list)를 list가 감싸고 있기 때문에 unpacking(*) 후에 zip으로 할당
  
    tensorized_input = tokenizer_bert(
        input_list,
        add_special_tokens = True,
        return_tensors = 'pt',
        padding = "longest", #batch 단위 longest -> dynamic padding
        truncation = True # max_length 이상이면 token 제거
    )
  
    tensorized_label = torch.tensor(target_list)
  
    return tensorized_input, tensorized_label
    
    

