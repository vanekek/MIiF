import pandas as pd
import numpy as np
import joblib
import torch.nn as nn
import torch

# Модель нейросети
class TransactionsRnn(nn.Module):
    def __init__(self, transactions_cat_features, embedding_projections, 
                 rnn_units=128, top_classifier_units=32):
        super(TransactionsRnn, self).__init__()
        
        self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature], 
                                                                                            padding_idx=None) 
                                                          for feature in transactions_cat_features])
        self._spatial_dropout = nn.Dropout2d(0.05)
        self._transaction_cat_embeddings_concated_dim = sum([embedding_projections[x][1] for x in transactions_cat_features])
      
        
        self._gru = nn.GRU(input_size=self._transaction_cat_embeddings_concated_dim,
                             hidden_size=rnn_units, batch_first=True, bidirectional=True)
        
        self._hidden_size = rnn_units
        
        # построим классификатор, он будет принимать на вход: 
        # [max_pool(gru_states), avg_pool(gru_states), product_embed]
        pooling_result_dimension = self._hidden_size * 2
         
        self._top_classifier = nn.Sequential(nn.Linear(in_features=2*pooling_result_dimension, 
                                                       out_features=top_classifier_units),
                                             nn.ReLU(),
                                             nn.Linear(in_features=top_classifier_units, out_features=1)
                                            )
        
    def forward(self, transactions_cat_features):
        batch_size = transactions_cat_features.shape[0]
        
        embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(0, 2, 1)

        states, _ = self._gru(dropout_embeddings)
        
        rnn_max_pool = states.max(dim=1)[0]
        rnn_avg_pool = states.sum(dim=1) / states.shape[1]        
        
                
        combined_input = torch.cat([rnn_max_pool, rnn_avg_pool], dim=-1)
            
        logit = self._top_classifier(combined_input)        
        return logit
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class TransactionsRnn2(nn.Module):
    def __init__(self, transactions_cat_features, embedding_projections, product_col_name='product', 
                 rnn_units=128, top_classifier_units=32):
        super(TransactionsRnn2, self).__init__()
        
        self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature], 
                                                                                            padding_idx=None) 
                                                          for feature in transactions_cat_features])
        self._spatial_dropout = nn.Dropout2d(0.1)
        self._transaction_cat_embeddings_concated_dim = sum([embedding_projections[x][1] for x in transactions_cat_features])
        
        self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None)
        
        self._gru = nn.GRU(input_size=self._transaction_cat_embeddings_concated_dim,
                             hidden_size=rnn_units, batch_first=True, bidirectional=True)
        
        self._hidden_size = rnn_units
        
        # построим классификатор, он будет принимать на вход: 
        # [max_pool(gru_states), avg_pool(gru_states), product_embed]
        pooling_result_dimension = self._hidden_size * 2
         
        self._top_classifier = nn.Sequential(nn.Linear(in_features=2*pooling_result_dimension + 
                                                       embedding_projections[product_col_name][1], 
                                                       out_features=top_classifier_units),
                                             nn.ReLU(),
                                             nn.Linear(in_features=top_classifier_units, out_features=1)
                                            )
        
    def forward(self, transactions_cat_features, product_feature):
        batch_size = product_feature.shape[0]
        
        embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(0, 2, 1)

        states, _ = self._gru(dropout_embeddings)
        
        rnn_max_pool = states.max(dim=1)[0]
        rnn_avg_pool = states.sum(dim=1) / states.shape[1]        
        
        product_embed = self._product_embedding(product_feature)
                
        combined_input = torch.cat([rnn_max_pool, rnn_avg_pool, product_embed], dim=-1)
            
        logit = self._top_classifier(combined_input)        
        return logit
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)