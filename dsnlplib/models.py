from .splitters import splitters
import torch.nn as nn
import torch.nn.functional as F

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def empty_config(config, max_seq_len,dls):
    """Deprecated, use DSConfig """
    return config

def cnn_config(config, max_seq_len,dls):
    """Deprecated, use DSConfig """
    return empty_config(config, max_seq_len,dls)

class TransformerClassifier(nn.Module):
    def __init__(self, config_dict, tokenizer, model_name, pretrained=True, use_activ=False):
        super().__init__()

        self.pretrained = pretrained

        # Pretrained model weights
        self.pretrain_id = model_name

        self.config = config_dict  
        self.config.output_hidden_states = True
        self.use_activ = use_activ

        self.tok = tokenizer

    def loadPretrained(self):
        if self.pretrained: self.transformer = self.transformer_cls.from_pretrained(self.pretrain_id, config=self.config)
        else: self.transformer = self.transformer_cls.from_config(config=self.config)

        self.transformer_spltr = splitters[self.variation]    


    def forward(self, inputs, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        input_ids, attention_mask, token_type_ids = inputs

        return self.cforward(input_ids, attention_mask, token_type_ids, position_ids=None, head_mask=None, inputs_embeds=None, labels=None)

    def cforward(self, input_ids, attention_mask, token_type_ids, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        raise NotImplementedError("Must override ")


class BertClfier(TransformerClassifier):
    transformer_cls = AutoModelForSequenceClassification
    variation = 'bert'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loadPretrained()
        
    def cforward(self, input_ids, attention_mask, token_type_ids, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        
        logits = self.transformer(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0] 

        return logits


# Uses the CLS token from the last 4 layers
class BertLast4ClsTokenClfier(TransformerClassifier):
    transformer_cls = AutoModel
    variation = 'bert-last-4-cls-token'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ 
        Architecture
        """ 
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        ## Classifier of course has to be 4 * hidden_dim because we concat 4 layers        
        self.classifier = nn.Linear(self.config.hidden_size*4, self.config.num_labels)

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def cforward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        logits = encoder_output[0]

        hidden_states = encoder_output[2]

        cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)

        cls_out = cat[:, 0, :]
        pooled_output = self.dropout(cls_out )
        
        logits = self.classifier(cls_out )
        
        if (self.use_activ):
            logits = self.activation(logits)
        
        return logits

# Uses the CLS token from the last 4 layers and then a dense + a dropout layer before classifier
class BertLast4ClsTokenDenseClfier(TransformerClassifier):
    transformer_cls = AutoModel
    variation = 'bert-last-4-cls-token-dense'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        """ 
        Architecture
        """ 

        self.dense = nn.Linear(self.config.hidden_size*4, self.config.hidden_size*4)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        ## Classifier of course has to be 4 * hidden_dim because we concat 4 layers        
        self.classifier = nn.Linear(self.config.hidden_size*4, self.config.num_labels)

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def cforward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids)
        
        logits = encoder_output[0]

        hidden_states = encoder_output[2]

        cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)

        cls_out = cat[:, 0, :]

        pooled_output = self.dense(cls_out)
        
        pooled_output = self.activation(pooled_output)

        pooled_output = self.dropout(pooled_output )
        
        # Cassifier of course has to be 4 * hidden_dim, because we concat 4 layers
        logits = self.classifier(pooled_output )

        if (self.use_activ):
            logits = self.activation(logits)

        return logits

# Uses all tokens from the last 4 layers and then a KimCNN
class BertLast4CnnClfier(TransformerClassifier):
    transformer_cls = AutoModel
    variation = 'bert-last-4-cnn'


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ 
        Architecture
        """ 

        V = self.config.embed_num
        D = self.config.embed_dim
        C = self.config.num_labels
        Co = self.config.kernel_num
        Ks = self.config.kernel_sizes
        
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(len(Ks) * Co, C)
        self.activation = nn.ReLU()


        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def cforward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        

        encoder_output = self.transformer(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        logits = encoder_output[0]

        hidden_states = encoder_output[2]

        cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=1)

        x = cat
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
      
        x = self.dropout(x)  # (N, len(Ks)*Co)

        logits = self.classifier(x)  # (N, C)

        if (self.use_activ):
            logits = self.activation(logits)

        return logits


# Same as BertLast4CnnInvertClfier, but uses AutoModel, pads the transformer output to serve fixed length vectors to the cnn
class BertLast4PadCnnClfier(TransformerClassifier):
    transformer_cls = AutoModel
    variation = 'bert-last-4-cnn'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # After ~ 800 iterations at 1-e3
        """ 
        Architecture
        """ 

        V = self.config.embed_num
        D = self.config.embed_dim
        C = self.config.num_labels
        Co = self.config.kernel_num
        Ks = self.config.kernel_sizes
        
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(len(Ks) * Co, C)

        self.activation = nn.ReLU()

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def cforward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        

        encoder_output = self.transformer(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        
        logits = encoder_output[0]

        hidden_states = encoder_output[2]

        cat = torch.cat(
            [
                nn.functional.pad(
                    hidden_states[i], 
                    (0, 0, 0, (self.config.embed_dim//4) -hidden_states[i].shape[1])
                )
              
                for i in [-1,-2,-3,-4]
            ], 
                dim=1
            )

        x = cat
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        
        x = self.dropout(x)  # (N, len(Ks)*Co)

        logits = self.classifier(x)  # (N, C)

        if (self.use_activ):
            logits = self.activation(logits)

        return logits 

class BertCoralLast4ClsTokenDenseClfier(TransformerClassifier):
    transformer_cls = AutoModelForSequenceClassification
    variation = 'bert-last-4-cls-token-dense'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        """ 
        Architecture
        """ 

        self.dense = nn.Linear(self.config.hidden_size*4, self.config.hidden_size*4)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        ## Classifier of course has to be 4 * hidden_dim because we concat 4 layers        
        self.classifier = nn.Linear(self.config.hidden_size*4, 1,bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.config.num_labels-1).float())

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def cforward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
    
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        logits = encoder_output[0]

        hidden_states = encoder_output[2]

        cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)

        cls_out = cat[:, 0, :]

        pooled_output = self.dense(cls_out)
        #pooled_output = self.activation(pooled_output)
        
        pooled_output = self.dropout(pooled_output )
        # classifier of course has to be 4 * hidden_dim, because we concat 4 layers

        logits = self.classifier(pooled_output)
        logits = logits + self.linear_1_bias

        ###################################################
        # Old:
        # probas = F.softmax(logits, dim=1)
        # New:
        #logits = logits + self.linear_1_bias
        #probas = torch.sigmoid(logits)
        return logits

class BertCoralLast4PadCnnClfier(TransformerClassifier):
    transformer_cls = AutoModel
    variation = 'bert-last-4-pad-cnn'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # After ~ 800 iterations at 1-e3
        """ 
        Architecture
        """ 

        V = self.config.embed_num
        D = self.config.embed_dim
        C = self.config.num_labels
        Co = self.config.kernel_num
        Ks = self.config.kernel_sizes
        
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        ## Classifier of course has to be 4 * hidden_dim because we concat 4 layers        
        self.classifier = nn.Linear(len(Ks) * Co, 1,bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.config.num_labels-1).float())

        self.activation = nn.ReLU()

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def cforward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        
        logits = encoder_output[0]

        hidden_states = encoder_output[2]

        cat = torch.cat(
            [
                nn.functional.pad(
                    hidden_states[i], 
                    (0, 0, 0, 512-hidden_states[i].shape[1])
                )
              
                for i in [-1,-2,-3,-4]
            ], 
                dim=1
            )

        x = cat
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)  # (N, len(Ks)*Co)

        logits = self.classifier(x)  # (N, 1)

        
        logits = logits + self.linear_1_bias

        #logits = self.activation(logits)

        return logits 
