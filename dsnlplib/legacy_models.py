from .splitters import splitters
import torch.nn as nn
import torch.nn.functional as F

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

class TransformerClassifier(nn.Module):
    def __init__(self, config_dict, n_class, model_name, pretrained=True):
        super(TransformerClassifier, self).__init__()

        self.pretrained = pretrained

        # Pretrained model weights
        self.pretrain_id = model_name

        self.config = config_dict  
        self.config.num_labels = n_class
        self.config.output_hidden_states = True


    def loadPretrained(self):
        if self.pretrained: self.transformer = self.transformer_cls.from_pretrained(self.pretrain_id, config=self.config)
        else: self.transformer = self.transformer_cls.from_config(config=self.config)

        self.transformer_spltr = splitters[self.variation]


    

class BertClfier(TransformerClassifier):
    transformer_cls = AutoModelForSequenceClassification
    variation = 'bert'

    test1 = '0.4724999964237213'

    def __init__(self, *args, **kwargs):
        super(BertClfier, self).__init__(*args, **kwargs)

        self.loadPretrained()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        attention_mask = (input_ids!=1).type(input_ids.type()) 
        logits = self.transformer(input_ids, attention_mask = attention_mask)[0] 

        return logits


# Uses the CLS token from the last 4 layers
class BertLast4ClsTokenClfier(TransformerClassifier):
    transformer_cls = AutoModelForSequenceClassification
    variation = 'bert-last-4-cls-token'

    test1 = '0.5608108043670654'

    def __init__(self, *args, **kwargs):
        super(BertLast4ClsTokenClfier, self).__init__(*args, **kwargs)



        """ 
        Architecture
        """ 

        # TODO: Am i using these two?
        self.dense = nn.Linear(self.config.hidden_size*4, self.config.hidden_size*4)
        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        ## Classifier of course has to be 4 * hidden_dim because we concat 4 layers        
        self.classifier = nn.Linear(self.config.hidden_size*4, self.config.num_labels)

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        attention_mask = (input_ids!=1).type(input_ids.type()) 
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask)
        
        logits = encoder_output[0]

        hidden_states = encoder_output[1]

        cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)

        cls_out = cat[:, 0, :]
        pooled_output = self.dropout(cls_out )
        

        # TODO: test this, didn't actually use dropout in first test
        # cls_out = self.dropout(cls_out )
        

        logits = self.classifier(cls_out )

        return logits

# Uses the CLS token from the last 4 layers and then a dense + a dropout layer before classifier
class BertLast4ClsTokenDenseClfier(TransformerClassifier):
    transformer_cls = AutoModelForSequenceClassification
    variation = 'bert-last-4-cls-token-dense'

    test1 = '0.53125'

    def __init__(self, *args, **kwargs):
        super(BertLast4ClsTokenDenseClfier, self).__init__(*args, **kwargs)


        """ 
        Architecture
        """ 

        self.dense = nn.Linear(self.config.hidden_size*4, self.config.hidden_size*4)
        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        ## Classifier of course has to be 4 * hidden_dim because we concat 4 layers        
        self.classifier = nn.Linear(self.config.hidden_size*4, self.config.num_labels)

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        attention_mask = (input_ids!=1).type(input_ids.type()) 
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask)
        
        logits = encoder_output[0]

        hidden_states = encoder_output[1]

        cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)

        cls_out = cat[:, 0, :]
        pooled_output = self.dense(cls_out)
        pooled_output = self.activation(pooled_output)
        
        pooled_output = self.dropout(pooled_output )
        # classifier of course has to be 4 * hidden_dim, because we concat 4 layers
        logits = self.classifier(pooled_output )

        return logits

# Uses all tokens from the last 4 layers and then a KimCNN
class BertLast4CnnClfier(TransformerClassifier):
    transformer_cls = AutoModelForSequenceClassification
    variation = 'bert-last-4-cnn'

    test1 = '0.5481418967247009'


    def __init__(self, *args, **kwargs):
        super(BertLast4CnnClfier, self).__init__(*args, **kwargs)


        """ 
        Architecture
        """ 

        V = self.config.embed_num
        D = self.config.embed_dim
        C = self.config.class_num
        Co = self.config.kernel_num
        Ks = self.config.kernel_sizes
        
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()

        ## Classifier of course has to be 4 * hidden_dim because we concat 4 layers        
        self.classifier = nn.Linear(self.config.hidden_size*4, self.config.num_labels)

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        attention_mask = (input_ids!=1).type(input_ids.type()) 
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask)
        
        logits = encoder_output[0]

        hidden_states = encoder_output[1]

        cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=1)

        x = cat
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc1(x)  # (N, C)

        output = self.sigmoid(logit)

        #TODO: no classifier layer here, why?

        logits = output

        return logits

# Uses all tokens from the last 4 layers and then a KimCNN, but moves the sigmoid right after the CNN (rather than after the final dense layer)
class BertLast4CnnInvertClfier(TransformerClassifier):
    transformer_cls = AutoModelForSequenceClassification
    variation = 'bert-last-4-cnn-invert'

    test1 = '0.5861486196517944'

    def __init__(self, *args, **kwargs):
        super(BertLast4CnnInvertClfier, self).__init__(*args, **kwargs)



        """ 
        Architecture
        """ 

        V = self.config.embed_num
        D = self.config.embed_dim
        C = self.config.class_num
        Co = self.config.kernel_num
        Ks = self.config.kernel_sizes
        
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.sigmoid = nn.Tanh()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc1 = nn.Linear(len(Ks) * Co, C)


        ## Classifier of course has to be 4 * hidden_dim because we concat 4 layers        
        self.classifier = nn.Linear(self.config.hidden_size*4, self.config.num_labels)

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        attention_mask = (input_ids!=1).type(input_ids.type()) 
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask)
        
        logits = encoder_output[0]

        hidden_states = encoder_output[1]

        cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=1)

        x = cat
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        
        x = self.sigmoid(x)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        output = self.fc1(x)  # (N, C)

        #TODO: no classifier layer here, why?

        logits = output

        return logits 

# Same as BertLast4CnnInvertClfier, but uses AutoModel, pads the transformer output to serve fixed length vectors to the cnn
class BertLast4PadCnnClfier(TransformerClassifier):
    transformer_cls = AutoModel
    variation = 'bert-last-4-pad-cnn-invert'

    test1 = '0.6571180820465088'    

    def __init__(self, *args, **kwargs):
        super(BertLast4PadCnnClfier, self).__init__(*args, **kwargs)


        # After ~ 800 iterations at 1-e3
        """ 
        Architecture
        """ 

        V = self.config.embed_num
        D = self.config.embed_dim
        C = self.config.class_num
        Co = self.config.kernel_num
        Ks = self.config.kernel_sizes
        
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        self.activation = nn.Tanh()

        # >>>>>>>>>>>>>>>>>>>
        self.loadPretrained()

        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        
        attention_mask = (input_ids!=1).type(input_ids.type()) 
        encoder_output = self.transformer(input_ids, attention_mask = attention_mask)
        
        
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

        x = self.activation(x)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        output = self.fc1(x)  # (N, C)

        logits = output

        return logits 
