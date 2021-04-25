from fastai.basics import *

# These functions define the layer groups for freezing 
# (i.e. learner.freeze_to(1) will freeze all layer groups up to the first)


def bert_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.bert.embeddings,
                m.transformer.bert.encoder.layer[0],
                m.transformer.bert.encoder.layer[1],
                m.transformer.bert.encoder.layer[2],
                m.transformer.bert.encoder.layer[3],
                m.transformer.bert.encoder.layer[4],
                m.transformer.bert.encoder.layer[5],
                m.transformer.bert.encoder.layer[6],
                m.transformer.bert.encoder.layer[7],
                m.transformer.bert.encoder.layer[8],
                m.transformer.bert.encoder.layer[9],                
                )]
    groups = L(groups + [nn.Sequential(m.transformer.bert.encoder.layer[10], m.transformer.bert.encoder.layer[11], m.transformer.bert.pooler) ] + [m.transformer.classifier]) 
    return groups.map(params)

def bertfinetuned_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.bert.embeddings,
                m.transformer.bert.encoder.layer[0],
                m.transformer.bert.encoder.layer[1],
                m.transformer.bert.encoder.layer[2],
                m.transformer.bert.encoder.layer[3],
                m.transformer.bert.encoder.layer[4],
                m.transformer.bert.encoder.layer[5],
                m.transformer.bert.encoder.layer[6],
                m.transformer.bert.encoder.layer[7],
                m.transformer.bert.encoder.layer[8],
                m.transformer.bert.encoder.layer[9],                
                )]
    groups = L(groups + [nn.Sequential(m.transformer.bert.encoder.layer[10], m.transformer.bert.encoder.layer[11], m.transformer.bert.pooler )] + [m.transformer.classifier] + [m.classifier2]) 
    return groups.map(params)

def bert4_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.embeddings,
                m.transformer.encoder.layer[0],
                m.transformer.encoder.layer[1],
                m.transformer.encoder.layer[2],
                m.transformer.encoder.layer[3],
                m.transformer.encoder.layer[4],
                m.transformer.encoder.layer[5],
                m.transformer.encoder.layer[6],
                m.transformer.encoder.layer[7])]
    groups = L(groups + [nn.Sequential(m.transformer.encoder.layer[8], m.transformer.encoder.layer[9], m.transformer.encoder.layer[10], m.transformer.encoder.layer[11])] + [m.classifier]) 
    return groups.map(params)

def bert4dense_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.embeddings,
                m.transformer.encoder.layer[0],
                m.transformer.encoder.layer[1],
                m.transformer.encoder.layer[2],
                m.transformer.encoder.layer[3],
                m.transformer.encoder.layer[4],
                m.transformer.encoder.layer[5],
                m.transformer.encoder.layer[6],
                m.transformer.encoder.layer[7])]
    groups = L(groups + [nn.Sequential(m.transformer.encoder.layer[8], m.transformer.encoder.layer[9], m.transformer.encoder.layer[10], m.transformer.encoder.layer[11])] + [m.dense, m.classifier]) 
    return groups.map(params)

def bert4cnn_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.embeddings,
                m.transformer.encoder.layer[0],
                m.transformer.encoder.layer[1],
                m.transformer.encoder.layer[2],
                m.transformer.encoder.layer[3],
                m.transformer.encoder.layer[4],
                m.transformer.encoder.layer[5],
                m.transformer.encoder.layer[6],
                m.transformer.encoder.layer[7]
                )]
    groups = L(groups + [nn.Sequential(m.transformer.encoder.layer[8], m.transformer.encoder.layer[9], m.transformer.encoder.layer[10], m.transformer.encoder.layer[11] )] + [m.convs1[0], m.convs1[1], m.convs1[2], m.classifier]) 
    return groups.map(params)

def bert4cnn_double_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.embeddings,
                m.transformer.encoder.layer[0],
                m.transformer.encoder.layer[1],
                m.transformer.encoder.layer[2],
                m.transformer.encoder.layer[3],
                m.transformer.encoder.layer[4],
                m.transformer.encoder.layer[5],
                m.transformer.encoder.layer[6],
                m.transformer.encoder.layer[7],

                m.transformer_q.embeddings,
                m.transformer_q.encoder.layer[0],
                m.transformer_q.encoder.layer[1],
                m.transformer_q.encoder.layer[2],
                m.transformer_q.encoder.layer[3],
                m.transformer_q.encoder.layer[4],
                m.transformer_q.encoder.layer[5],
                m.transformer_q.encoder.layer[6],
                m.transformer_q.encoder.layer[7]                
                )]
    groups = L(groups + [nn.Sequential(m.transformer.encoder.layer[8], m.transformer.encoder.layer[9], m.transformer.encoder.layer[10], m.transformer.encoder.layer[11], m.transformer_q.encoder.layer[8], m.transformer_q.encoder.layer[9], m.transformer_q.encoder.layer[10], m.transformer_q.encoder.layer[11] )] + [m.convs1[0], m.convs1[1], m.convs1[2], m.classifier]) 
    return groups.map(params)




def bertcnn_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.embeddings,
                m.transformer.encoder.layer[0],
                m.transformer.encoder.layer[1],
                m.transformer.encoder.layer[2],
                m.transformer.encoder.layer[3],
                m.transformer.encoder.layer[4],
                m.transformer.encoder.layer[5],
                m.transformer.encoder.layer[6],
                m.transformer.encoder.layer[7],
                m.transformer.encoder.layer[8],
                m.transformer.encoder.layer[9],
                m.transformer.encoder.layer[10]
                )]
    groups = L(groups + [m.transformer.encoder.layer[11]]) 
    return groups.map(params)

def albert_cls_splitter(m):
    groups = [nn.Sequential(m.transformer.albert.embeddings,
                m.transformer.albert.encoder.embedding_hidden_mapping_in, 
                m.transformer.albert.encoder.albert_layer_groups,
                m.transformer.albert.pooler)]
    groups = L(groups + [m.transformer.classifier]) 
    return groups.map(params)


def distilbert_cls_splitter(m):
    groups = [nn.Sequential(m.transformer.distilbert.embeddings,
                m.transformer.distilbert.transformer.layer[0], 
                m.transformer.distilbert.transformer.layer[1],
                m.transformer.distilbert.transformer.layer[2],
                m.transformer.distilbert.transformer.layer[3],
                m.transformer.distilbert.transformer.layer[4],
                m.transformer.distilbert.transformer.layer[5],
                m.transformer.pre_classifier)]
    groups = L(groups + [m.transformer.classifier]) 
    return groups.map(params)


def roberta_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.roberta.embeddings,
                  m.transformer.roberta.encoder.layer[0],
                  m.transformer.roberta.encoder.layer[1],
                  m.transformer.roberta.encoder.layer[2],
                  m.transformer.roberta.encoder.layer[3],
                  m.transformer.roberta.encoder.layer[4],
                  m.transformer.roberta.encoder.layer[5],
                  m.transformer.roberta.encoder.layer[6],
                  m.transformer.roberta.encoder.layer[7],
                  m.transformer.roberta.encoder.layer[8],
                  m.transformer.roberta.encoder.layer[9],
                  m.transformer.roberta.encoder.layer[10],
                  m.transformer.roberta.encoder.layer[11],
                  m.transformer.roberta.pooler)]
    groups = L(groups + [m.transformer.classifier])
    return groups.map(params)

splitters = {'bert':bert_cls_splitter,
            'bertfinetuned':bertfinetuned_cls_splitter,

            'bert-last-4-cls-token':bert4_cls_splitter,

            'bert-last-4-cls-token-dense':bert4dense_cls_splitter,
            
            'bert-last-4-cnn':bert4cnn_cls_splitter,

            'bert-last-4-cnn-double':bert4cnn_double_cls_splitter,

            'bert-last-4-pad-cnn':bert4cnn_cls_splitter,

            'bert4':bert4_cls_splitter,
            'bertcnn':bertcnn_cls_splitter,
            'bert4full':bert4dense_cls_splitter,
            'bert4cnn':bert4cnn_cls_splitter,

            'albert':albert_cls_splitter,
            'distilbert':distilbert_cls_splitter,
            'roberta':roberta_cls_splitter}

