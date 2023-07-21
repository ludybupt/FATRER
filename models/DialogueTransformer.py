import torch
from torch import nn
from torch.distributions import distribution
from transformers import BertConfig, BertModel
from torch import Tensor
from typing import Optional, Tuple
import numpy as np
import torch.nn.functional as F
from torch.distributions import LogNormal, Dirichlet, Gamma, Laplace
from torch.distributions import kl_divergence
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

acceptable_context_type = ['conve', 'intra', 'inter', 'tgt']


def assign_GPU(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to(device)
    token_type_ids = Tokenizer_output['token_type_ids'].to(device)
    attention_mask = Tokenizer_output['attention_mask'].to(device)

    output = {'input_ids': tokens_tensor,
              'token_type_ids': token_type_ids,
              'attention_mask': attention_mask}

    return output


def merge_encoding(*inputs):
    tokens_tensor = torch.cat([i['input_ids'] for i in inputs])
    token_type_ids = torch.cat([i['token_type_ids'] for i in inputs])
    attention_mask = torch.cat([i['attention_mask'] for i in inputs])

    output = {'input_ids': tokens_tensor,
              'token_type_ids': token_type_ids,
              'attention_mask': attention_mask}
    split = inputs[0]['input_ids'].shape[0]
    return output, split


class Normalized_Output():
    loss = None
    logits = None

    def __init__(self, loss, logits):
        Normalized_Output.loss = loss
        Normalized_Output.logits = logits

class DialogueTRM_Hierarchical(nn.Module):
    def __init__(self, num_labels, bert_path, vocab, tokenizer, topic_num):
    #def __init__(self, num_labels, bert_path):
        self.config = BertConfig()
        super(DialogueTRM_Hierarchical, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(bert_path)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size, nhead=12)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, **kwargs):
        intra_encoding = kwargs.get('intra')
        intra_encoding = assign_GPU(intra_encoding)
        labels = kwargs.get('labels', None).to(device)

        bert_output = self.bert(**intra_encoding, return_dict=True)

        r = bert_output.last_hidden_state[:, 0, :]
        r = r.unsqueeze(0)
        trm_output = self.trm(r)
        output = self.dropout(trm_output)
        logits = self.classifier(output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        # (loss), scores, (hidden_states), (attentions)
        return Normalized_Output(logits=logits, loss=loss)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        print('sqrt_dim',self.sqrt_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        #print('key.shape', key.shape)
        #print('key.transpose(1, 2).shape', key.transpose(1, 2).shape)
        #print('value.shape', value.shape)
        #print('query.shape', query.shape)
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        #print('score.shape', score.shape)
        attn = F.softmax(score, -1)
        #print('attn.shape', attn.shape)
        context = torch.bmm(attn, value)
        #print('context.shape', context.shape)
        return context, attn


class ScaledDotProduct(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim: int):
        super(ScaledDotProduct, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        # context = torch.bmm(attn, value)
        return attn

class DialogueTRM_Hierarchical_with_topic_soft_1(nn.Module):

    def __init__(self, num_labels, bert_path, vocab, tokenizer, topic_num):
        self.config = BertConfig()
        super(DialogueTRM_Hierarchical_with_topic_soft_1, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(bert_path)

        self.vocab_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(vocab)).to(device)
        self.word_embedding = self.bert.get_input_embeddings().to(device)
        self.p_w_z = F.softmax(torch.rand(
            topic_num, len(self.vocab_ids)), -1).to(device)
        print(self.p_w_z.shape)

        self.doc2topic = ScaledDotProductAttention(dim=self.config.hidden_size)
        self.topic2word = ScaledDotProductAttention(
            dim=self.config.hidden_size)

        word_emb = self.word_embedding(self.vocab_ids)
        self.topic_emb = torch.mm(self.p_w_z, word_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size*2, nhead=12)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*2, self.num_labels)

    def get_p_w_z(self):
        return self.p_w_z

    def forward(self, **kwargs):
        intra_encoding = kwargs.get('intra')
        intra_encoding = assign_GPU(intra_encoding)

        tgt_input_ids = kwargs.get('tgt_input_ids')
        tgt_input_ids = assign_GPU(tgt_input_ids)
        # intra_input_ids = kwargs.get('intra_input_ids').to(device)
        labels = kwargs.get('labels', None).to(device)
        fine_tune = kwargs.get('fine_tune', True)
        activation1 = nn.LeakyReLU(0.1)
        activation2 = nn.Sigmoid()

        for param in self.bert.parameters():
            param.requires_grad = fine_tune

        bert_output = self.bert(**intra_encoding, return_dict=True)

        doc_rep = bert_output.last_hidden_state[:, 0, :]  # [7,768]
        doc_rep = doc_rep.unsqueeze(0)

        query = self.word_embedding(tgt_input_ids['input_ids'])  # [7,len,768]
        mask = tgt_input_ids['attention_mask']  # [7,len]
        query = mask.unsqueeze(-1)*query  # [7,len,768]
        query = query.sum(axis=1)/mask.sum(axis=-1).unsqueeze(-1)  # [7,768]
        query = query.unsqueeze(0)  # [1,7,768]

        # topic_emb = self.topic_embedding(self.topic_ids)
        topic_emb = self.topic_emb.unsqueeze(0)  # [1,40,768]
        # print(topic_emb.shape)

        word_emb = self.word_embedding(self.vocab_ids)
        word_emb = word_emb.unsqueeze(0)  # [1, 1737, 768]
        # print(query.shape, topic_emb.shape,word_emb.shape)

        # r_z_d:[7, 1, 768], p_z_d:[1, 7, 40]
        r_z_d1, p_z_d = self.doc2topic(query, topic_emb, topic_emb)

        # r_w_z:[40, 1, 768] p_w_z:[1, 40, 1737]
        topic_emb_updated, p_w_z = self.topic2word(
            topic_emb, word_emb, word_emb)

        self.p_w_z = p_w_z.squeeze(0)
        self.topic_emb = topic_emb_updated.squeeze(0).detach().clone()
        # print(float(p_w_z[0,0,0]))
        # r_w_z = r_w_z.permute(1,0,2) #[1, 40, 768]
        # word_emb = word_emb.permute(1,0,2) #[1, 1737, 768]
        # query = query.permute(1,0,2)
        # topic_emb_by_words = torch.bmm(p_w_z, word_emb) #[1, 40, 768]
        r_z_d2 = torch.bmm(p_z_d, topic_emb_updated)  # [1, 7, 768]

        r_z_d1_sig = activation1(r_z_d1)
        r_z_d1_gat = activation2(r_z_d1)
        r_z_d2_sig = activation1(r_z_d2)
        r_z_d2_gat = activation2(r_z_d2)

        r_z_d = r_z_d1_gat*r_z_d2_sig + r_z_d2_gat*r_z_d1_sig

        r = torch.cat((r_z_d, doc_rep), 2)
        # r = torch.cat((r_z_d1 * r_z_d2, query),2)
        r = self.dropout(r)

        trm_output = self.trm(r)
        output = self.dropout(trm_output)
        logits = self.classifier(output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        # (loss), scores, (hidden_states), (attentions)
        return Normalized_Output(logits=logits, loss=loss)

class DialogueTRM_Hierarchical_with_topic_soft_with_kl_1(nn.Module):

    def __init__(self, num_labels, bert_path, vocab, tokenizer, topic_num):
        self.config = BertConfig()
        super(DialogueTRM_Hierarchical_with_topic_soft_with_kl_1, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(bert_path)
        #print('vocab', vocab)
        self.vocab_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(vocab)).to(device)
        self.word_embedding = self.bert.get_input_embeddings().to(device)
        self.p_w_z = F.softmax(torch.rand(topic_num, len(self.vocab_ids)), -1).to(device) #beta  topic2word
        print('self.p_w_z', self.p_w_z.shape)
        self.doc2topic = ScaledDotProductAttention(dim=self.config.hidden_size)
        print('self.doc2topic', self.doc2topic)
        self.topic2word = ScaledDotProductAttention(dim=self.config.hidden_size)

        word_emb = self.word_embedding(self.vocab_ids)
        print('word_emb.shape:', word_emb.shape)
        self.topic_emb = torch.mm(self.p_w_z, word_emb)
        print('self.topic_emb.shape:', self.topic_emb.shape)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size*2, nhead=12)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*2, self.num_labels)
        self.beta = 0.01

    def get_p_w_z(self):
        return self.p_w_z

    def forward(self, **kwargs):
        intra_encoding = kwargs.get('intra')
        intra_encoding = assign_GPU(intra_encoding)
        tgt_input_ids = kwargs.get('tgt_input_ids')
        tgt_input_ids = assign_GPU(tgt_input_ids)
        
        distribution_labels = kwargs.get('distribution_labels', None).to(device)
        labels = kwargs.get('labels', None).to(device)
        fine_tune = kwargs.get('fine_tune', True)
        activation1 = nn.LeakyReLU(0.1)
        activation2 = nn.Sigmoid()

        for param in self.bert.parameters():
            param.requires_grad = fine_tune

        bert_output = self.bert(**intra_encoding, return_dict=True)
        
        doc_rep = bert_output.last_hidden_state[:, 0, :]  # [7,768] cls_token is the first token  that is， r_i_context
        doc_rep = doc_rep.unsqueeze(0) # [1,7,768]

        query = self.word_embedding(tgt_input_ids['input_ids'])  # [7,len,768]
        mask = tgt_input_ids['attention_mask']  # [7,len]
        query = mask.unsqueeze(-1)*query  # [7,len,768]
        query = query.sum(axis=1)/mask.sum(axis=-1).unsqueeze(-1)  # [7,768] average of word embedding
        query = query.unsqueeze(0)  # [1,7,768]

        # topic_emb = self.topic_embedding(self.topic_ids)
        topic_emb = self.topic_emb.unsqueeze(0)  # [1,40,768]


        word_emb = self.word_embedding(self.vocab_ids)
        word_emb = word_emb.unsqueeze(0)  # [1, 1737, 768]
        r_z_d1, p_z_d = self.doc2topic(query, topic_emb, topic_emb)
        
        topic_emb_updated, p_w_z = self.topic2word(topic_emb, word_emb, word_emb)
        self.p_w_z = p_w_z.squeeze(0)
        self.topic_emb = topic_emb_updated.squeeze(0).detach().clone()
        
        # r_w_z = r_w_z.permute(1,0,2) #[1, 40, 768]
        # word_emb = word_emb.permute(1,0,2) #[1, 1737, 768]
        # query = query.permute(1,0,2)
        # topic_emb_by_words = torch.bmm(p_w_z, word_emb) #[1, 40, 768]
        p_w_z_d = torch.mm(p_z_d.squeeze(0), self.p_w_z) # doc2topic att [7, 50] * topic att [50, 2875]
        r_z_d2 = torch.bmm(p_z_d, topic_emb_updated)  # [1, 7, 768]

        r_z_d1_sig = activation1(r_z_d1)
        r_z_d1_gat = activation2(r_z_d1) # leaky relu
        r_z_d2_sig = activation1(r_z_d2)
        r_z_d2_gat = activation2(r_z_d2) # leaky relu

        r_z_d = r_z_d1_gat*r_z_d2_sig + r_z_d2_gat*r_z_d1_sig
        
        r = torch.cat((r_z_d, doc_rep), 2) #
        # r = torch.cat((r_z_d1 * r_z_d2, query),2)
        r = self.dropout(r)

        trm_output = self.trm(r)
        output = self.dropout(trm_output)
        logits = self.classifier(output) #softmax
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_dist(
                    logits, labels, self.num_labels, p_w_z_d, distribution_labels)

        # (loss), scores, (hidden_states), (attentions)
        return Normalized_Output(logits=logits, loss=loss)

    def loss_reco(self, logits, labels, num_labels, p_w_z, reconstruct_labels):
        ce = nn.CrossEntropyLoss()
        bce = nn.BCELoss()
        reconstruction_loss = bce(p_w_z, reconstruct_labels)
        classification_loss = ce(logits.view(-1, num_labels), labels.view(-1))

        loss_for_training = classification_loss * reconstruction_loss
        return loss_for_training

    def loss_dist(self, logits, labels, num_labels, p_w_z, distribution_labels):
        ce = nn.CrossEntropyLoss()
        kld = nn.KLDivLoss(reduction='batchmean')

        p_w_z = torch.log(p_w_z)
        distribution_loss = kld(p_w_z, distribution_labels)
        classification_loss = ce(logits.view(-1, num_labels), labels.view(-1))
        loss_for_training = classification_loss * 2
        return loss_for_training


class DialogueTRM_Hierarchical_with_topic_hard_1(nn.Module):

    def __init__(self, num_labels, bert_path, vocab, tokenizer, topic_num):
        self.config = BertConfig()
        super(DialogueTRM_Hierarchical_with_topic_hard_1, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(bert_path)

        self.vocab_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(vocab)).to(device)
        self.word_embedding = self.bert.get_input_embeddings().to(device)
        self.p_w_z = F.softmax(torch.rand(
            topic_num, len(self.vocab_ids)), -1).to(device)

        self.doc2topic = ScaledDotProductAttention(dim=self.config.hidden_size)
        self.topic2word = ScaledDotProductAttention(
            dim=self.config.hidden_size)

        word_emb = self.word_embedding(self.vocab_ids)
        self.topic_emb = torch.mm(self.p_w_z, word_emb).detach()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size*2, nhead=12)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*2, self.num_labels)
        self.p_w_z = None

    def get_p_w_z(self):
        return self.p_w_z

    def forward(self, **kwargs):
        intra_encoding = kwargs.get('intra')
        intra_encoding = assign_GPU(intra_encoding)

        tgt_input_ids = kwargs.get('tgt_input_ids')
        tgt_input_ids = assign_GPU(tgt_input_ids)
        # intra_input_ids = kwargs.get('intra_input_ids').to(device)

        labels = kwargs.get('labels', None).to(device)
        fine_tune = kwargs.get('fine_tune', True)
        activation1 = nn.LeakyReLU(0.1)
        activation2 = nn.Sigmoid()

        for param in self.bert.parameters():
            param.requires_grad = fine_tune

        bert_output = self.bert(**intra_encoding, return_dict=True)

        doc_rep = bert_output.last_hidden_state[:, 0, :]  # [7,768]
        doc_rep = doc_rep.unsqueeze(0)

        query = self.word_embedding(tgt_input_ids['input_ids'])  # [7,len,768]
        mask = tgt_input_ids['attention_mask']  # [7,len]
        query = mask.unsqueeze(-1)*query  # [7,len,768]
        query = query.sum(axis=1)/mask.sum(axis=-1).unsqueeze(-1)  # [7,768]
        query = query.unsqueeze(0)  # [1,7,768]

        # topic_emb = self.topic_embedding(self.topic_ids)
        topic_emb = self.topic_emb.unsqueeze(0)  # [1,40,768]

        word_emb = self.word_embedding(self.vocab_ids)
        word_emb = word_emb.unsqueeze(0)  # [1, 1737, 768]
        # print(query.shape, topic_emb.shape,word_emb.shape)

        # r_z_d:[1, 7, 768], p_z_d:[1, 7, 40]
        r_z_d1, p_z_d = self.doc2topic(query, topic_emb, topic_emb)

        dist = torch.distributions.Categorical(p_z_d)
        idx_topic = dist.sample().squeeze(0)  # [7,]
        sampled_topic_embedding = torch.index_select(
            topic_emb, 1, idx_topic)  # [1, 7, 768]

        topic_emb_updated, p_w_z = self.topic2word(
            sampled_topic_embedding, word_emb, word_emb)  # r_w_z:[1, 7, 768] p_w_z:[1, 7, 1737]

        self.p_w_z = p_w_z.squeeze(0)
        for i, idx in enumerate(idx_topic):
            self.topic_emb[idx, :] = topic_emb_updated.squeeze(0)[
                i, :].detach().clone()

        # r_w_z = r_w_z.permute(1,0,2) #[1, 7, 768]
        # word_emb = word_emb.permute(1,0,2) #[1, 1737, 768]
        # query = query.permute(1,0,2) #[1, 7, 768]
        # r_z_d2 = torch.bmm(p_w_z, word_emb) #[1, 7, 768]
        r_z_d1 = sampled_topic_embedding
        r_z_d2 = topic_emb_updated

        r_z_d1_sig = activation1(r_z_d1)
        r_z_d1_gat = activation2(r_z_d1)
        r_z_d2_sig = activation1(r_z_d2)
        r_z_d2_gat = activation2(r_z_d2)

        r_z_d = r_z_d1_gat*r_z_d2_sig + r_z_d2_gat*r_z_d1_sig

        r = torch.cat((r_z_d, doc_rep), 2)
        r = self.dropout(r)

        trm_output = self.trm(r)
        output = self.dropout(trm_output)
        logits = self.classifier(output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        # (loss), scores, (hidden_states), (attentions)
        return Normalized_Output(logits=logits, loss=loss)

class DialogueTRM_Hierarchical_with_topic_hard_with_kl_1(nn.Module):

    def __init__(self, num_labels, bert_path, vocab, tokenizer, topic_num):
        self.config = BertConfig()
        super(DialogueTRM_Hierarchical_with_topic_hard_with_kl_1, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(bert_path)

        self.vocab_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(vocab)).to(device)
        self.word_embedding = self.bert.get_input_embeddings().to(device)
        self.p_w_z = F.softmax(torch.rand(
            topic_num, len(self.vocab_ids)), -1).to(device)

        self.doc2topic = ScaledDotProductAttention(dim=self.config.hidden_size)
        self.topic2word = ScaledDotProductAttention(
            dim=self.config.hidden_size)

        word_emb = self.word_embedding(self.vocab_ids)
        self.topic_emb = torch.mm(self.p_w_z, word_emb).detach()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size*2, nhead=12)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*2, self.num_labels)
        # self.beta = nn.parameter.Parameter(torch.tensor([0.999, 0.001]), requires_grad=True)

    def get_p_w_z(self):
        return self.p_w_z

    def forward(self, **kwargs):
        intra_encoding = kwargs.get('intra')
        intra_encoding = assign_GPU(intra_encoding)

        tgt_input_ids = kwargs.get('tgt_input_ids')
        tgt_input_ids = assign_GPU(tgt_input_ids)
        # intra_input_ids = kwargs.get('intra_input_ids').to(device)

        distribution_labels = kwargs.get(
            'distribution_labels', None).to(device)
        reconstruct_labels = kwargs.get('reconstruct_labels', None).to(device)
        labels = kwargs.get('labels', None).to(device)
        fine_tune = kwargs.get('fine_tune', True)
        activation1 = nn.LeakyReLU(0.1)
        activation2 = nn.Sigmoid()

        for param in self.bert.parameters():
            param.requires_grad = fine_tune

        bert_output = self.bert(**intra_encoding, return_dict=True)

        doc_rep = bert_output.last_hidden_state[:, 0, :]  # [7,768]
        doc_rep = doc_rep.unsqueeze(0)

        query = self.word_embedding(tgt_input_ids['input_ids'])  # [7,len,768]
        mask = tgt_input_ids['attention_mask']  # [7,len]
        query = mask.unsqueeze(-1)*query  # [7,len,768]
        query = query.sum(axis=1)/mask.sum(axis=-1).unsqueeze(-1)  # [7,768]
        query = query.unsqueeze(0)  # [1,7,768]

        # topic_emb = self.topic_embedding(self.topic_ids)
        topic_emb = self.topic_emb.unsqueeze(0)  # [1,40,768]

        word_emb = self.word_embedding(self.vocab_ids)
        word_emb = word_emb.unsqueeze(0)  # [1, 1737, 768]
        # print(query.shape, topic_emb.shape,word_emb.shape)

        # r_z_d:[1, 7, 768], p_z_d:[1, 7, 40]
        r_z_d1, p_z_d = self.doc2topic(query, topic_emb, topic_emb)

        dist = torch.distributions.Categorical(p_z_d)
        idx_topic = dist.sample().squeeze(0)  # [7,]
        sampled_topic_embedding = torch.index_select(
            topic_emb, 1, idx_topic)  # [1, 7, 768]

        topic_emb_updated, p_w_z = self.topic2word(
            sampled_topic_embedding, word_emb, word_emb)  # r_w_z:[1, 7, 768] p_w_z:[1, 7, 1737]

        self.p_w_z = p_w_z.squeeze(0)
        for i, idx in enumerate(idx_topic):
            self.topic_emb[idx, :] = topic_emb_updated.squeeze(0)[
                i, :].detach().clone()

        # r_w_z = r_w_z.permute(1,0,2) #[1, 7, 768]
        # word_emb = word_emb.permute(1,0,2) #[1, 1737, 768]
        # query = query.permute(1,0,2) #[1, 7, 768]
        # r_z_d2 = torch.bmm(p_w_z, word_emb) #[1, 7, 768]
        r_z_d1 = sampled_topic_embedding
        r_z_d2 = topic_emb_updated

        r_z_d1_sig = activation1(r_z_d1)
        r_z_d1_gat = activation2(r_z_d1)
        r_z_d2_sig = activation1(r_z_d2)
        r_z_d2_gat = activation2(r_z_d2)

        r_z_d = r_z_d1_gat*r_z_d2_sig + r_z_d2_gat*r_z_d1_sig

        r = torch.cat((r_z_d, doc_rep), 2)
        r = self.dropout(r)

        trm_output = self.trm(r)
        output = self.dropout(trm_output)
        logits = self.classifier(output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_dist(
                    logits, labels, self.num_labels, self.p_w_z, distribution_labels)
                # loss = self.loss_reco(logits, labels, self.num_labels, self.p_w_z, reconstruct_labels)

        # (loss), scores, (hidden_states), (attentions)
        return Normalized_Output(logits=logits, loss=loss)

    def loss_reco(self, logits, labels, num_labels, p_w_z, reconstruct_labels):
        ce = nn.CrossEntropyLoss()
        bce = nn.BCELoss()
        # print(p_w_z.device)
        # print(reconstruct_labels.device)
        reconstruction_loss = bce(p_w_z, reconstruct_labels)
        classification_loss = ce(logits.view(-1, num_labels), labels.view(-1))

        loss_for_training = classification_loss * reconstruction_loss
        return loss_for_training

    def loss_dist(self, logits, labels, num_labels, p_w_z, distribution_labels):
        ce = nn.CrossEntropyLoss()
        kld = nn.KLDivLoss(reduction='batchmean')
        # print(p_w_z.device)
        # print(reconstruct_labels.device)
        # print(p_w_z.shape,distribution_labels.shape)
        p_w_z = torch.log(p_w_z)
        distribution_loss = kld(p_w_z, distribution_labels)
        classification_loss = ce(logits.view(-1, num_labels), labels.view(-1))
        loss_for_training = classification_loss * distribution_loss
        # print(classification_loss, distribution_loss)
        return loss_for_training

class EncoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_layer_one = nn.Linear(vocab_size, hidden_size[0])
        self.linear_layer_two = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear_layer_three = nn.Linear(hidden_size[1], hidden_size[2])
        
    def forward(self, inputs):
        activation = nn.LeakyReLU()
        hidden_layer_one = activation(self.linear_layer_one(inputs))
        hidden_layer_two = self.dropout(activation(self.linear_layer_two(hidden_layer_one)))
        hidden_layer_three = self.dropout(activation(self.linear_layer_three(hidden_layer_two)))
        return hidden_layer_three


class DecoderModule(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.topics_to_doc = nn.Linear(num_topics, vocab_size)
        self.batch_normalization = nn.BatchNorm1d(vocab_size, affine=False)
        
    def forward(self, inputs):
        log_softmax = nn.LogSoftmax(dim = 1)
        if inputs.shape[0] == 1:
            return log_softmax(self.topics_to_doc(inputs))
        else:
            return log_softmax(self.batch_normalization(self.topics_to_doc(inputs)))

class RepModule(nn.Module):
    def __init__(self, hidden_size, num_topics, dropout):
        super().__init__()
        self.topics_to_rep = nn.Linear(num_topics, hidden_size[2])
        
    def forward(self, inputs):
        return self.topics_to_rep(inputs)

class EncoderToLogNormal(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_mean = nn.Linear(hidden_size[2], num_topics)
        self.linear_var = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_mean = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_var = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        if hidden.shape[0] == 1:
            mean = self.linear_mean(hidden)
            var = 0.5 * self.linear_var(hidden)
        else:
            mean = self.batch_norm_mean(self.linear_mean(hidden))
            var = 0.5 * self.batch_norm_var(self.linear_var(hidden))
        dist = LogNormal(mean, var.exp())
        return dist
        

class EncoderToDirichlet(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_alpha = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_alpha = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        if hidden.shape[0] == 1:
            alpha = self.linear_alpha(hidden).exp().cpu()
        else:
            alpha = self.batch_norm_alpha(self.linear_alpha(hidden)).exp().cpu()
        dist = Dirichlet(alpha)
        return dist


class EncoderToLaplace(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_lambda = nn.Linear(hidden_size[2], num_topics)
        self.linear_k = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_lambda = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_k = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        if hidden.shape[0] == 1:
            loc = self.linear_lambda(hidden)
            scale = 0.5 * self.linear_k(hidden)
        else:
            loc = self.batch_norm_lambda(self.linear_lambda(hidden))
            scale = 0.5 * self.batch_norm_k(self.linear_k(hidden))
        dist = Laplace(loc, scale.exp())
        return dist


class EncoderToGamma(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_k = nn.Linear(hidden_size[2], num_topics)
        self.linear_theta = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_k = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_theta = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        if hidden.shape[0] == 1:
            k = 0.5 * self.linear_k(hidden)
            theta = 0.5 * self.linear_theta(hidden)
        else:
            k = 0.5 * self.batch_norm_k(self.linear_k(hidden))
            theta = 0.5 * self.batch_norm_theta(self.linear_theta(hidden))
        dist = Gamma(k.exp(), theta.exp())
        return dist


class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics, dropout, model_type, beta):
        super().__init__()
        self.encoder = EncoderModule(vocab_size, hidden_size, dropout)
        if model_type == 1:
            self.encoder_to_dist = EncoderToLogNormal(hidden_size, num_topics)
        elif model_type == 2:
            self.encoder_to_dist = EncoderToDirichlet(hidden_size, num_topics)
        elif model_type == 3:
            self.encoder_to_dist = EncoderToLaplace(hidden_size, num_topics)
        elif model_type == 4:
            self.encoder_to_dist = EncoderToGamma(hidden_size, num_topics)
        self.decoder = DecoderModule(vocab_size, num_topics, dropout)
        self.rep = RepModule(hidden_size, num_topics, dropout)
        self.beta = beta
        
    def forward(self, inputs):
        encoder_output = self.encoder(inputs)
        dist = self.encoder_to_dist(encoder_output)
        if self.training:
            dist_to_decoder = dist.rsample().to(inputs.device)
        else:
            dist_to_decoder = dist.mean.to(inputs.device)
        softmax = nn.Softmax(dim = 1)
        dist_to_decoder = softmax(dist_to_decoder)
        reconstructed_documents = self.decoder(dist_to_decoder)
        topic_rep = self.rep(dist_to_decoder)
        return reconstructed_documents, dist, topic_rep
    
    def loss(self, reconstructed, original, posterior): # We need to have NLL Loss as well KLD Loss
        if isinstance(posterior, LogNormal):
            loc = torch.zeros_like(posterior.loc)
            scale = torch.ones_like(posterior.scale)        
            prior = LogNormal(loc, scale)
        elif isinstance(posterior, Dirichlet):
            alphas = torch.ones_like(posterior.concentration) * 0.01
            prior = Dirichlet(alphas)
        elif isinstance(posterior, Laplace):
            loc = torch.zeros_like(posterior.loc)
            scale = torch.ones_like(posterior.scale)
            prior = Laplace(loc, scale)
        elif isinstance(posterior, Gamma):
            concentration = torch.ones_like(posterior.concentration) * 9
            rate = torch.ones_like(posterior.rate) * 0.5
            prior = Gamma(concentration, rate)
            
        NLL = - torch.sum(reconstructed*original) / (reconstructed.shape[0]*reconstructed.shape[1])
        KLD = torch.sum(kl_divergence(posterior, prior).to(reconstructed.device)) / (reconstructed.shape[0]*reconstructed.shape[1])
        loss_for_training = NLL + self.beta * KLD
        return NLL, KLD, loss_for_training

class DialogueTRM_Hierarchical_with_topic_NTM(nn.Module):
    
    def __init__(self, num_labels, bert_path, vocab, tokenizer, topic_num, model_type):
        self.config = BertConfig()
        super(DialogueTRM_Hierarchical_with_topic_NTM, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(bert_path)
        self.vae = VAE(len(vocab), [512, 256, self.config.hidden_size], topic_num, 0.2, model_type, 3)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size * 2, nhead=12)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size * 2, self.num_labels)
    
    def get_p_w_z(self):
        return self.p_w_z

    def forward(self, **kwargs):
        intra_encoding = kwargs.get('intra')
        intra_encoding = assign_GPU(intra_encoding)

        reconstruct_labels = kwargs.get('reconstruct_labels',None).to(device)
        labels = kwargs.get('labels', None).to(device)
        fine_tune = kwargs.get('fine_tune',True)
        activation1 = nn.LeakyReLU(0.1)
        activation2 = nn.Sigmoid()

        for param in self.bert.parameters():
            param.requires_grad = fine_tune
        
        bert_output = self.bert(**intra_encoding, return_dict=True)
        
        query = bert_output.last_hidden_state[:,0,:] 
        query = query.unsqueeze(0) #[1,7,768]

        out, posterior, dist_to_decoder = self.vae(reconstruct_labels)
        dist_to_decoder = dist_to_decoder.unsqueeze(0)
        r = torch.cat((dist_to_decoder, query),2)
        # r = torch.cat((r_z_d1 * r_z_d2, query),2)
        r = self.dropout(r)

        trm_output = self.trm(r)
        output = self.dropout(trm_output)
        logits = self.classifier(output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_dist(logits,labels, self.num_labels, out, reconstruct_labels, posterior)

        return Normalized_Output(logits = logits, loss = loss)  # (loss), scores, (hidden_states), (attentions)     

    def loss_dist(self, logits, labels, num_labels, out, reconstruct_labels, posterior):
        ce = nn.CrossEntropyLoss()
        # print(p_w_z.device)
        # print(reconstruct_labels.device)
        # print(p_w_z.shape,distribution_labels.shape)
        NLL, KLD, loss_for_vae = self.vae.loss(out, reconstruct_labels, posterior)
        classification_loss = ce(logits.view(-1, num_labels), labels.view(-1))
        loss_for_training = classification_loss * loss_for_vae
        # print(classification_loss, loss_for_vae)
        return loss_for_training
