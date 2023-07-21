from torch.distributions import distribution
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report, confusion_matrix
from scipy.stats import pearsonr
import torch
import random
import pickle
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
#import seaborn as sns
#import matplotlib.pyplot as plt
#import wandb

def train_seq_model(train_data, test_data, best_s1, best_s2, model, optimizer, scheduler, fine_tune, epoch_i):
    model.train()
    # For each batch of training data...
    t0 = time.time()
    tmp_loss = 0
    total_loss = 0
    l = len(train_data) - 1
    log_step = int(l*0.1)
    eval_step = int(l*1)
    wandb_log_list = []
    step = 0
    for batch in tqdm(train_data):
        # Progress update every 40 batches.
        batch['fine_tune'] = fine_tune
        if step % log_step == 0 and not step == 0:
            elapsed        = format_time(time.time() - t0)
            avg_train_loss = tmp_loss / log_step
            tmp_loss = 0
            print('  Batch {:>5,}. of {:>5,} train_loss = {:>5,} Elapsed: {:}.'.format(step, len(train_data), round(avg_train_loss, 4), elapsed))
        
        if step % eval_step == 0 and not step == 0:
            print("")
            print("Running Validation...")
            best_s1, best_s2, wandb_log_list_tmp = eval_seq_model(test_data, model, best_s1, best_s2, epoch_i)
            wandb_log_list += wandb_log_list_tmp

        model.zero_grad()
        output = model(**batch)
        loss, logits = output.loss, output.logits
        tmp_loss += loss.item()
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
    random.shuffle(train_data)
        
    avg_train_loss = total_loss / len(train_data)
    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    wandb_log_list += [{"Average training loss": avg_train_loss}]
    wandb_log_list += [{"Training epcoh took": time.time() - t0}]
    
    return best_s1, best_s2, wandb_log_list

def eval_seq_model(dataset, model, best_s1, best_s2, epoch_i):
    model.eval()
    # For each batch of training data...
    t0 = time.time()
    total_loss = 0
    preds = []
    label_ids = []
    wandb_log_list = []
    for batch in dataset:
        labels = batch['labels']
        with torch.no_grad():
            logits = model(**batch).logits
        num_lables = logits.shape[-1]
        tmp_logits = logits
        logits = logits.view(-1,num_lables)
        logits = logits.detach().cpu().numpy()
        preds.append(logits)
        label_ids.append(labels)
    preds = np.concatenate(preds)
    label_ids = np.concatenate(label_ids)
    
    if num_lables == 1:
        s1, s2 = flat_pear(preds, label_ids)
        name1 = 'MAE'
        name2 = 'R'
    else:
        s1, s2 = flat_accuracy(preds, label_ids)
        name1 = 'ACC'
        name2 = 'F1'
    if best_s2 is None or best_s2 < s2:
        best_s1 = s1
        best_s2 = s2
        # p_w_z = model.get_p_w_z().detach().cpu().numpy()
        # np.savetxt('p_w_z.txt',p_w_z)
        # pickle.dump(model,open('model.pkl','wb'))
    print("  "+ name1 +": {0:.4f}".format(s1))
    print("  "+ name2 +": {0:.4f}".format(s2))
    print("  best "+ name1 +": {0:.4f}".format(best_s1))
    print("  best "+ name2 +": {0:.4f}".format(best_s2))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    wandb_log_list += [{"best "+ name1: best_s1}]
    wandb_log_list += [{"best "+ name2: best_s2}]
    wandb_log_list += [{name1: s1}]
    wandb_log_list += [{name2: s2}]
    
    return best_s1, best_s2, wandb_log_list

def prepare_seq_data(dat, ids, tokenizer, vocab, config):
    win_cont = config['win_cont']
    batch_size = config['batch_size']
    vocab_ids = tokenizer.convert_tokens_to_ids(vocab)
    vocab_map = {}
    for i, id in enumerate(vocab_ids):
        vocab_map[id] = i
    dataset = []
    
    for t, vid in enumerate(ids):
        sample = pd.DataFrame(dat.loc[vid].to_dict())
        context = []
        speaker = []
        target = []
        labels = []
        context_intra = []

        count = 0
        for index, row in sample.iterrows():
        
            if count % batch_size == 0 and not count == 0:
                batch = {}
                batch['tgt_input_ids'] = tokenizer(target, add_special_tokens=False, padding=True, return_tensors="pt")
                batch['intra_input_ids'] = tokenizer(target, context_intra, add_special_tokens=True, padding=True, return_tensors="pt")
                batch['intra'] = tokenizer(target, context_intra, padding='longest', truncation='only_second', max_length=300, return_tensors="pt")
                batch['reconstruct_labels'] = get_reconstruct_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
                batch['distribution_labels'] = get_distribution_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
                
                batch['labels'] = torch.tensor(labels)
                dataset.append(batch)
                target = []
                context_intra = []
                labels = []
                
                count = 0
            
            target_sent = row['sentences']
            label = row['labels']
            s = row['speakers']
            intra = ' '.join([context[i] for i,sp in enumerate(speaker) if s == sp])

            target.append(target_sent)
            labels.append(label)
            context_intra.append(intra)
            
            context.append(target_sent)
            speaker.append(s)
            
            if len(context) > win_cont:
                context.pop(0)
                speaker.pop(0)
            
            count += 1
        
        if count != 0:
            batch = {}
            batch['tgt_input_ids'] = tokenizer(target, add_special_tokens=False, padding=True, return_tensors="pt")
            batch['intra_input_ids'] = tokenizer(target, context_intra, add_special_tokens=True, padding=True, return_tensors="pt")
            batch['intra'] = tokenizer(target, context_intra, padding='longest', truncation='only_second', max_length=300, return_tensors="pt")
            batch['reconstruct_labels'] = get_reconstruct_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
            batch['distribution_labels'] = get_distribution_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
            batch['labels'] = torch.tensor(labels)
            dataset.append(batch)
        
    return dataset

def prepare_seq_data_attack(dat, ids, tokenizer, vocab, config):
    win_cont = config['win_cont']
    batch_size = config['batch_size']
    vocab_ids = tokenizer.convert_tokens_to_ids(vocab)
    vocab_map = {}
    for i, id in enumerate(vocab_ids):
        vocab_map[id] = i
    dataset = []
    attack_dataset = []
    
    for t, vid in enumerate(ids):
        sample = pd.DataFrame(dat.loc[vid].to_dict())
        context = []
        speaker = []
        target = []
        labels = []
        context_intra = []
        speaker_set = set()
        uttr_list = []
        #print(f'vid:{vid}')
        count = 0
        for index, row in sample.iterrows():
        
            if count % batch_size == 0 and not count == 0:
                batch = {}
                batch['tgt_input_ids'] = tokenizer(target, add_special_tokens=False, padding=True, return_tensors="pt")
                batch['intra_input_ids'] = tokenizer(target, context_intra, add_special_tokens=True, padding=True, return_tensors="pt")
                batch['intra'] = tokenizer(target, context_intra, padding='longest', truncation='only_second', max_length=300, return_tensors="pt")
                batch['reconstruct_labels'] = get_reconstruct_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
                batch['distribution_labels'] = get_distribution_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
                
                batch['labels'] = torch.tensor(labels)
                dataset.append(batch)
                target = []
                context_intra = []
                labels = []
                
                count = 0

            target_sent = row['sentences']
            
            label = row['labels']
            s = row['speakers']
            uttr_tmp = f'target_sent:{target_sent}\tlabel:{label}\tspeakers:{s}'
            uttr_list += [uttr_tmp]
            speaker_set.add(s)
            
            intra = ' '.join([context[i] for i,sp in enumerate(speaker) if s == sp])
            #print('context', context)
            target.append(target_sent)
            labels.append(label)
            context_intra.append(intra)
            
            context.append(target_sent)
            speaker.append(s)
            
            if len(context) > win_cont:
                context.pop(0)
                speaker.pop(0)
            
            count += 1

            #attack
            attack_dataset_text = '######^^^^^^'.join([target_sent, intra])
            attack_dataset += [(attack_dataset_text, label)]
            
        if count != 0:
            batch = {}
            batch['tgt_input_ids'] = tokenizer(target, add_special_tokens=False, padding=True, return_tensors="pt")
            batch['intra_input_ids'] = tokenizer(target, context_intra, add_special_tokens=True, padding=True, return_tensors="pt")
            batch['intra'] = tokenizer(target, context_intra, padding='longest', truncation='only_second', max_length=300, return_tensors="pt")
            batch['reconstruct_labels'] = get_reconstruct_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
            batch['distribution_labels'] = get_distribution_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
            batch['labels'] = torch.tensor(labels)
            dataset.append(batch)
        #if len(speaker_set) > 2:
        #    print('\n'.join(uttr_list))
        
    return dataset, attack_dataset, vocab_ids, vocab_map

def train_rnd_model(train_dataloader, validation_dataloader, best_s1, best_s2, model, optimizer, scheduler):
    model.train()
    # For each batch of training data...
    t0 = time.time()
    tmp_loss = 0
    total_loss = 0
    l = len(train_dataloader) - 1
    log_step = int(l*0.1)
    eval_step = int(l*1)
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % log_step == 0 and not step == 0:
            elapsed        = format_time(time.time() - t0)
            avg_train_loss = tmp_loss / log_step
            tmp_loss = 0
            print('  Batch {:>5,}. of {:>5,} train_loss = {:>5,} Elapsed: {:}.'.format(step, len(train_dataloader), round(avg_train_loss, 4), elapsed))
        
        if step % eval_step == 0 and not step == 0:
            print("")
            print("Running Validation...")
            best_s1, best_s2 = eval_rnd_model(validation_dataloader, model, best_s1, best_s2)
        
        batch_encoding_conve = merge_encodings(*batch[:3])
        batch_encoding_intra = merge_encodings(*batch[3:6])
        batch_encoding_inter = merge_encodings(*batch[6:9])
        batch_encoding_tgt = merge_encodings(*batch[9:12])
        batch_encoding_label = batch[-1]
        kwargs = {'conve':batch_encoding_conve, 
                  'intra':batch_encoding_intra, 
                  'inter':batch_encoding_inter, 
                  'tgt':batch_encoding_tgt,
                  'labels':batch_encoding_label}
        
        model.zero_grad()
        output = model(**kwargs)
        loss, logits = output.loss, output.logits
        tmp_loss += loss.item()
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    avg_train_loss = total_loss / len(train_dataloader)
    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    return best_s1, best_s2

def eval_rnd_model(validation_dataloader, model, best_s1, best_s2):
    model.eval()
    # For each batch of training data...
    t0 = time.time()
    total_loss = 0
    preds = []
    label_ids = []
    for batch in validation_dataloader:
        batch_encoding_conve = merge_encodings(*batch[:3])
        batch_encoding_intra = merge_encodings(*batch[3:6])
        batch_encoding_inter = merge_encodings(*batch[6:9])
        batch_encoding_tgt = merge_encodings(*batch[9:12])
        batch_encoding_label = batch[-1]
        kwargs = {'conve':batch_encoding_conve, 
                  'intra':batch_encoding_intra, 
                  'inter':batch_encoding_inter, 
                  'tgt':batch_encoding_tgt,
                  'labels':batch_encoding_label}
        
        with torch.no_grad():
            logits = model(**kwargs).logits
        num_lables = logits.shape[-1]
        logits = logits.view(-1,num_lables)
        logits = logits.detach().cpu().numpy()
        preds.append(logits)
        label_ids.append(batch_encoding_label)
    preds = np.concatenate(preds)
    label_ids = np.concatenate(label_ids)
    
    if num_lables == 1:
        s1, s2 = flat_pear(preds, label_ids)
        name1 = 'MAE'
        name2 = 'R'
    else:
        s1, s2 = flat_accuracy(preds, label_ids)
        name1 = 'ACC'
        name2 = 'F1'
    if best_s2 is None or best_s2 < s2:
        best_s1 = s1
        best_s2 = s2
        pickle.dump(model,open('model.pkl','wb'))
    print("  "+ name1 +": {0:.4f}".format(s1))
    print("  "+ name2 +": {0:.4f}".format(s2))
    print("  best "+ name1 +": {0:.4f}".format(best_s1))
    print("  best "+ name2 +": {0:.4f}".format(best_s2))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    return best_s1, best_s2

def prepare_rnd_data(dat, ids, tokenizer, config):
    win_cont = config['win_cont']
    target = []
    labels = []
    context_intra = []
    context_inter = []
    context_conve = []
    # count_0 = 0
    # count_other = 0
    
    for t, vid in enumerate(ids):
        sample = pd.DataFrame(dat.loc[vid].to_dict())
        context = []
        speaker = []

        for index, row in sample.iterrows():
            target_sent = row['sentences']
            label = row['labels']
            s = row['speakers']

            context.append(target_sent)
            speaker.append(s)

            if len(context) > win_cont:
                context.pop(0)
                speaker.pop(0)
            
            context_r = context[::-1]
            
            conve = ' '.join(context_r)
            intra = ' '.join([context_r[i] for i,sp in enumerate(speaker) if s == sp])
            inter = ' '.join([context_r[i] for i,sp in enumerate(speaker) if s != sp])
            
            target.append(target_sent)
            labels.append(label)
            context_conve.append(conve)
            context_intra.append(intra)
            context_inter.append(inter)
            
    print('tokenizing ...')
    encoding_conve = tokenizer(target, context_conve, padding='max_length', truncation='only_second', max_length=300, return_tensors="pt")
    encoding_intra = tokenizer(target, context_intra, padding='max_length', truncation='only_second', max_length=300, return_tensors="pt")
    encoding_inter = tokenizer(target, context_inter, padding='max_length', truncation='only_second', max_length=300, return_tensors="pt")
    encoding_tgt = tokenizer(target, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    encoding_label = torch.tensor(labels)

    input_ids_conve, token_type_ids_conve, attention_mask_conve = split_encodings(encoding_conve)
    input_ids_intra, token_type_ids_intra, attention_mask_intra = split_encodings(encoding_intra)
    input_ids_inter, token_type_ids_inter, attention_mask_inter = split_encodings(encoding_inter)
    input_ids_tgt, token_type_ids_tgt, attention_mask_tgt = split_encodings(encoding_tgt)
    
    dataset = TensorDataset(input_ids_conve, token_type_ids_conve, attention_mask_conve,
                            input_ids_intra, token_type_ids_intra, attention_mask_intra,
                            input_ids_inter, token_type_ids_inter, attention_mask_inter,
                            input_ids_tgt, token_type_ids_tgt, attention_mask_tgt,
                            encoding_label)

    return dataset

def get_tgt_input_ids(target, tokenizer):
    tgt_input_ids = tokenizer()

def get_distribution_label(input_ids, shape, vocab_map):
    oritional = torch.zeros(shape)+(1/len(vocab_map)*100)
    for i, input_id in enumerate(input_ids):
        for id in input_id:
            id = int(id)
            if id in vocab_map:
                pos = vocab_map[id]
                oritional[i][pos] += 1
    total = oritional.sum(dim=1).unsqueeze(1)
    oritional = torch.div(oritional,total)

    return oritional 

def get_reconstruct_label(input_ids, shape, vocab_map):
    oritional = torch.zeros(shape)
    for i, input_id in enumerate(input_ids):
        for id in input_id:
            id = int(id)
            if id in vocab_map:
                pos = vocab_map[id]
                oritional[i][pos] = 1
    return oritional 

def merge_encodings(input_ids, token_type_ids, attention_mask):
    output = {'input_ids' : input_ids, 
              'token_type_ids' : token_type_ids, 
              'attention_mask' : attention_mask}
    return output

def split_encodings(tokenizer_encoding):
    input_ids = tokenizer_encoding['input_ids']
    token_type_ids = tokenizer_encoding['token_type_ids']
    attention_mask = tokenizer_encoding['attention_mask']
    return input_ids, token_type_ids, attention_mask

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    acc = accuracy_score(labels_flat,pred_flat)
    f1 = f1_score(labels_flat,pred_flat,average='weighted')
    print(classification_report(labels_flat,pred_flat,digits=4))

    return acc, f1

# Function to calculate the accuracy of our predictions vs labels
def flat_pear(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    mae = round(mean_absolute_error(labels, preds),4)
    pear = round(pearsonr(preds, labels)[0], 4)
    return mae, pear

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
 
def prepare_seq_data_without_label1(dat, ids, tokenizer, vocab, config):
    win_cont = config['win_cont']
    batch_size = config['batch_size']
    vocab_ids = tokenizer.convert_tokens_to_ids(vocab)
    vocab_map = {}
    for i, id in enumerate(vocab_ids):
        vocab_map[id] = i
    dataset = []
    
    for t, vid in enumerate(ids):
        sample = pd.DataFrame(dat.loc[vid].to_dict())
        context = []
        speaker = []
        target = []
        labels = []
        context_intra = []

        count = 0
        for index, row in sample.iterrows():
            if int(row['labels']) == 0:
                print(row['labels'])
                continue 
            if count % batch_size == 0 and not count == 0:
                batch = {}
                batch['tgt_input_ids'] = tokenizer(target, add_special_tokens=False, padding=True, return_tensors="pt")

                batch['intra_input_ids'] = tokenizer(target, context_intra, add_special_tokens=True, padding=True, return_tensors="pt")

                batch['intra'] = tokenizer(target, context_intra, padding='longest', truncation='only_second', max_length=300, return_tensors="pt")
                batch['reconstruct_labels'] = get_reconstruct_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
                batch['distribution_labels'] = get_distribution_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
                
                batch['labels'] = torch.tensor(labels)
                dataset.append(batch)
                target = []
                context_intra = []
                labels = []
                
                count = 0
            
            target_sent = row['sentences']

            label = row['labels']
            s = row['speakers']

            intra = ' '.join([context[i] for i,sp in enumerate(speaker) if s == sp])

            target.append(target_sent)
            labels.append(label)
            context_intra.append(intra)
            
            context.append(target_sent)
            speaker.append(s)
            
            if len(context) > win_cont:
                context.pop(0)
                speaker.pop(0)
            
            count += 1
        
        if count != 0:
            batch = {}
            batch['tgt_input_ids'] = tokenizer(target, add_special_tokens=False, padding=True, return_tensors="pt")
            batch['intra_input_ids'] = tokenizer(target, context_intra, add_special_tokens=True, padding=True, return_tensors="pt")
            batch['intra'] = tokenizer(target, context_intra, padding='longest', truncation='only_second', max_length=300, return_tensors="pt")
            batch['reconstruct_labels'] = get_reconstruct_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
            batch['distribution_labels'] = get_distribution_label(batch['intra']['input_ids'], (len(batch['intra']['input_ids']), len(vocab_ids)), vocab_map)
            batch['labels'] = torch.tensor(labels)
            dataset.append(batch)
        
    return dataset

