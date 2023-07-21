import sys
import time
import os
import torch
import yaml
from utils.yaml_util import load_yaml
from utils.map_util import model_type_dict
conf = load_yaml(sys.argv[1])
sys.path.insert(0, '/data/FATRER/textattack')
sys.path.insert(1, os.getcwd())

model_conf = conf['model']
attack_conf = conf['attack']
base_model = model_conf['base_model']
from models.attack_model import *
import textattack
from textattack import Attacker
from textattack import AttackArgs
import random
import pickle
from models.DialogueTransformer_utils import train_seq_model, eval_seq_model, prepare_seq_data_attack
from models.DialogueTransformer import *
from models.DialogueTransformer import DialogueTRM_Hierarchical, DialogueTRM_Hierarchical_with_topic_soft_with_kl_1, DialogueTRM_Hierarchical_with_topic_hard_with_kl_1
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW

log_path_name = f'logs/{base_model}_{str(int(time.time()))}'
os.makedirs(log_path_name, 0o777)

random.seed(model_conf['seed'])
torch.manual_seed(model_conf['seed'])
torch.cuda.manual_seed_all(model_conf['seed'])


# DEVICE
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# DATASET
print('Loading dataset ...')
dat, num_labels, train_ids, test_ids = pickle.load(
    open(model_conf['dataset_path'], 'rb'))
vocab = pickle.load(open(model_conf['vocab_path'], 'rb'))
tokenizer = AutoTokenizer.from_pretrained(model_conf['bert_path'])
print('Preparing training data ...')
train_data, train_data_attack_dataset, vocab_ids, vocab_map = prepare_seq_data_attack(
    dat, train_ids, tokenizer, vocab, model_conf)
print('Preparing testing data ...')
test_data, test_data_attack_dataset, vocab_ids, vocab_map = prepare_seq_data_attack(
    dat, test_ids, tokenizer, vocab, model_conf)
attack_dataset = textattack.datasets.Dataset(test_data_attack_dataset[:])
print('Initiating model ...')

# INIT ATTACK
if attack_conf['if_attack']:
    attack_tokenizer = ERCTokenizer(model_conf['bert_path'], vocab_ids, vocab_map)
    recipe_fun = eval(attack_conf['recipe'])

# INIT MODEL
print(f'base_model {base_model}')
if model_conf.get('model_type'):
    model = eval(base_model)(num_labels=num_labels,
                bert_path=model_conf['bert_path'], vocab=vocab, tokenizer=tokenizer, topic_num=model_conf['topic'], model_type=model_type_dict[model_conf['model_type']])
else:
    model = eval(base_model)(num_labels=num_labels,
                bert_path=model_conf['bert_path'], vocab=vocab, tokenizer=tokenizer, topic_num=model_conf['topic'])
model.to(device)
epochs = model_conf['epochs']
lr = model_conf['lr']
eps = model_conf['eps']
total_steps = len(train_data) * epochs
num_warmup_steps = int(model_conf['warmup_steps_ratio'] * total_steps)
optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
best_s1 = None
best_s2 = None

# TRAIN AND ATTACK
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    best_s1, best_s2, _ = train_seq_model(
        train_data, test_data, best_s1, best_s2, model, optimizer, scheduler, True, epoch_i+1)
    # attack
    if attack_conf['if_attack']: 
        if (epoch_i+1) % attack_conf['attack_epoch'] != 0:
            continue

        model_wrapper = ERCModelWrapper(model, attack_tokenizer)
        attack = recipe_fun.build(model_wrapper)
        
        if attack_conf.get('query_budget'):
            attack_args = AttackArgs(log_to_csv=f"logs/{log_path_name}/epoch_{epoch_i+1}.csv",
                                    query_budget=attack_conf['query_budget'], num_examples=attack_conf['num_examples'], disable_stdout=True, random_seed=attack_conf['attack_random_seed'])
        else:
            attack_args = AttackArgs(log_to_csv=f"logs/{log_path_name}/epoch_{epoch_i+1}.csv",
                                    num_examples=attack_conf['num_examples'], disable_stdout=True, random_seed=attack_conf['attack_random_seed'])

        attacker = Attacker(attack, attack_dataset, attack_args)
        attacker.update_attack_args(parallel=True)
        attack_results = attacker.attack_dataset()
        log_attack_details, process_result = attacker.attack_log_manager.get_log_summary()
        print(log_attack_details)
        print(process_result)
        for items in process_result:
            print(items[0], items[1])

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    #print("")
    #print("Running Validation...")
    #best_s1, best_s2 = eval_seq_model(test_data, model, best_s1, best_s2)

print("")
print("Training complete!")
