# FATRER
FATRER: Full-Attention Topic Regularizer for Accurate and Robust Conversational Emotion Recognition[[paper]](https://arxiv.org/abs/2307.12221) 
## Framework
Full-attention topic regularizer(FATRER) enables an emotion-related
global view when modeling the local context in a conversation. A
joint topic modeling strategy is introduced to implement regularization from both representation and loss perspectives. To avoid overregularization, FATRER drops the constraints on prior distributions that exist in traditional topic modeling and perform probabilistic approximations based entirely on attention alignment. Experiments show
that FATRER obtain more favorable results than state-of-the-art
models, and gain convincing robustness.

![fater_demo](./images/demo.png)
## Environment
### Prerequisites
- Python 3.9.12
- Pytorch 1.10.1+cu113
``` ruby
  pip instll -r requirements.txt
```
## Benchmark Datasets
- IEMOCAP/MELD/EmoryNLP/EmoryNLP
###  Generalization results on four datasets
![fater_demo](./images/table1.png)
### IEMOCAP
1. FARTER-Multi: 
``` ruby
    # train
    python main.py conf/FATRER_multi.yaml

    #train and conduct attack(U+C) based on PWWS(per 50 epoch):
    python main.py conf/FATRER_multi_pwws_attack.yaml

    #train and conduct attack(U+C) based on TextFooler(per 50 epoch):
    python main.py conf/FATRER_multi_textfooler_attack.yaml

    #train and conduct attack(U+C) based on TextBugger(per 50 epoch):
    python main.py conf/FATRER_multi_textbugger_attack.yaml
```
2. FARTER-Multi(without topic-oriented regularization):
``` ruby
    # train
    python main.py conf/FATRER_multi_wo_topic.yaml
```
3. FARTER-Single: 
``` ruby
    # train
    python main.py conf/FATRER_single.yaml
``` 
4. FARTER-Single(without topic-oriented regularization): 
``` ruby
    #train
    python main.py conf/FATRER_single_wo_topic.yaml
``` 
5. DialTRM(Baseline): 
``` ruby
    #train
    python main.py conf/Baseline.yaml
``` 
6. VAE(topic-oriented)
``` ruby
  #train VAE(Laplace)
  python main.py conf/VAE_Laplace.yaml

  #train VAE(Dirichlet)
  python main.py conf/VAE_Dirichlet.yaml

  #train VAE(Gamma)
  python main.py conf/VAE_Gamma.yaml

  #train VAE(LogNormal)
  python main.py conf/VAE_LogNormal.yaml
``` 
### MELD
1. FARTER-Multi: 
``` ruby
    python main.py conf/FATRER_multi_MELD.yaml
``` 
2. FARTER-Single: 
``` ruby
    python main.py conf/FATRER_single_MELD.yaml
``` 
### EmoryNLP
1. FARTER-Multi: 
``` ruby
    python main.py conf/FATRER_multi_EmoryNLP.yaml
```
2. FARTER-Single: 
``` ruby
    python main.py conf/FATRER_single_EmoryNLP.yaml
```

### DailyDialog
1. FARTER-Multi: 
``` ruby
    python main.py conf/FATRER_multi_DailyDialog.yaml
```
2. FARTER-Single: 
``` ruby
    python main.py conf/FATRER_single_DailyDialog.yaml
```
## Cite us
Cite this paper, if you use FARTER in your research publication.
```
@inproceedings{mao2023fatrer,
  title={FATRER: Full-Attention Topic Regularizer for Accurate and Robust Conversational Emotion Recognition},
  author={Yuzao , Mao and Di, Lu and Xiaojie, Wang and Yang, Zhang},
  booktitle={ECAI},
  pages={0--1},
  year={2023}
}
```

## License
BSD 3-Clause License
