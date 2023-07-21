from textattack.attack_recipes import TextBuggerLi2018
from textattack.attack_recipes import TextFoolerJin2019
from textattack.attack_recipes import PWWSRen2019
from textattack.models.wrappers import ERCModelWrapper
from textattack.models.tokenizers import ERCTokenizer
from textattack.loggers import AttackLogManager
from models.textattack_fix import *
AttackLogManager.get_log_summary = get_log_summary