import json
import re
import torch
from torch import Tensor

class Text:
    def __init__(self):
        self.tokens = json.load(open('data/latex_tokens.json', 'r'))
        self.map_tokens = dict(zip(self.tokens, range(len(self.tokens))))
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.TOKENIZE_PATTERN = re.compile("(\\\\[a-zA-Z]+)|"+ # \[command name]
                              #"(\{\w+?\})|"+ # {[text-here]} Check if this is needed
                              "((\\\\)*[$-/:-?{-~!\"^_`\[\]])|"+ # math symbols
                              "(\w)|"+ # single letters or other chars
                              "(\\\\)") # \ characters

    def tokenize(self, formula: str):
        """Returns list of tokens in given formula.
        formula - string containing the LaTeX formula to be tokenized
        Note: Somewhat work-in-progress"""
        tokens = re.finditer(self.TOKENIZE_PATTERN, formula)
        tokens = list(map(lambda x: x.group(0), tokens))
        tokens = [x for x in tokens if x is not None and x != ""]
        return tokens

    def int2text(self, x: Tensor):
        return ''.join([self.tokens[i] for i in x if i > self.eos_id])
    
    def text2int(self, formula: str):
        return torch.LongTensor([self.map_tokens[i] for i in self.tokenize(formula)])