import re
from nltk.corpus import words
word_list = set(words.words())
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import nltk
nltk.download('words')

word_to_number = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
}

def convert_word_to_number(s):
    s = s.split(' ')
    for i, ss in enumerate(s):
        ss = ss.lower()
        if ss in word_to_number:
            s[i] = str(word_to_number[ss])
    return ' '.join(s)

def remove_unit(s):
    result = re.match(r'^([0-9]+\.?[0-9]*)\s+[A-Za-z\s]+$', s, re.DOTALL)
    if result:
        return result.group(1)
    return s

def remove_commas_in_numbers(s):
    return re.sub(r'(?<=\d),(?=\d)', '', s)

def decide_choice(s):
    if len(s) > 0 and s[0] in ['A', 'B', 'C', 'D']:
        return s[0]
    return s

def remove_percent(s):
    if s.endswith('%'):
        return s[:-1]
    return s

def remove_dollar(s):
    if s.startswith('$'):
        return s[1:]
    return s

def remove_century_from_date(s):
    return re.sub(r'(\b\d{2})[/-](\d{2})[/-](\d{2})(\d{2}\b)', r'\1/\2/\4', s)

def remove_bracket(s:str):
    return s.strip('[').strip(']').strip('(').strip(')').strip('{').strip('}')

def decide_yes_or_no(s):
    if s.startswith('Yes') or s.startswith('yes'):
        return 'Yes'
    elif s.startswith('No') or s.startswith('no'):
        return 'No'
    return s

def is_removable_ing(word):
    if word.endswith("ing"):
        root_word = word[:-3]
        if root_word in word_list:
            return True
        elif len(root_word) > 1 and root_word[-1] == root_word[-2] and root_word[:-1] in word_list:
            return True
        elif root_word + 'e' in word_list:
            return True
    return False

def remove_ing(s):
    words = s.split(' ')
    for i, w in enumerate(words):
        if w.endswith("ing"):
            root_word = w[:-3].lower()
            if len(root_word) <= 1:
                continue
            if root_word in word_list:
                words[i] = root_word
            elif len(root_word) > 1 and root_word[-1] == root_word[-2] and root_word[:-1] in word_list:
                words[i] = root_word[:-1]
            elif root_word + 'e' in word_list:
                words[i] = root_word + 'e'
    return ' '.join(words)

def plural2singular(s):
    words = s.split(' ')
    for i, w in enumerate(words):
        words[i] =  lemmatizer.lemmatize(w.lower(), 'n')
    return ' '.join(words)

def fraction2float(s):
    for div_punc in [':', '/']:
        if not div_punc in s:
            continue
        nums = s.split(div_punc)
        if len(nums) != 2:
            continue
        try:
            dividend = float(nums[0])
            divisor = float(nums[1])
            return str(dividend / divisor)
        except:
            continue
    return s

def rectify(task: str, an: str):
    an = an.strip().strip('.')
    funcs = []
    
    if task in ['svqa']:
        funcs.extend([decide_choice])
    elif task in ['tabfact']:
        funcs.extend([decide_yes_or_no])
    elif task in ['chartqa', 'clevr', 'deepform', 'okvqa', 'dvqa', 'gqa', 'infovqa', 'ocrvqa', 'visualmrc', 'vizwiz', 'wikitablequestions']:
        funcs.extend([decide_yes_or_no, convert_word_to_number, remove_commas_in_numbers, remove_unit, remove_percent, remove_bracket, remove_ing, plural2singular, fraction2float])
        
    for f in funcs:
        an = f(an)
    return an.strip()