# -*- coding: utf-8 -*-
import hunspell
import string
import unicodedata
import ast
import numpy as np

from urllib import unquote as unquote_url
import HTMLParser

from django.utils.encoding import smart_str
from Levenshtein import distance as lev_dist

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics

from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

HTMLPARSE = HTMLParser.HTMLParser()

HOBJ = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')

PUNC_DICT = {
    '!': ' exclamationpoint ',
    '"': ' doublequote ',
    '#': ' numbersign ',
    '$': ' dollarsign ',
    '%': ' percentsign ',
    '&': ' amp ',
    '(': ' openparenthesis ',
    ')': ' closeparenthesis ',
    '*': ' astrixsign ',
    '+': ' plussign ',
    ',': ' commasign ',
    '.': ' periodmark ',
    '/': ' fowardslash ',
    ':': ' fullcolon ',
    ';': ' semicolon ',
    '<': ' lessthan ',
    '=': ' equalsign ',
    '>': ' greaterthan ',
    '?': ' questionmark ',
    '@': ' atsign ',
    '[': ' openbracket ',
    '\\': ' backslash ',
    ']': ' closebracket ',
    '^': ' carrotsign ',
    '_': ' underscore ',
    '{': ' opencurly ',
    '|': ' pipebar ',
    '}': ' closecurly ',
    '~': ' tildasign ',
}

WORD_PUNC_LIST = [
    'exclamationpoint',
    'doublequote',
    'numbersign',
    'dollarsign',
    'percentsign',
    'amp',
    'openparenthesis',
    'closeparenthesis',
    'astrixsign',
    'plussign',
    'commasign',
    'periodmark',
    'fowardslash',
    'fullcolon',
    'semicolon',
    'lessthan',
    'equalsign',
    'greaterthan',
    'questionmark',
    'atsign',
    'openbracket',
    'backslash',
    'closebracket',
    'carrotsign',
    'underscore',
    'opencurly',
    'pipebar',
    'closecurly',
    'tildasign',
]


class svm_text(SVC):
#    svm_ = SVC(C=500, kernel='poly', gamma=.01, shrinking=True, probability=False, degree= 10, coef0=2,
#        tol=0.001, cache_size=20000, class_weight=None, verbose=False, max_iter=-1)
    def __init__(self, train_data, C=5, kernel='poly', gamma=.001, degree=10, coef0=2, n_features=10000000,
                 ngram_range=(1, 10), tfidf=False, dfrange=(2, 1.0), probability=False, class_weight=None):
        self.conn = None
        self.is_tfidf = tfidf
        if tfidf:
            self.vectorizer = TfidfVectorizer(stop_words=None, min_df=dfrange[0], max_df=dfrange[1],
                                              max_features=n_features, strip_accents='unicode',
                                              ngram_range=ngram_range, analyzer='word', norm='l2')
        else:
            self.vectorizer = HashingVectorizer(stop_words=None, non_negative=True,
                                                n_features=n_features, strip_accents='unicode',
                                                ngram_range=ngram_range, analyzer='word', norm='l2')
        self.param_set = {'C': str(C), 'kernel': str(kernel), 'gamma': str(gamma),
                          'degree': str(degree), 'coef0': str(coef0), 'n_features': str(n_features)}
        if class_weight == 'auto':
            class_weight = {}
            for item in train_data.target:
                if class_weight.get(item):
                    class_weight.update({item: class_weight[item] + 1.0})
                else:
                    class_weight.update({item: 1.0})
            for key in class_weight:
                class_weight.update({key: 1.0 / class_weight[key]})
        self.class_weight_dict = class_weight
        super(svm_text, self).__init__(C=C, kernel=kernel, gamma=gamma, shrinking=True, probability=probability, degree=degree, coef0=coef0,
                                       tol=0.001, cache_size=20000, class_weight=class_weight, verbose=False, max_iter=-1)
        if self.is_tfidf:
            train_x = self.vectorizer.fit_transform(train_data.data)
        else:
            train_x = self.vectorizer.transform(train_data.data)
        self.fit(train_x, train_data.target)
    def test_data(self, test_data):
        test_x = self.vectorizer.transform(test_data.data)
        predicted_values = self.predict(test_x)
        test_y = test_data.target
        self.score = metrics.f1_score(test_y, predicted_values)
        self.accuracy = metrics.accuracy_score(test_y, predicted_values)
    def guess_text(self, text_text):
        text_x = self.vectorizer.transform([pre_proc(text_text, removestop=False, alwayskeep=True, word_punc=True, unquote=True),])
        return self.predict(text_x)


class svm_multi_label_text(OneVsRestClassifier):
#    svm_ = SVC(C=500, kernel='poly', gamma=.01, shrinking=True, probability=False, degree= 10, coef0=2,
#        tol=0.001, cache_size=20000, class_weight=None, verbose=False, max_iter=-1)
    def __init__(self, train_data,  C=None, n_features=10000000, loss='l2', penalty='l1',
                 ngram_range=(1, 10), tfidf=False, dfrange=(2, 1.0), dual=True, tol=1e-4):
        self.conn = None
        self.is_tfidf = tfidf
        if tfidf:
            self.vectorizer = TfidfVectorizer(stop_words=None, min_df=dfrange[0], max_df=dfrange[1],
                                              max_features=n_features, strip_accents='unicode',
                                              ngram_range=ngram_range, analyzer='word')
        else:
            self.vectorizer = HashingVectorizer(stop_words=None, non_negative=True,
                                                n_features=n_features, strip_accents='unicode',
                                                ngram_range=ngram_range, analyzer='word')
        self.param_set = {'C': str(), 'kernel': str(), 'gamma': str(),
                          'degree': str(), 'coef0': str(), 'n_features': str(n_features)}
        super(svm_multi_label_text, self).__init__(LinearSVC(C=C, loss=loss, penalty=penalty,
                                                             dual=(False if penalty == 'l1' else dual), tol=tol))
        if self.is_tfidf:
            train_x = self.vectorizer.fit_transform(train_data.data)
        else:
            train_x = self.vectorizer.transform(train_data.data)
        train_y = train_data.target
        self.fit(train_x, train_y)
    def test_data(self, test_data):
        test_x = self.vectorizer.transform(test_data.data)
        predicted_values = self.predict(test_x)
        test_y = test_data.target
        try:
            self.score = metrics.f1_score(test_y, predicted_values)
        except ZeroDivisionError:
            self.score = -0.1
        try:
            self.accuracy = metrics.accuracy_score(test_y, predicted_values)
        except ZeroDivisionError:
            self.accuracy = -0.1
    def guess_text(self, text_text):
        text_x = self.vectorizer.transform([pre_proc(text_text, removestop=False, alwayskeep=True, word_punc=True, unquote=True),])
        return self.predict(text_x)


def pre_proc(in_str, removestop=True, alwayskeep=False, word_punc=False, unquote=False):
    # remove accents, wordify punctuation
    in_str = strip_accents(in_str, wordify=word_punc, unquote=unquote)
    en_stem = EnglishStemmer()
    # tokenize string
    if removestop:  # remove stop words
        tok_list = filter(lambda x: x not in stopwords.words('english'), wordpunct_tokenize(in_str))
    else:
        tok_list = wordpunct_tokenize(in_str)
    new_tok_list = []
    for tok in tok_list:
        if tok not in WORD_PUNC_LIST:
            correct_spell = HOBJ.spell(tok)
            if not correct_spell:
                suggestions = [strip_accents(tmp_sug).lower() for tmp_sug in HOBJ.suggest(tok)]
            else:
                suggestions = []
            if correct_spell or (tok.lower() in suggestions):
                new_tok_list.append(tok)
                tok_stem = en_stem.stem(tok)
                if tok_stem != tok:
                    new_tok_list.append(tok_stem)
            elif len(tok) >= 3:
                tok_sug = None
                lev_perc = .34
                for sug in suggestions:
                    if not tok_sug and tok == sug[1:]:
                        tok_sug = sug
                if not tok_sug:
                    for sug in suggestions:
                        tmp_lev_perc = float(lev_dist(tok, sug)) / float(max(len(tok),len(sug)))
                        if not tok_sug and tmp_lev_perc < lev_perc:
                            tok_sug = sug
                            lev_perc = tmp_lev_perc
                if tok_sug:
                    new_tok_list.append(tok_sug)
                    tok_stem = en_stem.stem(tok_sug)
                    if tok_stem != tok_sug:
                        new_tok_list.append(tok_stem)
                elif alwayskeep:
                    new_tok_list.append(tok)
            elif alwayskeep:
                new_tok_list.append(tok)
        else:
            new_tok_list.append(tok)
    out_str = string.join(new_tok_list, ' ')
    return out_str.lower()


def strip_accents(s, wordify=False, unquote=False):
    if type(s) is str:
        s = ast.literal_eval('u' + repr(s))
    if unquote:
        s = unquote_url(s)
        s = HTMLPARSE.unescape(s)
    if wordify:
        pre_rv = smart_str(''.join((c if c not in '!"#$%&()*+,./:;<=>?@[\\]^_{|}~' else PUNC_DICT[c] \
                                    for c in unicodedata.normalize('NFD', s) \
                                    if unicodedata.category(c) != 'Mn' and c not in '\'`')))
    else:
        pre_rv = smart_str(''.join((c if c not in '!"#$%&()*+,./:;<=>?@[\\]^_{|}~' else ' '\
                                    for c in unicodedata.normalize('NFD', s) \
                                    if unicodedata.category(c) != 'Mn' and c not in '\'`')))
    rv = ''
    for c in pre_rv:
        if repr(c).__contains__(r'\x'):
            rv += repr(c)[2:-1]
        else:
            rv += c
    return rv


def bunch_data(data_bundle, fav_number=None):
    new_data = []
    new_targets = []
    for i in range(len(data_bundle.data)):
        if data_bundle.data[i] not in new_data:
            new_data.append(data_bundle.data[i])
            if fav_number is None:
                new_targets.append((unicode(data_bundle.target[i]),))
            else:
                new_targets.append(unicode(data_bundle.target[i]))
        else:
            for j in range(len(new_data)):
                if data_bundle.data[i] == new_data[j] and fav_number is None:
                    new_targets[j] += (unicode(data_bundle.target[i]),)
                elif data_bundle.data[i] == new_data[j] and \
                        abs(float(data_bundle.target[i]) - fav_number) < abs(float(new_targets[j]) - fav_number):
                    new_targets[j] = unicode(data_bundle.target[i])
    return Bunch(data=new_data, target=np.array(new_targets, dtype=np.float64), target_names=[])
