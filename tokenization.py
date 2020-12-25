# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """Checks whether the casing config is consistent with the checkpoint name."""

  # The casing has to be passed in by the user and there is no explicit check
  # as to whether it matches the checkpoint. The casing information probably
  # should have been stored in the bert_config.json file, but it's not, so
  # we have to heuristically detect it to validate.

  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  model_name = m.group(1)

  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  """把text转换成Unicode编码，假设输入是utf-8"""
  if six.PY3:
    # 在python3版本中
    # 如果text是字符串则返回text
    if isinstance(text, str):
      return text
    # 如果text是bytes则用utf-8解码text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    # 在python2版本中
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""
  """返回以适合打印或tf.logging的方式编码的文本。"""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  """加载词表，用字典保存"""
  # 初始化有序字典
  vocab = collections.OrderedDict()
  index = 0
  """
  tf.gfile.GFile(filename, mode)类似于os的open
  filename是文件路径，mode是以什么方式读写文件
  """
  # 读取词表
  with tf.io.gfile.GFile(vocab_file, "r") as reader:
    while True:
      # 读取每一行，转成Unicode编码
      token = convert_to_unicode(reader.readline())
      # 当token为空时即读完所有行，结束while循环
      if not token:
        break
      # 去除token两侧的空格
      token = token.strip()
      # 给token一个唯一编号index（对应于在词表中的位置）
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  """用vocab把序列的token/id转成对应的编码id/token"""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  # 把token序列转成id序列
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  # 把id序列转成token序列
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  """基本的空格清理和文本分割"""
  # 去除text两侧的空格
  text = text.strip()
  if not text:
    # 若text为空，则返回空list
    return []
  # 按空格分割文本
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""
  """端到端分词"""

  def __init__(self, vocab_file, do_lower_case=True):
    """
    :vocab_file: 词表路径
    :do_lowe_case: 是否区分大小写，默认为True
    """
    # 读取词表，key是token，value是id
    self.vocab = load_vocab(vocab_file)
    # 逆词表，key是id，value是token
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    # 基本分词（按空格）
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    # 更细粒度的WordPiece分词（针对英文）
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    """
    分词，先经过Basic分词，再经过WordPiece分词
    中文文本只需要Basic分词

    对于忽略大小写的英文情况：
      输入：A ship-shipping ship ships shipping-ships.
      输出：['a', 'shi', '##p', '-', 'shi', '##pp',
            '##ing', 'shi', '##p', 'shi', '##ps',
            'shi', '##pp', '##ing', '-', 'shi',
            '##ps', '.']

    中文：
      输入：你是基佬。
      输出：['你', '是', '基', '佬', '。']
    """
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    # 把token序列转成id序列
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    # 把id序列转成token序列
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""
  """基本分词（标点分割、小写等）"""

  def __init__(self, do_lower_case=True):
    """
    构造一个BasicTokenizer

    参数:
      do_lower_case: 输入是否转成小写
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    """
    分词

    对于忽略大小写的英文情况：
      输入：A ship-shipping ship ships shipping-ships.
      输出：['a', 'ship', '-', 'shipping', 'ship', 'ships', 'shipping', '-', 'ships', '.']
    
    中文：
      输入：你是基佬。
      输出：['你', '是', '基', '佬', '。']
    """
    # 把text转成Unicode编码
    # A ship-shipping ship ships shipping-ships.
    # 你是基佬。
    text = convert_to_unicode(text)
    # 清洗文本，去除text中的无意义字符和空格
    # A ship-shipping ship ships shipping-ships.
    # 你是基佬。
    text = self._clean_text(text)
    # 在中文字符前后添加空格
    # A ship-shipping ship ships shipping-ships.
    # ' 你  是  基  佬 。'
    text = self._tokenize_chinese_chars(text)
    # 按空格分割文本
    # ['A', 'ship-shipping', 'ship', 'ships', 'shipping-ships.']
    # ['你', '是', '基', '佬', '。']
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        # 不区分大小写时，全部转成小写
        token = token.lower()
        # 去除重音符号
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    # split_tokens:
    # ['a', 'ship', '-', 'shipping', 'ship',
    # 'ships', 'shipping', '-', 'ships', '.']
    # ['你', '是', '基', '佬', '。']
    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    """
    去除text中的重音符号

    unicodedata.normalize("NFD", text)把text标准化
      可以把字母和声调分开，从而达到去除重音符号的目的
      例如，'é'和'e\u0301'，打印出来都是 é
      但两者是不一样的，长度也不一样
        s1 = 'é'
        s2 = 'e\u0301'
        print(s1 == s2) # False
        print(len(s1)) # 1
        print(len(s2)) # 2
        s1 = unicodedata.normalize("NFD", s1)
        s2 = unicodedata.normalize("NFD", s2)
        print(s1 == s2) # True
        print(len(s1)) # 2
        print(len(s2)) # 2
      经过标准化后两者已经完全一样，且字母和声调被分开
    """
    text = unicodedata.normalize("NFD", text)

    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    """
    拆分文本中的标点符号
    
    英文：
      输入：A ship-shipping ship ships shipping-ships.
      输出：['A ship', '-', 'shipping ship ships shipping', '-', 'ships', '.']

    中文：
      输入：你是基佬。
      输出：['你是基佬', '。']
    """
    # 以字符为单位拆开text
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    """
    在中文字符前后添加空格
    输入：'你是基佬。'
    输出：' 你  是  基  佬 。'
    """
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    """
    检查CP是否为CJK字符的代码点，用于判断一个unicode字符是否中文字符
    通常我们判断一个字符是否为中文都是用4E00-9FA5的范围，这只是基本汉字的编码
    更多的汉字unicode编码范围见https://www.cnblogs.com/straybirds/p/6392306.html
    """

    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    """去除text中的无意义字符和空格"""
    output = []
    for char in text:
      # ord()函数主要用来返回对应字符的ASCII码
      cp = ord(char)
      # codepoint为0的是无意义的字符，0xfffd显示为�
      # 控制字符如\n \t \r
      if cp == 0 or cp == 0xfffd or _is_control(char):
        # print(chr(0xfffd))
        continue
      # 把空格字符替换成" "，否则保留原字符
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""
  """WordPiece分词，比单词更细粒度的分词"""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    """
    参数：
        vocab: 词表
        unk_token: 在词表外的标记符
        max_input_chars_per_word: 每个单词的最大输入长度
    """
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """
    """
    把text分词成WordPiece
    使用贪心的最大正向匹配算法
    例如：
      input = "unaffable"
      output = ["un", "##aff", "##able"]
    参数：
        text: 一个或多个空格分隔的token。这个是经过BasicTokenizer处理的。
    输出：
        分成WordPiece的列表
    """
    # 把text转成unicode编码
    text = convert_to_unicode(text)
    output_tokens = []
    # whitespace_tokenize(text)-->['unaffable']
    for token in whitespace_tokenize(text):
      # ['u', 'n', 'a', 'f', 'f', 'a', 'b', 'l', 'e']
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        # 如果token长度超出指定长度，标记为[UNK]
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      """
      一开是令start=0，令end为chars的长度
      首先看unaffable在不在词表里，
      如果在，那么unaffable就是一个wordpiece，
      如果不在，那么end-1，看unaffabl在不在词表里，
      如果在，把unaffabl添加到sub_tokens中，令start=end，
        剩余的字符串前面加##，##表示是接着前面的
      如果不在，那么end-1...，
      如此类推，直到start=lens(chars)
      """
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  """检查char是不是一个空格字符"""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  # \t、\n和\r在技术上是控制字符，但我们将它们视为空白，因为它们通常被视为空白。
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  # 函数unicodedata.category()返回这个Unicode字符的Category
  # 返回Zs则表示该字符是一个空格
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  """检查char是不是一个控制字符"""
  # 这些在技术上是控制字符，但我们将其视为空白字符。
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  """
  检查char是不是标点符号
  我们将所有非字母/数字ASCII都视为标点符号。
  诸如“^”、“$”和“`”等字符不在Unicode标点符号类中，
  但为了保持一致性，我们无论如何都将它们视为标点符号。
  """
  cp = ord(char)
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False