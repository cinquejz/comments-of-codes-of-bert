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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS
#! 必须参数：input_file  output_file  vocab_file
flags.DEFINE_string("input_file", 'samples.txt',
                    "输入原始文件（或用逗号分隔的文件列表）")

flags.DEFINE_string(
    "output_file", 'output.tfrecords',
    "输入TF示例文件（或用逗号分隔的文件列表）")

flags.DEFINE_string("vocab_file", 'vocab.txt',
                    "训练BERT模型的词汇表文件")

flags.DEFINE_bool(
    "do_lower_case", True,
    "是否忽略大小写（全部转成小写），True则忽略大小写，False则区分大小写")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "mask整个单词还是mask单个WordPiece")

flags.DEFINE_integer("max_seq_length", 128, "最大序列长度")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "每个序列最大的MLM预测数量")

flags.DEFINE_integer("random_seed", 12345, "用于生成数据的随机种子")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "复制输入数据的次数（使用不同的mask）")

flags.DEFINE_float("masked_lm_prob", 0.15, "MLM的概率")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "创建比最大长度短的序列的概率")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""
  """单个训练实例（句子对）"""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    """
    把初始化参数变成字符串
    例如：
      tokens = ['[CLS]', '一', '[MASK]', '三', '。', '[SEP]', '四', '五', '[MASK]', '。', '[SEP]']
      segment_ids = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
      is_random_next = False
      masked_lm_positions = [1, 5]
      masked_lm_labels = ['二', '五']

    返回的s为一个字符串，print出来的结果如下：
      tokens: [CLS] 一 [MASK] 三 。 [SEP] 四 五 [MASK] 。 [SEP]
      segment_ids: 1 1 1 1 1 1 0 0 0 0 0
      is_random_next: False
      masked_lm_positions: 2 8
      masked_lm_labels: 二 五
    """
    s = ""
    # 把tokens变成适合打印或tf.logging的文本，然后用空格隔开
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    # 把segment_ids用空格隔开
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    # 显示is_random_next
    s += "is_random_next: %s\n" % self.is_random_next
    # 把masked_lm_positions用空格隔开
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    # 把masked_lm_labels变成适合打印或tf.logging的文本，然后用空格隔开
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    # 返回__str__的结果
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  """
  从所有'TrainingInstance'中创建TF示例文件
  
  参数:
    instances: 所有示例，每个示例都是TrainingInstance类
    tokenizer: 分词器
    max_seq_length: 最大序列长度
    max_predictions_per_seq: 每个序列的最大MLM预测数
    output_files: 输出文件的路径，可以是列表，文件后缀为`.tfrecords`
  """
  writers = []
  for output_file in output_files:
    # 每个输出文件对应一个TFRecord生成器
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  # 下标从0开始遍历所有示例
  for (inst_index, instance) in enumerate(instances):
    # 把序列中的每个token转成对应的id
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    # 初始化mask序列，取值全为1
    input_mask = [1] * len(input_ids)
    # segment_ids用list保存
    segment_ids = list(instance.segment_ids)
    # 序列不能比最大序列长度max_seq_length长
    assert len(input_ids) <= max_seq_length
    # 当序列长度小于最大长度时，padding（补0）
    while len(input_ids) < max_seq_length:
      # 对input_ids、input_mask和segment_ids补0
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
    # 此时上面三个的长度要等于最大长度
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # masked_lm_positions用list保存
    masked_lm_positions = list(instance.masked_lm_positions)
    # 把masked_lm_labels转成对应的id序列
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    # MLM权重为1.0
    masked_lm_weights = [1.0] * len(masked_lm_ids)
    # 当MLM预测数量小于最大预测数量时，padding
    while len(masked_lm_positions) < max_predictions_per_seq:
      # 对位置、id和权重进行padding
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)
    # 如果is_random_next为True，为1，否则为0
    next_sentence_label = 1 if instance.is_random_next else 0
    # 把上面的变量存在有序字典features中
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])
    # 把features转成Example结构，类似于字典保存为json
    """
    例如：
      input_ids = [1, 2, 3]
      input_mask = [1, 1, 1]

    features {
      feature {
        key: "input_ids"
        value {
          int64_list {
            value: 1
            value: 2
            value: 3
          }
        }
      }
      feature {
        key: "input_mask"
        value {
          int64_list {
            value: 1
            value: 1
            value: 1
          }
        }
      }
    }
    """
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    # 把tf_example序列化为字符串，写入第writer_index个TFRecord文件
    writers[writer_index].write(tf_example.SerializeToString())
    # 每len(writers)个示例写入同一个TFRecord文件
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1
    # 打印前20个示例的信息，这部分代码可以删掉
    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
  # 关闭每个TFRecord生成器
  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  """
    创建整数特征
    输入values只能是list或np.array，
    且只能是一维，即不能嵌套list或二维及以上的数组
    例如values = [1, 2, 3]
      首先tf.train.Int64List(value=list(values))
      会把values转化成特定的数据结构即Int64List，结果是：
        value: 1
        value: 2
        value: 3
      然后tf.train.Feature()将其转化成Feature结构，类似于字典
        int64_list {
          value: 1
          value: 2
          value: 3
        }
  """
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  # 类似于create_int_feature，在这里是float
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  """
  从源文件创建TrainingInstance

  参数:
    input_files: 输入文件路径，可以是list
    tokenizer: 分词器
    max_seq_length: 最大序列长度
    dupe_factor: 复制输入数据的次数（使用不同的mask）
    short_seq_prob: 
    masked_lm_prob: MLM的概率
    max_predictions_per_seq: 每个序列的最大MLM预测数
    rng: random.Random(seed)
  
  输入文件格式：
    (1) 一行一句话。理想情况下，这些应该是实际的句子，而不是整段或任意的文本跨度。
      （因为我们在“下一个句子预测”任务中使用句子边界）。
    (2) 文档之间要有空白行。文档边界是必需的，这样“下一个句子预测”任务不会跨越文档之间。  
  """
  #! 所有文档用一个list保存，一共嵌套三个list
  #! 最外一层是所有文档，第二层是每一个文档，最里面一层是文档中的每一句话（按行）
  #! 最里面的list是分词后的句子
  all_documents = [[]]

  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        # 把每一行文本转换成Unicode编码
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        # 去除两端的空白
        line = line.strip()

        # Empty lines are used as document delimiters
        # 空行用作是文档分隔符
        if not line:
          # 在all_documents中用空list隔开不同文档
          all_documents.append([])
        # 分词
        tokens = tokenizer.tokenize(line)
        if tokens:
          # 把分词后的文本添加到all_documents最后一个list中
          all_documents[-1].append(tokens)

  # Remove empty documents
  # 删除空白文档
  all_documents = [x for x in all_documents if x]
  # 打乱文档顺序
  rng.shuffle(all_documents)
  # 获取词表的所有token
  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  # 对输入数据复制dupe_factor次
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))
  # 打乱示例顺序
  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  """
  从单个文档创建TrainingInstance
  
  参数:
    all_documents: 所有文档
    document_index: 当前文档位置
    max_seq_length: 最大序列长度
    short_seq_prob: 短序列比例
    masked_lm_prob: MLM的概率
    max_predictions_per_seq: 每个序列的最大MLM预测数
    vocab_words: list，词表的所有token
    rng: random.Random(seed)

  输出:
    instances: TrainingInstance
  """
  # 定位当前文档
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  # 减去插入的[CLS], [SEP], [SEP]才是实际的序列长度
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  # 我们*通常*想要填充整个序列，因为我们总是把序列填充到'max_seq_length'
  # 所以短序列通常是浪费计算的。
  # 然而，我们*有时*（例如，short_seq_prob == 0.1 == 10%的比例）
  # 希望使用较短的序列来减小最训练和微调之间的不匹配。
  # 然而，'target_seq_length'只是一个粗略的目标，
  # 而'max_seq_length'是一个硬性限制。
  target_seq_length = max_num_tokens
  # 短序列比例为short_seq_prob
  if rng.random() < short_seq_prob:
    # 短序列长度为[2, max_num_tokens]
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  # 我们不会将文档中的所有标记拼接成一个长序列，然后选择任意的拆分点，
  # 因为这会使下一个句子预测任务变得过于简单。
  # 相反，我们根据用户输入提供的实际“句子”将输入分成“A”和“B”段。
  #? instances中的每一个元素都是 TrainingInstance
  instances = []
  #? 一个句子为一个chunk，用list保存
  current_chunk = []
  #? current_chunk中所有chunk的总长度
  current_length = 0
  i = 0
  while i < len(document):
    #? 一直添加第i个句子到current_chunk
    #? 直到到了document最后一个句子或总长度超过最大长度
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        # `a_end`是`current_chunk`中进入（第一个句子）句子A的句子数量
        a_end = 1
        if len(current_chunk) >= 2:
          # 数量随机选
          a_end = rng.randint(1, len(current_chunk) - 1)
        # 句子A的所有token，用一个list保存
        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])
        # 句子B的所有token，用一个list保存
        tokens_b = []
        # Random next
        # 句子B是否是句子A真实的下一句
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          #! 此时句子B是随机选的，不是句子A的下一句
          is_random_next = True
          # 句子B的目标长度 = 最大长度 - 句子A的长度
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          # 对于大型语料库来说，这很少需要一次以上的迭代。
          # 但是，为了小心起见，我们尽量确保随机文档与我们正在处理的文档不同。
          for _ in range(10):
            # 从语料库中随机选一个文档，但不能是当前文档
            # 用于从中选出一个句子作为下一句
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break
          # 随机选中的文档
          random_document = all_documents[random_document_index]
          # 随机选择一个片段作为句子B
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          # 实际上没有使用这些片段，所以“把它们放回去”，这样它们就不会浪费了。
          # 比如，current_chunk一共有10个句子，现在只从current_chunk中抽了3个句子
          # 剩下的7个句子还会用来生成实例
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          #! 此时句子B是句子A的下一句
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        # 对句子A和句子B进行随机截断，直到整个长度小于最大长度
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1
        #! tokens为 `[CLS] 句子A的token [SEP] 句子B的token [SEP]`
        #! hello everyone! today is sunday
        #! 例如, `[CLS] he ##llo [MASK] ##one [SEP] [MASK] ##od [SEP]`
        tokens = []
        #! 句子A和句子B的句子id, 0表示句子A, 1表示句子B
        #! [CLS]和第一个[SEP]属于句子A, 最后一个[SEP]属于句子B
        #! `[0, 0, 0, 0, 0, 0, 1, 1, 1]`
        segment_ids = []
        #? 在首位添加`[CLS]`
        tokens.append("[CLS]")
        #? 对应句子id添加`0`
        segment_ids.append(0)
        for token in tokens_a:
          #? 添加句子A的token和句子id
          tokens.append(token)
          segment_ids.append(0)
        #? 句子A和句子B之间的`[SEP]`以及id
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          #? 添加句子B的token和句子id
          tokens.append(token)
          segment_ids.append(1)
        #? 以`[SEP]`结尾
        tokens.append("[SEP]")
        segment_ids.append(1)
        #! 创建MLM预测
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        #! 每一个实例转为TrainingInstance
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances

# 创建一个名字为MaskedLmInstance的元组，维度为2
# 分别是index和label，用于保存MLM实例的真实位置和token
# label就是该token
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""
  """
  为MLM任务创建预测
  
  参数:
    tokens: 句子
    masked_lm_prob: MLM的概率
    max_predictions_per_seq: 每个序列的最大预测数量
    vocab_words: 词表的所有token
    rng: random.Random(seed)

  输出: (output_tokens, masked_lm_positions, masked_lm_labels)
    output_tokens: list, mask后的token
    masked_lm_positions: MLM在句子中的位置
    masked_lm_labels: 被mask的token
  """

  cand_indexes = []
  # 把token转化为索引，每个token的索引用一个list保存
  # 假设有一个token序列: ['[CLS]', 'th', '##is', 'a', 'cat', '[SEP]']
  # 如果做全词mask，则得到索引序列: [[1, 2], [3], [4]]
  # 如果不做全词mask，则得到索引序列: [[1], [2], [3], [4]]
  for (i, token) in enumerate(tokens):
    # mask时忽略CLS]和[SEP]
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.

    # 全词mask(Whole Word Masking)是指把对应于同一个原始词的所有wordpiece都mask掉
    # 当一个词已经被切分成wordpiece，第一个token不会有任何标记，其他子token会有一个前缀 ##
    # 因此，每当我们看到 ## token时，我们会将它附加到前面的一组单词索引中。
    # 
    # 全词mask不会改变训练代码——我们仍然独立地预测每个wordpiece，在整个词表中进行softmax
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])
  # 打乱索引list
  rng.shuffle(cand_indexes)
  # tokens本身就是个list，output_tokens是mask后的list
  output_tokens = list(tokens)
  # MLM预测数
  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))
  # MLM的预测
  masked_lms = []
  covered_indexes = set()
  # 随机选择MLM预测的token
  for index_set in cand_indexes:
    # MLM最大预测数为 num_to_predict
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    # 如果选择了全词mask，当前预测数+该词的wordpiece数超过最大预测数
    # 则跳过这个词，不对其进行mask
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      # 80%的概率替换为 [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        # 10%的概率保持不变
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        # 10%的概率随机替换为另一个词
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
      # 替换当前token
      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  # 按token在原句子的位置升序排序
  masked_lms = sorted(masked_lms, key=lambda x: x.index)
  # MLM预测的token的位置和标签
  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  """
  将序列对截断为最大长度序列
    保持中间片段的连续性，每次随机从前或从后删除一个token
  
  参数:
    tokens_a: 句子A
    tokens_b: 句子B
    max_num_tokens: 最大token数量（= 最大序列长度 - 3）
    rng: random.Random(seed) 
  """
  while True:
    # 总长度= 句子A + 句子B
    total_length = len(tokens_a) + len(tokens_b)
    # 总长度小于max_num_tokens就不用截断
    if total_length <= max_num_tokens:
      break
    # 选择更长的句子进行截断
    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    # 截断的长句子长度不能为空
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    # 有时从前面截断，有时从后面截断，以增加随机性并避免偏差。
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  # 初始化分词器
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  # 所有文件的路径名
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)
  # 用于生成随机数
  rng = random.Random(FLAGS.random_seed)
  # 创建预训练实例
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)
  # 保存的文件名
  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)
  # 把训练实例写入到文件
  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
