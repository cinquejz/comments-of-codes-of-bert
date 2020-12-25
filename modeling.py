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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    """
    构建Bert的模型结构BertConfig

    参数:
      vocab_size: 词表大小
      hidden_size: 编码器隐藏层和池化层大小
      num_hidden_layers: 隐藏层层数
      num_attention_heads: 每个注意力层的注意力头数
      intermediate_size: Transformer编码器中间层大小，如全连接层
      hidden_act: 非线性激活函数（函数或字符串）
      hidden_dropout_prob: dropout率
      attention_probs_dropout_prob: 注意力的dropout率
      max_position_embeddings: 模型的最大（输入）序列长度（如512/1024/2048）
      type_vocab_size: 
      initializer_range: 截断正态分布初始化的标准差（用于初始化所有权重）
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    """通过Python字典保存的参数来构造BertConfig"""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    """通过json文件保存的参数来构造BertConfig"""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    """将此实例序列化为Python字典"""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    """将此实例序列化为json"""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """
  """
  BERT模型
    使用例子：
    # 已经被转换为WordPiece token 的id
      input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
      input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
      token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

      config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

      model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

      label_embeddings = tf.get_variable(...)
      pooled_output = model.get_pooled_output()
      logits = tf.matmul(pooled_output, label_embeddings)
      ...    
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    """
    Bert的建立

    参数：
      config: BertConfig实例，存储模型的结构参数
      is_training: 布尔值。True为训练模型，False为验证模型。用来控制是否dropout
      input_ids: 形状为[batch_size, seq_length]的int32 Tensor
      input_mask: （可选的）形状为[batch_size, seq_length]的int32 Tensor
      token_type_ids: （可选的）形状为[batch_size, seq_length]的int32 Tensor
      use_one_hot_embeddings: （可选的）布尔值。
        表示是否使用one-hot词嵌入或者用tf.embedding_lookup()表示嵌入。
      scope: （可选的）变量作用域。默认为"bert"
    """
    # 深拷贝，在函数内部对config修改不会影响传进来的参数config
    config = copy.deepcopy(config)
    if not is_training:
      # 如果不训练模型，关闭dropout
      # hidden_dropout_prob和attention_probs_dropout_prob设为0
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
    # 获取输入的形状，[batch_size, seq_length]
    # 预期维度为2，传入其他维度的输入会报错
    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      # 如果没有指定input_mask，就创建和input_shape形状一样的全1张量
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      # 如果没有指定token_type_ids，就创建和input_shape形状一样的全0张量
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      # 在这里创建的变量将被命名成bert/[tensor_name]
      # 分别有embeddings、encoder和pooler
      with tf.variable_scope("embeddings"):
        # 在这里创建的变量将被命名为bert/embeddings/[tensor_name]
        # Perform embedding lookup on the word ids.
        # 根据词的id随机初始化词嵌入word embedding
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids, # 输入的id向量
            vocab_size=config.vocab_size, # 词表大小
            embedding_size=config.hidden_size, # embedding向量维度
            initializer_range=config.initializer_range, # 随机初始化范围
            word_embedding_name="word_embeddings", # 词嵌入变量的名称
            use_one_hot_embeddings=use_one_hot_embeddings) # 是否使用one-hot

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        # 加上位置embedding和句子embedding，然后进行层归一化和dropout
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output, # 词嵌入
            use_token_type=True, # 添加句子embedding
            token_type_ids=token_type_ids, # 句子向量
            token_type_vocab_size=config.type_vocab_size, # 句子数量，一般为2
            token_type_embedding_name="token_type_embeddings", # 句子embedding变量的名称
            use_position_embeddings=True, # 添加位置embedding
            position_embedding_name="position_embeddings", # 位置embedding变量的名称
            initializer_range=config.initializer_range, # 随机初始化范围
            max_position_embeddings=config.max_position_embeddings, # 最大长度
            dropout_prob=config.hidden_dropout_prob) # dropout率

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        # 将形状为[batch_size, seq_length]的2D mask转换成
        # 形状为[batch_size, seq_length, seq_length]的3D mask，
        # 这将用于计算注意力得分
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        # 堆叠Transformer，sequence_output是最后一层输出
        # 形状为[batch_size, seq_length, hidden_size]
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output, # 输入到模型的embedding
            attention_mask=attention_mask,
            hidden_size=config.hidden_size, # 隐藏层大小
            num_hidden_layers=config.num_hidden_layers, # 隐藏层层数
            num_attention_heads=config.num_attention_heads, # 注意力头数
            intermediate_size=config.intermediate_size, # 中间层大小
            intermediate_act_fn=get_activation(config.hidden_act), # 激活函数
            hidden_dropout_prob=config.hidden_dropout_prob, # dropout率
            attention_probs_dropout_prob=config.attention_probs_dropout_prob, # dropout率
            initializer_range=config.initializer_range, # 随机初始化范围
            do_return_all_layers=True) # 返回所有层

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      
      # pooler将形状为[batch_size, seq_length, hidden_size]的编码序列tensor
      # 转换成形状为[batch_size, hidden_size]的tensor
      # 这对于句子级别（句子对级别）的分类任务来说是必须的，
      # 因为我们需要句子的固定维度表示
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        # pooler的作用就是取出[CLS]对应的隐状态作为句子向量，
        # 我们假设模型是预训练的
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    # 获取[CLS]向量，即句子向量
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    """
    获取编码器的最后一个隐藏层

    输出: 
      形状为[batch_size, seq_length, hidden_size]的浮点型tensor，
      对应于Transformer编码器的最后一层隐藏层
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    # 获取编码器的所有隐藏层
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    """
    获取embedding（例如，输入到Transformer）

    输出：
      形状为[batch_size，seq_length，hidden_size]的浮点张量，
      对应于词嵌入与句子embedding、位置embedding相加后的embedding，
      然后进行层归一化，这是Transformer的输入
    """
    return self.embedding_output

  def get_embedding_table(self):
    # 获取embedding矩阵
    return self.embedding_table


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  """
  高斯误差线性单元（Gaussian Error Linear Unit，GELU）
  这是RELU的平滑版本
  原始论文：https://arxiv.org/abs/1606.08415

  GELU(x) = x * P(X<=x), X~N(miu,sigma^2)
  近似公式：
    GELU(x) = 0.5 * (1 + tanh(obj))
    obj = sqrt(2 / pi) * (x + 0.044715 * x^3)

  参数:
    x: 经过激活函数作用之前的浮点型tensor

  输出:
    经过激活函数作用后的x
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """
  """
  把字符串映射成Python函数，例如，"relu" => tf.nn.relu

  参数:
    activation_string: 激活函数的字符串名字

  输出:
    对应于激活函数的Python函数。
      如果activation_string为None、空或'linear'，将返回None。
      如果activation_string不是一个字符串，将返回activation_string

  引起:
    ValueError: activation_string没有对应于一个已知的激活函数
      仅支持四种激活函数：线性、relu、gelu、tanh
  """
  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  # 我们假设一个非字符串输入已经是一个激活函数，因此我们返回它
  if not isinstance(activation_string, six.string_types):
    # 判断activation_string是否为字符串
    return activation_string

  if not activation_string:
    # 如果activation_string为空或None
    return None
  # 字符串转成小写，即输入的字符串不区分大小写
  act = activation_string.lower()
  if act == "linear":
    # 线性激活函数，即f(x) = x
    return None
  elif act == "relu":
    # relu激活函数
    return tf.nn.relu
  elif act == "gelu":
    # gelu激活函数
    return gelu
  elif act == "tanh":
    # tanh激活函数
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  """
  计算当前变量和检查点变量的交集
  
  参数:
    tvars: 当前可训练的变量，tf.trainable_variables()
    init_checkpoint: 初始的检查点
  
  输出: (assignment_map, initialized_variable_names)
    assignment_map: 字典，相同的变量，key和value都是变量名称本身
    initialized_variable_names: 字典，
      key是变量名称本身以及name+':0'，value都是1
  """
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    # 当前变量名称
    name = var.name
    # 只匹配以数字结尾的变量的名字，如kernel:0
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      # 如果匹配成功，则取第一个括号匹配到的部分
      # 即kernel:01只取kernel
      name = m.group(1)
    # 把该变量添加到字典中
    name_to_variable[name] = var
  # 从检查点初始化的参数，用list保存
  # list中的每个元素为一个元组
  # 第一个为变量名称，第二个为变量值
  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  """
  执行dropout

  参数:
    intput_tensor: 浮点型tensor
    dropout_prob: Python浮点数，dropout的概率

  输出:
    input_tensor经过dropout的一个版本
  """
  if dropout_prob is None or dropout_prob == 0.0:
    # 如果dropout_prob为None或0.0，不执行dropout，返回input_tensor
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  """在tensor的最后一个维度进行层归一化（layer normalization）"""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  """先层归一化，再dropout"""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  """创建一个给定范围的截断正态分布初始器"""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.gather()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  """
  查找id tensor的词嵌入
    作用就是把形状为[batch_size, seq_length]的id tensor
    输出成形状为[batch_size, seq_length, embedding_size]的tensor

  参数:
    input_ids: 存储单词id的int32 tensor，形状为[batch_size, seq_length]
    vocab_size: 整数，词表的大小，要与vocab.txt一致
    embedding_size: 整数，embedding后的向量大小
    initializer_range: 浮点数，embedding随机初始化的范围
    word_embedding_name: 字符串，embedding表的名字，默认为word_embeddings
    use_one_hot_embeddings: 布尔值，
      如果为True，使用one-hot来计算embedding
      如果为False，使用tf.gather()

  输出:
    (output, embedding_table)
    output: 浮点型tensor，形状为[batch_size, seq_length, embedding_size]
    embedding_table: embedding矩阵，形状为[vocab_size, embedding_size]
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  # 此函数假设输入的形状为[batch_size, seq_length, num_inputs]
  # 如果输入是一个形状为[batch_size, seq_length]的2D张量，
  # 我们将其重塑成[batch_size, seq_length, 1]
  # 例如，输入的形状为2*3: [[1, 2, 3],[4, 5, 6]]，将被reshape为
  # [[[1], [2], [3]], [[4], [5], [6]]]，此时形状为2*3*1
  if input_ids.shape.ndims == 2:
    # tf.expand_dims(input, axis=None)
    # 给定一个input，在axis轴处给input增加一个为1的维度
    # axis=[-1]表示最后一个维度
    input_ids = tf.expand_dims(input_ids, axis=[-1])
  # 构造embedding矩阵，形状为[vocab_size, embedding_size]
  embedding_table = tf.get_variable(
      name=word_embedding_name, # 名字
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))
  # 把input_ids打平，压成一维向量[1, 2, 3, 4, 5, 6]
  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    # 把flat_input_ids转成oe-hot，depth为词表大小
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    # [batch_size, vocab_size] * [vocab_size, embedding_size]
    # output => [batch_size, embedding_size]
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    # 从embedding_table中取出第flat_input_ids[k]个向量
    # 放在flat_input_ids的第k个位置上
    # output => [batch_size, embedding_size]
    output = tf.gather(embedding_table, flat_input_ids)
  # 获取input_ids的形状，[batch_size, seq_length, 1]
  input_shape = get_shape_list(input_ids)
  # 将output从[batch_size, embedding_size]
  # reshape成[batch_size, seq_length, embedding_size]
  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  """
  对词嵌入张量进行各种后处理
  模型输入 = 词嵌入 + 句子embedding + 位置embedding

  参数:
    input_tensor: 浮点tensor，形状为[batch_size, seq_length, embedding_size]
    use_token_type: 布尔值，是否添加句子embedding
    token_type_ids: (可选的)int32 tensor，形状为[batch_size, seq_length]
      如果use_token_type为True，则必须指定
    token_type_vocab_size: 整数，token type的个数，一般为2（区分句子A和句子B）
    token_type_embedding_name: 字符串，token_type_ids的embedding矩阵变量的名称
    use_position_embeddings: 布尔值，是否添加位置embedding
    position_embedding_name: 布尔值，位置embedding的embedding矩阵变量的名称
    initializer_range: 浮点数，权重初始化的范围
    max_position_embeddings: 整数，位置embedding的最大序列长度。
      最大长度可以比input_tensor长，但不能比它短。
    dropout_prob: 浮点数，用于最终输出tensor的dropout概率

  输出:
    与input_tensor形状相同的浮点tensor

  引起:
    ValueError: 张量形状或输入值无效
  """
  # 获取input_tensor的形状，预期维度为3
  # [batch_size, seq_length, embedding_size]
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2] # embedding_size

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    # 句子embedding矩阵
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    # 因为token type会很小(2)，所以我们用one-hot会更快
    # 
    # 把token_type_ids压成一维向量，batch_size * seq_length
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    # 转化成one-hot矩阵，[batch_size * seq_length, token_type_vocab_size]
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    # [batch_size * seq_length, token_type_vocab_size] * [token_type_vocab_size, width]
    # =>[batch_size * seq_length, width]
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    # reshape成[batch_size, seq_length, width]，2D tensor-->3D tensor
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    # word embedding + 句子embedding
    output += token_type_embeddings

  if use_position_embeddings:
    # tf.assert_less_equal(x, y)，如果x>y就抛出异常
    # 序列长度不能大于最大序列长度
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    # with tf.control_dependencies([a, b]):
    #   c = ... 
    # c的操作依赖于a和b的操作，在这里相当于if True: ...
    with tf.control_dependencies([assert_op]):
      # 位置embedding
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      # 位置embedding是可学习的参数，形状为[max_position_embeddings, width]的矩阵
      # 输入序列可能比max_position_embeddings短，full_position_embeddings实际上是
      # 位置[0, 1, 2, ..., max_position_embeddings-1]的embedding矩阵，
      # 为了更快地训练没有长序列的任务，用tf.slice取出[0, 1, 2, ... seq_length-1]部分
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      # output的维度为3
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      # 词嵌入的形状为[batch_size, seq_length, embedding_size]
      # 位置embedding只与输入长度有关，形状为[seq_length, width]
      # 2D tensor和3D tensor无法直接相加，因此要把位置嵌入
      # reshape成[1, seq_length, width]，
      # 这样就可以通过广播将词嵌入和位置嵌入相加
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        # 只循环了一次，position_broadcast_shape = [1]
        position_broadcast_shape.append(1)
      # [1, seq_length, width]
      position_broadcast_shape.extend([seq_length, width])
      # reshape成[1, seq_length, width]
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      # word embedding + 句子embedding + 位置embedding
      output += position_embeddings
  # 层归一化和dropout
  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  """
  从2D mask tensor构造3D注意力mask

  参数:
    from_tensor: 2D或3Dtensor，形状为[batch_size, from_seq_length, ...]
    to_mask: int32 tensor，形状为[batch_size, to_seq_length]

  输出:
    浮点tensor，形状为[batch_size, from_seq_length, to_seq_length]
  """
  # 获取from_tensor的形状，预期为2D或3D tensor
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  # [batch_size, from_seq_length, ...]
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]
  # 获取to_mask的形状，预期为2D tensor
  to_shape = get_shape_list(to_mask, expected_rank=2)
  # [batch_size, to_seq_length]
  to_seq_length = to_shape[1]
  # 这里tf.cast()把数据转换成tf.float32
  # 将to_mask重塑成[batch_size, 1, to_seq_length]
  # 为了后面构造Attention Mask矩阵
  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]

  # 我们不认为from_tensor是一个mask（尽管它可能是），
  # 实际上我们不关心是否参与*from* padding的token（仅是*to* padding）
  # 所以我们创造一个取值全为1的tensor
  # broadcast_ones的形状为[batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  # 构造Attention Mask tensor，在后两个维度进行广播
  # 形状为[batch_size, from_seq_length, to_seq_length]
  mask = broadcast_ones * to_mask

  return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  """
  从from_tensor到to_tensor执行多头注意力机制
   from_tensor提供Query，to_tensor提供Key和Value

  这是一个基于论文《Attention is all you need》的多头注意力的实现。如果from_tensor
  和to_tensor是相同的，那么就是自注意力。from_tensor每个时刻都会关注（attends to）
  to_tensor的对应序列，并返回一个固定响亮

  这个函数首先把from_tensor投影为query，把to_tensor投影为key和value。这些都是长度
  为num_attention_heads的tensor列表，其中每个tensor的形状都是
  [batch_size, seq_length, size_per_head]

  然后，对query和key进行点积和放缩，通过softmax函数来得到注意力权重，用这些权重对
  value进行加权，然后把这些tensor拼接成一个tensor并返回。

  实际上，多头注意力是通过转置和重塑而不是实际的分离tensor来完成的。

  Args:
    from_tensor: 浮点tensor，形状为[batch_size, from_seq_length, from_width]
    to_tensor: 浮点tensor，形状为[batch_size, to_seq_length, to_width]
    attention_mask: (可选的)int32 tensor，每个位置取值0或1，
      形状为 [batch_size, from_seq_length, to_seq_length]，
      对于mask中为0的位置，注意力分数将设为负无穷，对于1的位置，注意力分数保持不变
    num_attention_heads: 整数，注意力头数
    size_per_head: 整数，每个注意力头的大小
    query_act: （可选的）query的激活函数
    key_act: （可选的）key的激活函数
    value_act: （可选的）value的激活函数
    attention_probs_dropout_prob: （可选的）浮点数。注意力权重的dropout率
    initializer_range: 浮点数，权重初始化的范围
    do_return_2d_tensor: 布尔值，
      如果为True，输入/输出的形状为
      [batch_size * from_seq_length, num_attention_heads * size_per_head]
      如果为False，输入/输出的形状为
      [batch_size, from_seq_length, num_attention_heads * size_per_head]
    batch_size: （可选的）整数，如果输入是2D的，这是from_tensor和to_tensor的
      3D版本的batch size
    from_seq_length: （可选的）如果输入是2D的，这是from_tensor的3D版本的seq length
    to_seq_length: （可选的）如果输入是2D的，这是to_tensor的3D版本的seq length

  输出:
    浮点tensor，形状为
      [batch_size, from_seq_length, num_attention_heads * size_per_head]
      如果do_return_2d_tensor为True，形状则为
      [batch_size * from_seq_length, num_attention_heads * size_per_head]

  引起:
    ValueError: 任何参数或tensor形状无效则引起异常
  """
  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    # 先reshape成[batch_size, seq_length, num_attention_heads, width]
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])
    # 对中间两个轴进行转置
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor
  # 获取from_tensor和to_tensor的形状
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    # from_tensor和to_tensor的形状要一致，否则抛出异常
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    # 如果都是3D tensor，获取前两个维度
    # [batch_size, from_seq_length, from_width]
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    # 如果是2D tensor，要指定batch_size、from_seq_length和to_seq_length
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  # 下面的注释用缩写表示：
  # B = 序列数量，即batch size
  # F = from_tensor的序列长度
  # T = to_tensor的序列长度
  # N = 注意力头数，即num_attention_heads
  # H = 每个注意力头的大小，即size_per_head

  # 把from_tensor和to_tensor压成矩阵
  # [batch_size, from_seq_length, from_width]
  # => [batch_size * from_seq_length, from_width]
  # 第i*from_seq_length行~第(i+1)*from_seq_length行为第i个batch
  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  # query层 = [B*F, N*H]
  query_layer = tf.layers.dense( # 全连接层
      from_tensor_2d, # 输入
      num_attention_heads * size_per_head, # 神经元数量
      activation=query_act, # 激活函数
      name="query", # 名称
      kernel_initializer=create_initializer(initializer_range)) # 权重初始化

  # `key_layer` = [B*T, N*H]
  # key层 = [B*F, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  # value层 = [B*F, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  # 张量乘法：最后两个维度要满足矩阵乘法，其他维度保持相同
  # [B, N, F, H] * [B, N, H, T] = [B, N, F, T]
  # query和key进行点积运算，Q * K.T
  # `attention_scores` = [B, N, F, T]
  # tf.matmul(a, b) a左乘b
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  # tf.multiply(a, b) a和b对应位置相乘
  # 放缩
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # 为了广播张量，给3D的attention_mask增加一个维度变成4D tensor
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # attention_mask中取值为1.0的地方是我们想要attend to的，0.0的是masked的位置
    # 对于1.0的位置，我们希望它经过softmax后保持权重不变
    # 而0.0的位置是masked的，我们希望它经过softmax后的权重为0
    # 因此给attention_mask加上一个adder，使得attention_mask中
    # 1.0的位置不变，0.0的位置变成-9999.0
    # (1.0 - 1.0) * -10000.0 + 1.0 = 1.0
    # (0.0 - 1.0) * -10000.0 + 0.0 = -9999.0
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # 把注意力得分归一化为权重（概率）
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  # 对注意力进行dropout，实际上是丢弃了整一个token然后attend to
  # 这看起来有点奇怪，但Transformer原始论文是这么做的
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  # [B*F, N*H] => [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer`中间两个轴转置
  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # 计算context_layer（语义层？）
  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # 为了输出2D tensor或3D tensor，需要对中间两个轴转置
  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # 输出2D tensor
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # 输出3D tensor
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  """
  论文《Attention is All You Need》中的多头、多层的Transformer

  这几乎是原Transformer编码器的精确实现

  一个Transformer编码器的结构：
    多头注意力-->残差连接、层归一化-->全连接-->残差连接、层归一化

  原论文: https://arxiv.org/abs/1706.03762

  Transformer源码:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  参数:
    input_tensor: 浮点tensor，形状为[batch_size, seq_length, hidden_size]
    attention_mask: （可选的）int32 tensor，形状为[batch_size, seq_length, seq_length], 
      1的位置表示可以attend to，0的位置是masked的，不可attend to
    hidden_size: 整数，Transformer的隐藏层大小
    num_hidden_layers: 整数，Tansformer的层数（即堆叠多少个Transformer编码器）
    num_attention_heads: 整数，Transformer中的注意力头数
    intermediate_size: 整数，中间层（全连接层）的大小
    intermediate_act_fn: 函数，非线性激活函数，用于中间层/全连接层的输出
    hidden_dropout_prob: 浮点数，隐藏层的dropout率
    attention_probs_dropout_prob: 浮点数，注意力权重的的dropout率
    initializer_range: 浮点数，初始化范围（截断正态分布的标准差）
    do_return_all_layers: 返回所有层还是只返回最后一层

  输出:
    浮点tensor，Transformer的最后一层
      形状为[batch_size, seq_length, hidden_size]

  引起:
    ValueError: tensor形状或参数无效
  """
  if hidden_size % num_attention_heads != 0:
    # 隐藏层大小必须能被注意力头数整除
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))
  # 计算每个注意力头部大小
  attention_head_size = int(hidden_size / num_attention_heads)
  # 获取input_tensor的形状，预期为3D tensor
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  # [batch_size, seq_length, hidden_size]
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  # Transformer会在所有层执行残差连接，因此输入大小要和隐藏层大小相同
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  # 我们把表示保持为一个2D tensor，以避免将它从3D tensor reshape成2D tensor
  # 在CPU/GPU上reshape是自由的,但在TPU上可能不是自由的
  # 因此我们最小化它们来帮助优化器
  #
  # 将input_tensor压成矩阵
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  # 堆叠Transformer，num_hidden_layers层
  for layer_idx in range(num_hidden_layers):
    # 第layer_idx层Transformer下变量的名称
    with tf.variable_scope("layer_%d" % layer_idx):
      # 下一层的输入是上一层的输出
      layer_input = prev_output
      # 注意力层
      with tf.variable_scope("attention"):
        attention_heads = []
        # 自注意力机制
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input, # 自注意力机制，对自身进行
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads, # 注意力头数
              size_per_head=attention_head_size, # 注意力头部大小
              attention_probs_dropout_prob=attention_probs_dropout_prob, # dropout
              initializer_range=initializer_range, # 初始化范围
              do_return_2d_tensor=True, # 输入为2D tensor
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          # 在有其他序列的情况下，我们只是在投影之前拼接自注意力头部
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        # 残差连接和层归一化
        # 注意力层输出名称
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output, # 注意力层输出
              hidden_size, # 神经元数量
              kernel_initializer=create_initializer(initializer_range))
          # dropout
          attention_output = dropout(attention_output, hidden_dropout_prob)
          # 先残差连接，再进行层归一化
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      # 激活函数只用在中间层
      # 中间层名称
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      # 残差连接和层归一化
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        # 把这一层的输出传给下一层
        prev_output = layer_output
        # 记录每一层Transformer的输出
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    # 返回所有层
    final_outputs = []
    for layer_output in all_layer_outputs:
      # 压成矩阵
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    # 只返回最后一层
    # 压成矩阵
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  """
  返回tensor（张量）形状的列表，首选静态维度
    
  参数：
    tensor：一个待查找形状的tf.Tensor对象
    expected_rank：（可选的）整数。“tensor”的预期维度，
      如果指定了它，并且“张量”的维度不同，则将引发异常。
    name：错误消息的张量可选名字
    
  输出：
    张量形状的维数列表。
      所有静态维度将作为python整数返回，
      而动态维度将作为tf.Tensor标量。
  """
  if name is None:
    # 如果不选择name，那么令name为tensor的名字
    name = tensor.name

  if expected_rank is not None:
    # 如果指定了预期的维度，
    assert_rank(tensor, expected_rank, name)
  # 获取tensor的形状，保存为列表形式
  shape = tensor.shape.as_list()

  non_static_indexes = []
  # 获取形状中取值为None的维度所在位置，即不确定的维度
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)
  # 如果形状中各维度都是确定的，返回该形状
  if not non_static_indexes:
    return shape
  # 动态维度
  dyn_shape = tf.shape(tensor)
  # 修改None的维度为动态维度
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  """
  把一个维度>=2的tensor重塑成一个二维张量，例如矩阵
  
  参数：
    input_tensor：要重塑的输入tensor
      假设input_tensor是k维张量，形状为
      [n_1, n_2, n_3, ..., n_{k-1}, n_k]
  
  输出：
    一个二维张量，形状为 [N, n_k]
      其中，N = n_1 * n_2 * ... * n_{k-1}
      """
  # 获取input_tensor的维度
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor
  # 输入形状的最后一个维度大小
  width = input_tensor.shape[-1]
  # 保持最后一个维度大小不变，把输入压成矩阵，-1表示自适应
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  """
  将矩阵复原成原来维度>=2的张量
  
  参数:
    output_tensor: 一个二维张量，即矩阵
    orig_shape_list: 张量原来的维度

  输出:
    reshape后的tensor
  """
  if len(orig_shape_list) == 2:
    return output_tensor
  # 获取矩阵的形状
  output_shape = get_shape_list(output_tensor)
  # 原始形状中除去最后一个维度
  orig_dims = orig_shape_list[0:-1]
  # 矩阵形状中最后一个维度（=原始形状的最后一个维度）
  width = output_shape[-1]
  # 把矩阵复原回原来的形状
  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  """
  如果张量的维度与预期维度不一致就引起异常

  参数:
    tensor: 要检查维度的tf.Tensor对象
    expected_rank: 预期的维度，Python整数或一个整数列表
    name: （可选的）张量名字，用于打印错误信息
  
  引起:
    ValueError: 如果预期形状与实际形状不匹配
  """
  if name is None:
    # 如果不定义name，令name为tensor的名称
    name = tensor.name

  expected_rank_dict = {}
  # 区分expected_rank是整数还是一个整数列表
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True
  # tensor的实际维度
  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))