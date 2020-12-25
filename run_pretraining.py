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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
#! 必须要的参数
flags.DEFINE_string(
  "bert_config_file", None, "bert的模型结构")
    # "The config json file corresponding to the pre-trained BERT model. "
    # "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None, "输入TF实例文件（路径）")
    # "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None, "输出路径，用来保存模型checkpoints")
    # "The output directory where the model checkpoints will be written.")

## Other parameters
#! 其他参数
flags.DEFINE_string(
    "init_checkpoint", None, "初始化参数(通常是来自预训练的bert)")
    # "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128, "最大序列长度")
    # "The maximum total input sequence length after WordPiece tokenization. "
    # "Sequences longer than this will be truncated, and sequences shorter "
    # "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20, "每个序列的最大MLM预测数")
    # "Maximum number of masked LM predictions per sequence. "
    # "Must match data generation.")

flags.DEFINE_bool("do_train", False, "是否训练模型")

flags.DEFINE_bool("do_eval", False, "是否在dev集上进行验证")

flags.DEFINE_integer("train_batch_size", 32, "训练时的batch_size")

flags.DEFINE_integer("eval_batch_size", 8, "验证时的batch_size")

flags.DEFINE_float("learning_rate", 5e-5, "Adam的初始学习率")

flags.DEFINE_integer("num_train_steps", 100000, "训练步数")

flags.DEFINE_integer("num_warmup_steps", 10000, "学习率预热步数")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "保存模型的频率")
                    #  "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000, "在每个估计器调用中要执行的步数")
                    #  "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "验证的最大步数")

flags.DEFINE_bool("use_tpu", False, "用TPU还是GPU/CPU")

tf.flags.DEFINE_string(
    "tpu_name", None, "TPU名字")
    # "The Cloud TPU to use for training. This should be either the name "
    # "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    # "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None, 'TPU相关参数')
    # "[Optional] GCE zone where the Cloud TPU is located in. If not "
    # "specified, we will attempt to automatically detect the GCE project from "
    # "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None, 'TPU相关参数')
    # "[Optional] Project name for the Cloud TPU-enabled project. If not "
    # "specified, we will attempt to automatically detect the GCE project from "
    # "metadata.")

tf.flags.DEFINE_string("master", None, "[可选的]TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8, 'TPU核数')
    # "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""
  """
  返回TPUEstimator的闭包`model_fn`
  
  参数:
    bert_config: bert模型结构
    init_checkpoint: 初始参数
    learning_rate: 学习率
    num_train_steps: 训练步数
    num_warmup_steps: 学习率预热步数
    use_tpu: 是否用TPU
    use_one_hot_embeddings: 是否用onehot embedding
  
  返回:
    model_fn
  """

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

    """The `model_fn` for TPUEstimator."""
    """TPUEstimator的`model_fn`"""
    #! features来自input_fn_builder中的input_fn
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    #? 一些模型的输入特征
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]
    #? 是否训练 tf.estimator.ModeKeys.TRAIN = 'train'
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    #! 创建bert模型
    model = modeling.BertModel(
        config=bert_config, #? bert模型结构
        is_training=is_training, #? True为训练，False为推力。用来控制是否dropout
        input_ids=input_ids, #? token id
        input_mask=input_mask, #? mask向量
        token_type_ids=segment_ids, #? 句子id
        use_one_hot_embeddings=use_one_hot_embeddings) #? 是否用onehot embedding
    #! 获取MLM任务的batch损失、每个样本的损失和对数概率
    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)
    #! 获取NSP任务的batch损失、每个样本的损失和对数概率
    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)
    #! 总损失 = MLM loss + NSP loss
    total_loss = masked_lm_loss + next_sentence_loss
    #? 可训练变量
    tvars = tf.trainable_variables()
    #? 初始化变量的名称
    initialized_variable_names = {}
    scaffold_fn = None
    #! 如果有保存过模型, 就恢复参数
    if init_checkpoint:
      #? 取tvars和initialized_variable_names的交集变量
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    #? 打印可训练变量
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    #! 训练时
    if mode == tf.estimator.ModeKeys.TRAIN:
      #? 创建优化器训练操作
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      #? 输出空间, 返回TPUEstimatorSpec实例
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    #! 验证时
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        """
        计算模型的损失和准确率
        
        参数:
          masked_lm_example_loss: MLM任务每个样本的loss
          #?  1D Tensor [eval_batch_size * max_predictions_per_seq, ]
          masked_lm_log_probs: MLM任务的对数概率
          #?  2D Tensor [eval_batch_size * max_predictions_per_seq, vocab_size]
          masked_lm_ids: MLM任务masked token的token id
          #?  2D Tensor [eval_batch_size, max_predictions_per_seq]
          masked_lm_weights: MLM任务输出层权重
          #?  2D Tensor [eval_batch_size, max_predictions_per_seq]
          next_sentence_example_loss: NSP任务每个样本的loss
          #?  1D Tensor [eval_batch_size, ]
          next_sentence_log_probs: NSP任务的对数概率
          #?  2D Tensor [eval_batch_size, 2]
          next_sentence_labels: NSP任务的label
          #?  2D Tensor [eval_batch_size, 1]

        输出:
          字典, 包含MLM和NSP任务的loss和准确率
        """
        #? 输入的是2D, reshape后没变化
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        #? 求masked_lm_log_probs每一行的最大值所在列的位置
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        #? 输入的是1D, reshape后没变化
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        #? 把masked_lm_ids和masked_lm_weights压成一维向量
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        #! 计算MLM任务的准确率
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        #! 计算MLM任务的loss
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        #? 输入的是2D, reshape后没变化
        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        #? 求next_sentence_log_probs每一行的最大值所在列的位置
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        #? 把next_sentence_labels压成一维向量
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        #! 计算NSP任务的准确率
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        #! 计算NSP任务的loss
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }
      
      #? 把meetric_fn及其传参打包成eval_metrics, 传入TPUEstimatorSpec
      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      #? 输出空间, 返回TPUEstimatorSpec实例
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  """
  获取MLM的loss和对数概率

  参数:
    bert_config: bert模型结构
    input_tensor: 每个token对应的向量
    output_weights: 输出层权重
    positions: MLM预测的位置
    label_ids: MLM预测的token id
    label_weights: MLM权重

  输出: (loss, per_example_loss, log_probs)
    loss: batch的loss
    per_example_loss: 每个样本的loss
    log_probs: 对数概率
  """
  # 根据位置获取MLM token对应的向量
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    #! 我们在输出层之前再用一个非线性变换。
    #! 预训练之后(fine-tune)不使用该矩阵。
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      #? 层归一化
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    #? 输出权重与输入嵌入相同，但是每个token都有一个输出的偏置。
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    #? logits = X * W^T
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    #? logits += b
    logits = tf.nn.bias_add(logits, output_bias)
    #? 对数概率
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    #? 把label_ids和label_weights 压成一维向量
    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])
    #? 把label_ids转成onehot
    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    #! `positions`可能是用0填充的(如果序列太短而没有最大数量的预测)。
    #! `label_weights`的每个实际预测值为1.0，填充预测值为0.0
    #? 每个样本的损失
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    #? padding的损失要mask掉
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    #? 实际预测的token数量, 可能为0
    #? 因此需要加一个很小的数, 防止计算平均loss时除以0
    denominator = tf.reduce_sum(label_weights) + 1e-5
    #? batch上的loss
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""
  """
  获取NSP任务的loss和对数概率

  参数:
    bert_config: bert模型结构
    input_tensor: [CLS]对应的最后一层向量
    labels: 真实的label向量

  输出: (loss, per_example_loss, log_probs)
    loss: batch的loss
    per_example_loss: 每个样本的loss
    log_probs: 对数概率

  """
  #! 简单的二分类任务, 0为真实的下一句, 1为随机的下一句
  #! 在预训练之后不使用权重矩阵
  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size], #? 二分类
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())
    #? logits = X * W^T
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    #? logits += b
    logits = tf.nn.bias_add(logits, output_bias)
    #? 对数概率(log softmax)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    #? 把labels压成一维向量
    labels = tf.reshape(labels, [-1])
    #? 把labels转成onehot
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    #? 每个样本的loss
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    #? 整个batch的loss(取平均)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  """
  获取小批量指定位置的向量
  
  参数:
    sequence_tensor: 每个token对应的最后一层向量
      3D Tensor, [batch_size, seq_length, width]
    positions: MLM的位置, `masked_lm_positions`
      2D Tensor, [batch_size, max_predictions_per_seq]
  
  输出:
    output_tensor: MLM对应的向量
  """
  #! sequence_tensor的预期形状是3维
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0] #? 批量大小
  seq_length = sequence_shape[1] #? 最大序列长度
  width = sequence_shape[2] #? 向量维度

  """
  sequence_tensor = [  |  positions = [
    [                  |                [3 2 0],
  3/  [8 8 3 9 4],     |                [2 1 0],
      [0 8 5 4 6],     |              ]
  2/  [8 0 0 9 5],     |  此时有,
  1/  [8 0 2 9 0],     |    batch_size = 2
      [4 0 1 3 7],     |    seq_length = 6
      [0 1 1 0 5],     |    width = 5
    ],                 |  这里要取出的向量为
    [                  |  左边的sequence_tensor
  6/  [9 6 9 3 6],     |  标出了`i/`的行
  5/  [7 1 0 7 2],     |  i是取出的顺序
  4/  [4 9 5 6 5],     |
      [5 5 6 2 8],     |
      [2 9 8 9 2],     |
      [4 5 9 8 3],     |
    ],                 |
  ]                    |

    首先看 #? a = tf.range(0, batch_size, dtype=tf.int32) * seq_length
  结果为: [0 6] (结果均为tf.Tensor对象)
    然后 #? flat_offsets = tf.reshape(a, [-1, 1])
  结果为: [[0]
          [6]]
    #? b = positions + flat_offsets
  结果为: [[3 2 0]
          [8 7 6]]
    #? flat_positions = tf.reshape(b, [-1])
  结果为: [3 2 0 8 7 6]
    #?   flat_sequence_tensor = tf.reshape(sequence_tensor,
    #?                               [batch_size * seq_length, width])
  结果为: [[8 8 3 9 4]
          [0 8 5 4 6]
          [8 0 0 9 5]
          [8 0 2 9 0]
          [4 0 1 3 7]
          [0 1 1 0 5]
          [9 6 9 3 6]
          [7 1 0 7 2]
          [4 9 5 6 5]
          [5 5 6 2 8]
          [2 9 8 9 2]
          [4 5 9 8 3]]
    #? output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  结果为: [[8 0 2 9 0]
          [8 0 0 9 5]
          [8 8 3 9 4]
          [4 9 5 6 5]
          [7 1 0 7 2]
          [9 6 9 3 6]] 
  """
  #! 由于positions中的位置是MLM在每个句子中的位置
  #! 在这里要转成在整个batch中的位置(3D tensor压成矩阵后)
  #! 所以要给positions一个batch大小的偏移量(offset)
  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  #! 压成矩阵后MLM在batch中的位置
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  #! 把sequence_tensor压成矩阵
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  #! 取出MLM对应位置的向量
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  """
  创建输入到TPUEstimator的`input_fn`闭包
  
  参数:
    input_files: list, 数据的路径
    max_seq_length: 最大序列长度
    max_predictions_per_seq: 每个序列的最大MLM预测数
    is_training: bool, True为训练, False为推理(eval or test)
    num_cpu_threads: cpu线程数, 默认为4

  返回:
    input_fn
  """

  def input_fn(params):
    """The actual input function."""
    """实际的输入函数"""
    #? batch_size
    batch_size = params["batch_size"]
    #? 定义定长变量及其数据类型
    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    #! 训练时，我们需要大量的并行读取和打乱。
    #! 推理时，我们不需要打乱和并行读取。
    if is_training:
      #? tf.constant(input_files) -> [b'F:\\bert-master\\output.tf_record', ...]
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      #? repeat(count) --> 重复count次dataset
      #? repeat()不传参 --> 无限重复dataset
      d = d.repeat()
      #? 打乱数据集
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      #? `cycle_length`是并行读取文件的数量
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      #? `sloppy`模式意味着交错不精确, 这给训练管道增加了更多的随机性。
      #! dataset.apply(transformation_func)
      #! 将transformation_func这个函数应用到dataset上
      d = d.apply(
        #? 并行读取文件数据, 并行数量为cycle_length
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      #? 打乱数据
      d = d.shuffle(buffer_size=100)
    else:
      #! 推理时不需要并行读取和打乱
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      #? 因为我们对固定数量的步骤进行评估，所以我们不希望遇到out-of-range异常。
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    #? 我们必须在训练中把`drop_remainder`设为True, 因为TPU需要固定的尺寸。
    #? `drop_remainder`=True, 把最后一个batch丢掉, 如果最后一个batch数量 < batch_size时
    #? 对于eval，假设是在CPU或GPU上进行评估，不丢掉最后一个batch，否则最后一个bacth将不会预测。
    d = d.apply(
      #? map和batch同时进行
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  """
  转换int精度
    tf.Example只支持tf.int64, TPU只支持tf.int32
    把所有tf.int64转成tf.int32
  """
  # 从TFRecords文件中读取数据
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")
  #! 从json文件初始化bert模型
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  #! 如果output_dir不存在则创建文件夹, 否则跳过
  tf.gfile.MakeDirs(FLAGS.output_dir)
  #! 用list保存输入文件路径
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  #! TPU相关配置
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
  #! 构建模型, Estimator的一个参数
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  #! 如果TPU不可用, 这将在CPU/GPU上使用普通的Estimator
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)
  #! 训练
  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    #! 构建模型输入
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    #! 进行训练
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
  #! 验证
  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    #! 构建模型输入
    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)
    #! 进行验证
    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
    #! 保存验证结果
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
