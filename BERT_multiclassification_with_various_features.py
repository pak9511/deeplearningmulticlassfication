# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
import os,collections,pprint,json,re,konlpy
from tqdm import tqdm
import numpy as np
#!pip install bert-tensorflow==1.0.1
print("tensorflow version : ", tf.__version__)

#Importing BERT modules
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

from transformers import BertTokenizer

"""**Data preprocessing**"""

def preprocess_text(sen):
  # Removing html tags
  sentence = re.sub(r'<[^>]+>', '', sen)
  # Remove punctuations and numbers
  sentence = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]+', ' ', sentence)
  # Single character removal
  sentence =  re.sub(r"\s+[ㄱ-ㅎㅏ-ㅣ가-힣]\s+", ' ', sentence)
  # Removing multiple spaces
  sentence = re.sub(r'\s+', ' ', sentence)

  return sentence

def Pipeline(sen):
  with open ('stopwordsKor.txt', 'r',encoding='utf-8') as fr:
    stopwords=[i.strip() for i in fr]
  
  sentence=konlpy.tag.Mecab().pos(sen)
  sent_res=""
  for ko in sentence:
    if not ko[0] in stopwords:
      sent_res = sent_res+ko[0]+' '
  return sent_res

input_file = 'input.txt'
mode='Pipeline' #Pipeline or preprocess_text
df = pd.read_csv(input_file, encoding='utf-8', engine='python', header=0, sep='\t',index_col=0)

df.dropna(how='any')
df['topic']=df['topic'].astype(float)
df['sentiment'] = df['sentiment']*0.1
df['sentiment']=df['sentiment'].round(2)
print(df.dtypes)

print(df.columns)
print(df['text'].head())
print(df['topic'].value_counts())
print(df['sentiment'].value_counts())
print(df['level'].value_counts())

tqdm.pandas()
random_seed = 42
sample_size=df['level'].count()

cleaned=[]

print('mode  :  '+mode)
if mode=='Pipeline':  
  for sen in df['text'].values:
    c_sen=preprocess_text(sen)
    cleaned.append(Pipeline(c_sen))

elif mode =='preprocess_text':
  for sen in df['text'].values:
    cleaned.append(preprocess_text(sen))

extra_features = pd.DataFrame({'text':cleaned , 
                            'label':[x for x in df['level']],
                              'sentiment':df['sentiment'].tolist() ,
                              'topic': df['topic'].tolist() })

# # shuffle
extra_features = extra_features.sample(frac=1, random_state=random_seed)        
        
# # split train and test
train,test = train_test_split(extra_features, test_size=0.20, random_state=42)
num_of_extra_features = 1


MAX_SEQ_LENGTH = 256
label_list=df['level'].unique().tolist()
print('label list : '+str(label_list))

tokenizer= BertTokenizer(vocab_file='/gdrive/My Drive/Colab Notebooks/multi_cased_L-12_H-768_A-12/vocab.txt',do_lower_case=False)

class InputExampleExtraFeatures(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None,sentiment=None,topic=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
      extra_features: 1-D numpy array of extra features
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.sentiment=sentiment
    self.topic=topic

def convert_examples_to_features_extra(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
      
    feature = convert_single_example_extra_features(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

train_InputExamples_extra_features = train.apply(lambda x: InputExampleExtraFeatures(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x['text'], 
                                                                   text_b = None, 
                                                                   label = x['label'], 
                                                                   sentiment = x['sentiment'],# adding extra features
                                                                   topic = x['topic']), axis = 1)

test_InputExamples_extra_features = test.apply(lambda x: InputExampleExtraFeatures(guid=None, 
                                                                   text_a = x['text'], 
                                                                   text_b = None, 
                                                                   label = x['label'],
                                                                   sentiment = x['sentiment'], # adding extra features
                                                                   topic = x['topic']), axis = 1)

print("Row 0 - guid of training set : ", train_InputExamples_extra_features.iloc[0].guid)
print("\n__________\nRow 0 - text_a of training set : ", train_InputExamples_extra_features.iloc[0].text_a)
print("\n__________\nRow 0 - text_b of training set : ", train_InputExamples_extra_features.iloc[0].text_b)
print("\n__________\nRow 0 - label of training set : ", train_InputExamples_extra_features.iloc[0].label)

#Here is what the tokenised sample of the first training set observation looks like
print("\n__________\nRow 0 - tokenization : ",tokenizer.tokenize(train_InputExamples_extra_features.iloc[0].text_a))

# From https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L161
class PaddingInputExample_ExtraFeatures(object):
    pass

class InputFeatures_ExtraFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               sentiment,
               topic
              #  extra_features
               ):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.sentiment = sentiment
    self.topic = topic

def convert_single_example_extra_features(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample_ExtraFeatures):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        # if this is padding, insert 3 zeros for extra features
        sentiment=[0] * num_of_extra_features,
        topic=[0] * num_of_extra_features
        )

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None

  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    tf.logging.info("sentiment: %s " % (example.sentiment))
    tf.logging.info("topic: %s " % (example.topic))
    
  feature = InputFeatures_ExtraFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      topic=example.topic,
      sentiment=example.sentiment
      )
  return feature

# Convert our train and validation features to InputFeatures that BERT understands.
train_features_extra = convert_examples_to_features_extra(train_InputExamples_extra_features, label_list, MAX_SEQ_LENGTH, tokenizer)
val_features_extra = convert_examples_to_features_extra(test_InputExamples_extra_features, label_list, MAX_SEQ_LENGTH, tokenizer)

def create_model_extra_features(bert_config,is_training,is_predicting, input_ids, input_mask, segment_ids,sentiment,topic,
                 labels, num_labels):
  model = modeling.BertModel(
      config=bert_config,
      is_training=True,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      )

  output_layer = model.get_pooled_output()
  output_layer_extra_features = tf.concat([output_layer,tf.convert_to_tensor(sentiment, dtype=tf.float32),tf.convert_to_tensor(topic, dtype=tf.float32)],axis=1)  

  hidden_size = output_layer_extra_features.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # Dropout helps prevent overfitting
      output_layer_extra_features = tf.nn.dropout(output_layer_extra_features, keep_prob=0.9)

    logits = tf.matmul(output_layer_extra_features, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    
    return (loss, predicted_labels, log_probs)

def input_fn_builder_extra_features(features, seq_length, is_training,is_predicting, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []
  all_sentiment=[]
  all_topic=[]

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)
    all_sentiment.append(feature.sentiment)
    all_topic.append(feature.topic)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        "sentiment":
          tf.constant(all_sentiment,shape=[num_examples,num_of_extra_features] ,dtype=tf.float32),
        "topic":
          tf.constant(all_topic,shape=[num_examples,num_of_extra_features] ,dtype=tf.float32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d


  return input_fn

def model_fn_builder_extra_features(bert_config,num_labels,init_checkpoint, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  # model_fn : 딥러닝 custom algorithm이 수행되는 곳
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    sentiment=features['sentiment']
    topic=features['topic']
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    is_eval= (mode == tf.estimator.ModeKeys.EVAL)
    # TRAIN and EVAL
    if is_training or is_eval :
      (loss, predicted_labels, log_probs) = create_model_extra_features(
          bert_config,is_training,is_predicting, input_ids, input_mask, segment_ids, sentiment,topic, label_ids,num_labels)

      train_op = bert.optimization.create_optimizer(
            loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
          accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
          true_pos = tf.metrics.true_positives(
              label_ids,
              predicted_labels)
          true_neg = tf.metrics.true_negatives(
              label_ids,
              predicted_labels)   
          false_pos = tf.metrics.false_positives(
              label_ids,
              predicted_labels)  
          false_neg = tf.metrics.false_negatives(
              label_ids,
              predicted_labels)
          tf.summary.scalar('accuracy', accuracy[1])
 
          return {
              "eval_accuracy": accuracy,
              "true_positives": true_pos,
              "true_negatives": true_neg,
              "false_positives": false_pos,
              "false_negatives": false_neg
              }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op,
          )
      elif mode == tf.estimator.ModeKeys.EVAL:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model_extra_features(
        bert_config,is_training,is_predicting, input_ids, input_mask, segment_ids, sentiment,topic, label_ids,num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 10.0
# Warmup is a period of time where the learning rate is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 300
SAVE_SUMMARY_STEPS = 100

OUTPUT_DIR_EXTRA='res/'

# Compute train and warmup steps from batch size
num_train_steps = int(len(train_features_extra) * NUM_TRAIN_EPOCHS/ BATCH_SIZE )
# num_train_steps = int(len(train_features_extra) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify output directory and number of checkpoint steps to save
run_config_extra = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR_EXTRA,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

#Initializing the model and the estimator
model_fn_extra_features = model_fn_builder_extra_features(
  bert_config=  modeling.BertConfig.from_json_file('multi_cased_L-12_H-768_A-12/bert_config.json'),
  num_labels=len(label_list),
  init_checkpoint=OUTPUT_DIR_EXTRA+'/bert_model.ckpt',
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps
 )
# custom estimator 선언
estimator_extra = tf.estimator.Estimator(
  model_fn=model_fn_extra_features,
  config=run_config_extra,
  params={"batch_size": BATCH_SIZE})

train_input_fn_extra = input_fn_builder_extra_features(
    features=train_features_extra,
    seq_length=MAX_SEQ_LENGTH,
    is_predicting=False,
    is_training=True,
    drop_remainder=False)

val_input_fn_extra = input_fn_builder_extra_features(
    features=val_features_extra,
    seq_length=MAX_SEQ_LENGTH,
    is_predicting=False,
    is_training=False,
    drop_remainder=False)

print(f'Beginning Training!')
current_time = datetime.now()
estimator_extra.train(input_fn=train_input_fn_extra, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

#Evaluating the model with Validation set
estimator_extra.evaluate(input_fn=val_input_fn_extra, steps=None)