# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
"""Utility functions for GLUE classification tasks."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
import collections
import csv
import os
from albert import fine_tuning_utils
from albert import modeling
from albert import optimization
from albert import tokenization
import tensorflow.compat.v1 as tf
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import tpu as contrib_tpu


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               guid=None,
               example_id=None,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def __init__(self, use_spm, do_lower_case):
    super(DataProcessor, self).__init__()
    self.use_spm = use_spm
    self.do_lower_case = do_lower_case

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def process_text(self, text):
    if self.use_spm:
      return tokenization.preprocess_text(text, lower=self.do_lower_case)
    else:
      return tokenization.convert_to_unicode(text)



class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "test_matched.tsv")),
        "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      # Note(mingdachen): We will rely on this guid for GLUE submission.
      guid = self.process_text(line[0])
      text_a = self.process_text(line[8])
      text_b = self.process_text(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = self.process_text(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class BC5CDRProcessor(DataProcessor):
    """NER dataset"""
    def get_train_examples(self, data_dir):
        l1 = self._read_data(os.path.join(data_dir, "train.tsv"))
        l2 = self._read_data(os.path.join(data_dir, "devel.tsv"))
        return self._create_example(l1 + l2, "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

class GADProcessor(DataProcessor):
  """Processor for the BioALBERT data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")),
        "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")),
        "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i ==0: #and set_type=='test':
        continue
      if set_type != "test":
        guid = "%s-%s" % (set_type, i)
        text_a = self.process_text(line[0])
        label = self.process_text(line[1])
      else:
        guid = self.process_text(line[0])
        #guid = "%s-%s" % (set_type, line[0])
        text_a = self.process_text(line[1])
        label = "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

class ChemProtProcessor(GADProcessor):

    def get_labels(self):
        """See base class."""
        return [i.lower() for i in ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]]
    
    def _create_example(self, lines, set_type):
       """Creates examples for the training and dev sets."""
       examples = []
       for (i, line) in enumerate(lines):
           #skip header
           if i==0 and set_type=='test':
               continue
           guid = line[0]
           text_a = tokenization.convert_to_unicode(line[1])
           if set_type == "test":
               label = self.get_labels()[-1]
           else:
               try:
                   label = tokenization.convert_to_unicode(line[2])
               except IndexError:
                   logging.exception(line)
                   exit(1)
           examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
       return examples

class euadrProcessor(DataProcessor):
  """Processor for the BioALBERT data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")),
        "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")),
        "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i ==0  and set_type=='test':
        continue
      if set_type != "test":
        guid = "%s-%s" % (set_type, i)
        text_a = self.process_text(line[0])
        label = self.process_text(line[1])
      else:
        guid = self.process_text(line[0])
        #guid = "%s-%s" % (set_type, line[0])
        text_a = self.process_text(line[1])
        label = "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class MisMnliProcessor(MnliProcessor):
  """Processor for the Mismatched MultiNLI data set (GLUE version)."""

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "dev_mismatched.tsv")),
        "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "test_mismatched.tsv")),
        "test")

class chawalProcessor(DataProcessor):
  """Processor for the Mismatched MultiNLI data set (GLUE version)."""

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "dev_mismatched.tsv")),
        "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "test_mismatched.tsv")),
        "test")


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MRPC", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MRPC", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MRPC", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = self.process_text(line[3])
      text_b = self.process_text(line[4])
      if set_type == "test":
        guid = line[0]
        label = "0"
      else:
        label = self.process_text(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,  "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,  "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,  "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        guid = line[0]
        text_a = self.process_text(line[1])
        label = "0"
      else:
        text_a = self.process_text(line[3])
        label = self.process_text(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class Sst2Processor(DataProcessor):
  """Processor for the SST-2 data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "SST-2", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "SST-2", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "SST-2", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      if set_type != "test":
        guid = "%s-%s" % (set_type, i)
        text_a = self.process_text(line[0])
        label = self.process_text(line[1])
      else:
        guid = self.process_text(line[0])
        # guid = "%s-%s" % (set_type, line[0])
        text_a = self.process_text(line[1])
        label = "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


  def _create_examples_backup(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      if set_type != "test":
        guid = "%s-%s" % (set_type, i)
        text_a = self.process_text(line[0])
        label = self.process_text(line[1])
      else:
        #guid = self.process_text(line[0])
        guid = "%s-%s" % (set_type, line[0])
        text_a = self.process_text(line[1])
        label = "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class StsbProcessor(DataProcessor):
  """Processor for the STS-B data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "STS-B", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "STS-B", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "STS-B", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return [None]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = self.process_text(line[0])
      # guid = "%s-%s" % (set_type, line[0])
      text_a = self.process_text(line[7])
      text_b = self.process_text(line[8])
      if set_type != "test":
        label = float(line[-1])
      else:
        label = 0
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples



class QnliProcessor(DataProcessor):
  """Processor for the QNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QNLI", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QNLI", "dev.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QNLI", "test.tsv")),
        "test_matched")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = self.process_text(line[0])
      # guid = "%s-%s" % (set_type, line[0])
      text_a = self.process_text(line[1])
      text_b = self.process_text(line[2])
      if set_type == "test_matched":
        label = "entailment"
      else:
        label = self.process_text(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class RteProcessor(DataProcessor):
  """Processor for the RTE data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "RTE", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "RTE", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "RTE", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = self.process_text(line[0])
      # guid = "%s-%s" % (set_type, line[0])
      text_a = self.process_text(line[1])
      text_b = self.process_text(line[2])
      if set_type == "test":
        label = "entailment"
      else:
        label = self.process_text(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class WnliProcessor(DataProcessor):
  """Processor for the WNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "WNLI", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "WNLI", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "WNLI", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = self.process_text(line[0])
      # guid = "%s-%s" % (set_type, line[0])
      text_a = self.process_text(line[1])
      text_b = self.process_text(line[2])
      if set_type != "test":
        label = self.process_text(line[-1])
      else:
        label = "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class AXProcessor(DataProcessor):
  """Processor for the AX data set (GLUE version)."""

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "diagnostic", "diagnostic.tsv")),
        "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      # Note(mingdachen): We will rely on this guid for GLUE submission.
      guid = self.process_text(line[0])
      text_a = self.process_text(line[1])
      text_b = self.process_text(line[2])
      if set_type == "test":
        label = "contradiction"
      else:
        label = self.process_text(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class HoCProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        label_list = []
        
        # num_aspect=FLAGS.num_aspects
        # --aspect_value_list="0,1"
        aspect_value_list = [0, 1] 
        # --num_aspects=10 
        for i in range(10): #range(FLAGS.num_aspects):
            for value in aspect_value_list:
                label_list.append(str(i) + "_" + str(value))
        return label_list  # [ {'0_-2': 0, '0_-1': 1, '0_0': 2, '0_1': 3,....'19_-2': 76, '19_-1': 77, '19_0': 78, '19_1': 79}]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            #if set_type == "test":
            #  text_a = tokenization.convert_to_unicode(line[1])
            #  label = "0"
            #else:
            #  text_a = tokenization.convert_to_unicode(line[3])
            #  label = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            text_a = tokenization.convert_to_unicode(line[1])

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, task_name):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  #if task_name != "sts-b":
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

  # The convention in ALBERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
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
  label_id = None
  #tf.logging.info('processing multiple labels')
  if "," in example.label:  # multiple label
    #get list of label
    label_id_list = []
    label_list = example.label.split(",")
    for label_ in label_list:
        tf.logging.info(label_)
        try:
            label_id_list.append(label_map[label_])
        except:
            tf.logging.error(label_)
            tf.logging.error(label_map)
            exit(1)
        # print("label_id_list:",label_id_list
        # convert to multi-hot style
    label_id = [0 for l in range(len(label_map))]
    for j, label_index in enumerate(label_id_list):
        label_id[label_index] = 1
  else:  # single label
      label_id = label_map[example.label]

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    if "," in example.label: tf.logging.info("label: %s (id_list = %s)" % (str(example.label),str(label_id_list)))  # if label_id is a list, try print multi-hot value: label_id_list
    tf.logging.info("label: %s (id = %s)" % (str(example.label), str(label_id)))  # %d

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, task_name):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, task_name)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    # if feature.label_id is already a list, then no need to add [].
    if isinstance(feature.label_id, list):
        label_ids = feature.label_id
    else:
        label_ids = [feature.label_id]
    features["label_ids"] = create_int_feature(label_ids)
    features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, task_name, use_tpu, bsz,
                                multiple=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  labeltype = tf.float32 if task_name == "sts-b" else tf.int64
  num_classes = 20
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "label_ids": tf.FixedLenFeature([num_classes], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if use_tpu:
      batch_size = params["batch_size"]
    else:
      batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(albert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, task_name,
                 hub_module):
  """Creates a classification model."""
  (output_layer, _) = fine_tuning_utils.create_albert(
      albert_config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      use_einsum=True,
      hub_module=hub_module)

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    # print("labels:",labels,";logits:",logits,"isinstance(labels,list):",isinstance(labels,list))
    # mulit-label classification: 1.multi-hot==> then use sigmoid to transform it to possibility
    probabilities = tf.nn.sigmoid(logits)
    # log_probs=tf.log(probabilities)
    labels = tf.cast(labels, tf.float32)
    #probabilities = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
    #log_probs = tf.nn.log_softmax(logits, axis=-1)
    #one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    ##per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(per_example_loss)


    return (loss, per_example_loss, probabilities, logits, predictions)


def model_fn_builder(albert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, task_name, hub_module=None,
                     optimizer="adamw"):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, probabilities, logits, predictions) = create_model(albert_config, 
                                                                                      is_training, 
                                                                                      input_ids, 
                                                                                      input_mask,
                                                                                      segment_ids, 
                                                                                      label_ids, 
                                                                                      num_labels, 
                                                                                      use_one_hot_embeddings,
                                                                                      task_name, 
                                                                                      hub_module)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
          #predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          # print("###metric_fn.logits:",logits.shape) # (?,80)
          # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          # print("###metric_fn.label_ids:",label_ids.shape,";predictions:",predictions.shape) # label_ids: (?,80);predictions:(?,)
          num_aspects = 10 # magic number
          logits_split = tf.split(logits, num_aspects, axis=-1)  # a list. length is num_aspects
          label_ids_split = tf.split(logits, num_aspects, axis=-1)  # a list. length is num_aspects
          accuracy = tf.constant(0.0, dtype=tf.float64)

          for j, logits in enumerate(logits_split):  #
              #  accuracy = tf.metrics.accuracy(label_ids, predictions)

              predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)  # should be [batch_size,]
              label_id_ = tf.cast(tf.argmax(label_ids_split[j], axis=-1), dtype=tf.int32)
              tf.logging.debug("label_ids_split[j] = %s; predictions = %s; label_id_ = %s" %
                               (label_ids_split[j], predictions, label_id_))
              current_accuracy, update_op_accuracy = tf.metrics.accuracy(
                  labels=label_id_, predictions=predictions, weights=is_real_example)
              accuracy += tf.cast(current_accuracy, dtype=tf.float64)
          accuracy = accuracy / tf.constant(num_aspects, dtype=tf.float64)
          loss = tf.metrics.mean(per_example_loss)
          return {
              "eval_accuracy": (accuracy, update_op_accuracy),
              "eval_loss": loss,
          }

        eval_metrics = (metric_fn,
                        [per_example_loss, label_ids, logits, is_real_example])
        output_spec = contrib_tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)

    else:
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "probabilities": probabilities,
              "predictions": predictions
          },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
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
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, task_name):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, task_name)

    features.append(feature)
  return features
