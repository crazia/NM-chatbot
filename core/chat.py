import tensorflow as tf
import argparse
import numpy as np
import math
import sys
import re

# other module


from . import nmt
from . import attention_model
from . import gnmt_model
from . import model as nmt_model
from . import model_helper

from . import constants
from konlpy.tag import Mecab

from .utils import vocab_utils

from .utils import misc_utils as utils
from .utils import nmt_utils


FLAGS = constants

mecab = Mecab()

def remove_special_char(s_input):
    return re.sub("([.,!?\"':;)(])", "", s_input)


def apply_nlpy(s_input):
    """
    형태소 분석을 통해서 교육 효율을 높인다.
    """
    result = mecab.morphs(s_input)
    return ' '.join(result)
    

class ChatBot:
    ckpt = None
    hparams = None
    infer_model = None
    sess = None
    loaded_infer_model = None
    
    def __init__(self):
        pass

    def _do_reply(self, input):
        # print(input)
        # 원 소스가 이리 되어 있삼

        # 입력소스가 0 일때 리턴
        if len(input) == 0:
            return ''

        infer_data = [apply_nlpy(remove_special_char(input))]

        self.sess.run(
            self.infer_model.iterator.initializer,
            feed_dict={
                self.infer_model.src_placeholder: infer_data,
                self.infer_model.batch_size_placeholder: self.hparams.infer_batch_size})

        # variable check
            
        beam_width = self.hparams.beam_width
        num_translations_per_input = max(
            min(1, beam_width), 1)
            
        nmt_outputs, _ = self.loaded_infer_model.decode(self.sess)
        if beam_width == 0:
            nmt_outputs = np.expand_dims(nmt_outputs, 0)

        batch_size = nmt_outputs.shape[1]

        for sent_id in range(batch_size):
            for beam_id in range(num_translations_per_input):
                translation = nmt_utils.get_translation(
                    nmt_outputs[beam_id],
                    sent_id,
                    tgt_eos=self.hparams.eos,
                    subword_option=self.hparams.subword_option)
                        
        return translation.decode('utf-8')

    def nmt_main(self, flags, default_hparams, scope=None):
        ## Train / Decode
        out_dir = flags.out_dir

        if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

        # Load hparams.
        self.hparams = nmt.create_or_load_hparams(
            out_dir, default_hparams, flags.hparams_path, save_hparams=False)

        self.ckpt = flags.ckpt
        if not self.ckpt:
            self.ckpt = tf.train.latest_checkpoint(out_dir)

        if not self.ckpt:
            print('Train is needed')
            sys.exit()

        hparams = self.hparams
        
        if not hparams.attention:
            model_creator = nmt_model.Model
        elif hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
            model_creator = gnmt_model.GNMTModel
        else:
            raise ValueError("Unknown model architecture")
        self.infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

        # get tensorflow session

        self.sess = tf.Session(graph=self.infer_model.graph, config=utils.get_config_proto())

        with self.infer_model.graph.as_default():
            self.loaded_infer_model = model_helper.load_model(
                self.infer_model.model, self.ckpt, self.sess, 'infer')

    
    def run(self,flags, default_hparams):
        # load all parameters
        self.nmt_main(flags, default_hparams)
        try:
            sys.stdout.write("> ")
            sys.stdout.flush()
            line = sys.stdin.readline()

            while line:
                print(self._do_reply(line.strip()))

                sys.stdout.write("\n> ")
                sys.stdout.flush()

                line = sys.stdin.readline()

        except KeyboardInterrupt:
            self.sess.close()
            sys.exit()

def create_hparams(flags):
    """Create training hparams."""


    return tf.contrib.training.HParams(
        out_dir=flags.out_dir,
        override_loaded_hparams=flags.override_loaded_hparams,
    )

def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    print('starting... \n')
    chatbot = ChatBot()
    chatbot.run(FLAGS, default_hparams)

if __name__ == "__main__":
    tf.app.run(main=main, argv=[sys.argv[0]])
