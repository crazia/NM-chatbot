import tensorflow as tf
import argparse
import numpy as np
import math
import sys

# other module


import nmt
import attention_model
import gnmt_model
import model as nmt_model
import model_helper

from utils import vocab_utils

from utils import misc_utils as utils
from utils import nmt_utils


FLAGS = None

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
        infer_data = [input]

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
        # Job
        jobid = flags.jobid
        num_workers = flags.num_workers
        
        ## Train / Decode
        out_dir = flags.out_dir
        if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

        # Load hparams.
        self.hparams = nmt.create_or_load_hparams(
            out_dir, default_hparams, flags.hparams_path, save_hparams=(jobid == 0))

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

        # self.sess = tf.Session(graph=self.infer_model.graph, config=utils.get_config_proto())
        
        # self.loaded_infer_model = model_helper.load_model(
        #     self.infer_model.model, self.ckpt, self.sess, 'infer')

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


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--out_dir', type=str, help="Store log/model files.", required=True)
    parser.add_argument("--hparams_path", type=str, default=None,
                        help=("Path to standard hparams json file that overrides"
                              "hparams values from FLAGS."))

    # Job info
    parser.add_argument("--jobid", type=int, default=0,
                        help="Task id of the worker.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers (inference only).")
    
    # Misc
    parser.add_argument("--override_loaded_hparams", type="bool", nargs="?",
                        const=True, default=False,
                        help="Override loaded hparams with values specified")

    # Inference
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--beam_width", type=int, default=0,
                        help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))

    # Vocab
    parser.add_argument("--sos", type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                        help="End-of-sentence symbol.")

    # SPM
    parser.add_argument("--subword_option", type=str, default="",
                        choices=["", "bpe", "spm"],
                        help="""\
                        Set to bpe or spm to activate subword desegmentation.\
                        """)

def create_hparams(flags):
    """Create training hparams."""
    return tf.contrib.training.HParams(
        out_dir=flags.out_dir,
        override_loaded_hparams=flags.override_loaded_hparams,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        sos=flags.sos if flags.sos else vocab_utils.SOS,
        eos=flags.eos if flags.eos else vocab_utils.EOS,
        subword_option=flags.subword_option,
    )

def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    print('starting... \n')
    chatbot = ChatBot()
    chatbot.run(FLAGS, default_hparams)

if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
