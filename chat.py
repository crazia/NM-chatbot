import tensorflow as tf
import argparse
import numpy as np
import math
import sys

# other module

import nmt

FLAGS = None

class ChatBot:
    def __init__(self):
        pass

    def _do_something(self, input):
        # print(input)
        return 'do something'


    def nmt_main(self, flags, default_hparams):
        # Job
        jobid = flags.jobid
        num_workers = flags.num_workers
        
        ## Train / Decode
        out_dir = flags.out_dir
        if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

        # Load hparams.
        hparams = nmt.create_or_load_hparams(
            out_dir, default_hparams, flags.hparams_path, save_hparams=(jobid == 0))

        ckpt = flags.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)

        if not ckpt:
            print('Train is needed')
            sys.exit()
    
    def run(self,flags, default_hparams):
        # load all parameters
        self.nmt_main(flags, default_hparams)
        try:
            sys.stdout.write("> ")
            sys.stdout.flush()
            line = sys.stdin.readline()

            while line:
                print(self._do_something(line.strip()))

                sys.stdout.write("\n> ")
                sys.stdout.flush()

                line = sys.stdin.readline()

        except KeyboardInterrupt:
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
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
