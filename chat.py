import tensorflow as tf
import numpy as np
import math
import sys


class ChatBot:
    def __init__(self):
        pass


    def _do_something(self, input):
        # print(input)
        return 'do something'
    
    def run(self):
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


def main(_):
    print('starting... \n')
    chatbot = ChatBot()
    chatbot.run()

if __name__ == "__main__":
    tf.app.run()
