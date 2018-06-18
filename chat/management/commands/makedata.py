from django.core.management.base import BaseCommand, CommandError
from chat.models import Chat
from core.chat import remove_special_char, apply_nlpy
import subprocess
import os
import time


TRAIN_INPUT = 'train.req'
TRAIN_OUTPUT = 'train.rep'

TEST_INPUT = 'test.req'
TEST_OUTPUT = 'test.rep'

VOCAB_REQ = 'vocab.req'
VOCAB_REP = 'vocab.rep'

# password setting for systemctl restart chabot.service
PASSWORD = 'password for my linux account'
CHATBOT = 'chatbot.service'

class Command(BaseCommand):
    help = '교육용 자료 만들기'

    # make train data
    def make_train_data(self, t_input, t_output):
        chats = Chat.objects.filter (verify=Chat.VERIFY_OK)

        req = open(t_input, 'w')
        rep = open(t_output, 'w')

        for chat in chats:
            req.write(apply_nlpy(remove_special_char(chat.question))+'\n')
            rep.write(chat.answer +'\n')
        
        req.close()
        rep.close()


    def run_cmd_sp(self, cmd):
        try:
            result = subprocess.Popen(cmd, shell=True)
            os.waitpid(result.pid, 0)
        except Exception as e:
            print('subprocess error ' + e.args)
        
        

    def make_vocab(self, t_input, v_output):
        cmd = f'python ./core/bin/generate_vocab.py < {t_input} > {v_output}'
        self.run_cmd_sp(cmd)


    def run_cmd(self, cmd):
        args = cmd.split()

        try:
            result = subprocess.check_output(args)
        except Exception as e:
            print('subprocess error ' + e.args)
        
    
    def add_arguments(self, parser):
        # Named (optional) arguments
        parser.add_argument(
            '--initial',
            action='store_true',
            dest='initial',
            default=False,
            help='make initial setting',
        )

    def handle(self, *args, **options):

        if options['initial']:
            # make initia setting
            print('making initial setting for launch..')
            cmd = f'cd data && rm -rf current && ln -s initial current && cd ..'
            self.run_cmd_sp(cmd)
            
            return

        # make train data and train it
        self.make_train_data(TRAIN_INPUT, TRAIN_OUTPUT)

        # vocabulary
        self.make_vocab(TRAIN_INPUT, VOCAB_REQ)
        self.make_vocab(TRAIN_OUTPUT, VOCAB_REP)

        # make directory
        folder = time.strftime('%Y-%m-%d')

        cmd = f'mkdir -p ./data/{folder}'
        self.run_cmd(cmd)

        # copy train -> dev
        cmd = f'cp {TRAIN_INPUT} {TEST_INPUT}'
        self.run_cmd(cmd)
        cmd = f'cp {TRAIN_OUTPUT} {TEST_OUTPUT}'
        self.run_cmd(cmd)

        # move all to destination
        cmd = f'mv {TRAIN_INPUT} {TRAIN_OUTPUT} {TEST_INPUT} {TEST_OUTPUT} {VOCAB_REQ} {VOCAB_REP} ./data/{folder}/'
        self.run_cmd(cmd)

        # running train
        cmd = f'python -m core.nmt --attention=scaled_luong --src=req --tgt=rep --vocab_prefix=./data/{folder}/vocab --train_prefix=./data/{folder}/train --dev_prefix=./data/{folder}/test --test_prefix=./data/{folder}/test --out_dir=./data/{folder}/model --num_train_steps=12000 --steps_per_stats=100 --num_layers=4 --num_units=128 --dropout=0.5 --metrics=bleu'
        self.run_cmd(cmd)

        # link current - {folder}
        cmd = f'cd data && rm -rf current && ln -s {folder} current && cd ..'
        self.run_cmd_sp(cmd)

        # restart chabot service 
        cmd = f'echo {PASSWORD} | sudo -S systemctl restart {CHATBOT}'
        self.run_cmd_sp(cmd)
