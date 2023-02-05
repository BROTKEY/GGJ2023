import time
import mido
from mido import MidiFile, Message, MetaMessage
import numpy as np
from numpy import random

class MidiPlayer:
    def __init__(self, filename):
        self.midi = MidiFile(filename)
        self.speed = 1.0
        self.accuracy = 1.0
        self.should_stop = False
        self.output = mido.open_output()
    
    def set_accuracy(self, acc):
        self.accuracy = acc
        print(f'received acc = {acc}')

    def stop(self):
        self.should_stop = True

    def run(self):
        while not self.should_stop:
            start_time = time.time()
            input_time = 0.0

            for msg in self.midi:
                input_time += msg.time

                playback_time = time.time() - start_time
                duration_to_next_event = input_time - playback_time
                # print(duration_to_next_event)
                
                acc_speed = 1.0
                acc_inv = 1 - self.accuracy
                if self.accuracy < 1.0:
                    #acc_speed = np.clip(random.normal(scale=0.8 * (1 - self.accuracy)), -0.8, 0.8)
                    acc_inv *= 0.5
                    acc_speed = random.uniform(-acc_inv, acc_inv)
                if duration_to_next_event > 0:
                    time.sleep(duration_to_next_event / (self.speed + acc_speed))

                if isinstance(msg, MetaMessage):
                    continue
                else:
                    if self.accuracy < 0.9 and msg.type == 'note_on':
                        print(msg.note)
                        x = random.rand()
                        if (x > self.accuracy):
                            msg.note += random.randint(-1, 2)
                    self.output.send(msg)


if __name__ == '__main__':
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument('filename')
    args.add_argument('--speed', '-s', type=float, default=1.0)
    args.add_argument('--accuracy', '-a', type=float, default=1.0)
    args = args.parse_args()
    player = MidiPlayer(args.filename)
    player.speed = args.speed
    player.set_accuracy(args.accuracy)
    player.run()