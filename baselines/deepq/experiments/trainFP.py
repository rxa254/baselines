'''
openAI/gym training environment

https://gym.openai.com/docs/
'''
import gym

from baselines import deepq
import signal
import sys

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learningrate", type=float, default = 1e-3,
                    help="Learning Rate for Backprop")
parser.add_argument("-ms", "--maxsteps", type=float, default = 1e5,
                    help="How many training steps?")
args = parser.parse_args()


####  this catches the Ctrl-C
def signal_handler(signal, frame):
        print('...user interrupt detected...')
        #print("Saving model to FPcav_model.pkl")
        #act.save("FPcav_model.pkl")
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
###

def callback(lcl, _glb):
    # stop training if reward exceeds -100
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 100
    return is_solved


def main():
    funame = "FPcav_model.pkl"
    env    = gym.make("simplePendulum-v1")
    model  = deepq.models.mlp([64,64])
    act    = deepq.learn(
        env,
        q_func                = model,
        lr                    = args.learningrate,
        max_timesteps         = int(args.maxsteps),
        buffer_size           = int(args.maxsteps/2),
        exploration_fraction  = 0.5,
        exploration_final_eps = 0.02,
        print_freq            = 100,
        callback              = callback
    )
    print("Saving model to " + funame)
    act.save(funame)


if __name__ == '__main__':
    main()
