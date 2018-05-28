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
parser.add_argument("-ms", "--maxsteps", type=float, default = int(1e5),
                    help="How many training steps?")
args = parser.parse_args()


####  this catches the Ctrl-C
def signal_handler(signal, frame):
        print('...user interrupt detected...')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
###

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    model = deepq.models.mlp([64, 64])
    act = deepq.learn(
        env,
        q_func                = model,
        lr                    = args.learningrate,
        max_timesteps         = int(args.maxsteps),
        buffer_size           = int(args.maxsteps/2),
        exploration_fraction  = 0.95,
        exploration_final_eps = 0.02,
        print_freq            = 100,
        callback              = callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
