import gym
import signal
import sys
import os
import numpy as np

from baselines import deepq

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-fn", "--filename", type=str, default = "FPcav_model.pkl",
                    help="Deep Q Learning model file")
args = parser.parse_args()



####  this catches the Ctrl-C
def signal_handler(signal, frame):
        print('...user interrupt detected...')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
###


def main():
    env    = gym.make("simplePendulum-v1")
    funame = args.filename
    act    = deepq.load(funame)
    otim   = os.stat(funame).st_mtime

    while True:
        mtim = os.stat(funame).st_mtime
        if mtim != otim:
            #act = None
            
            #act = deepq.load(funame)
            print("Loaded new controller...")
        obs, done = env.reset(), False
        episode_rew = 0
        nsteps = 0
        while nsteps < 500:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
            nsteps += 1
        print("Episode reward = ", round(episode_rew,2))
        #print("Angle = {0:2.2f} deg, Vel = {1:2.2f} deg/s, Torque = {2:2.2f} N/m".format(180/np.pi*(obs[0]), 180/np.pi*obs[1], obs[2]))
        print("Angle = {0:2.2f} deg, Vel = {1:2.2f} deg/s".format(
            180/np.pi*np.arccos(obs[0]), 180/np.pi*obs[2]))
        print("  ")


if __name__ == '__main__':
    main()
