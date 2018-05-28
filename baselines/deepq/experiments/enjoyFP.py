import gym
import signal
import sys
import os
from numpy import arccos

from baselines import deepq

####  this catches the Ctrl-C
def signal_handler(signal, frame):
        print('...user interrupt detected...')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
###


def main():
    env    = gym.make("simplePendulum-v1")
    funame = "FPcav_model.pkl"
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
        print("Angle = {0:2.2f}, Vel = {1:2.2f}".format(arccos(obs[0]), obs[2]))
        print("  ")


if __name__ == '__main__':
    main()
