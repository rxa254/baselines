import gym
import signal
import sys

from baselines import deepq

####  this catches the Ctrl-C
def signal_handler(signal, frame):
        print('...user interrupt detected...')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
###


def main():
    env = gym.make("CartPole-v0")
    act = deepq.load("cartpole_model.pkl")

    k = 1
    while k < 1e5:
        obs, done = env.reset(), False
        episode_rew = 0
        nsteps = 0
        while nsteps < 100:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
            k += 1
            nsteps += 1
        print("Episode " + str(k) + " reward = ", episode_rew)
        print("{0:2.3f} {1:2.3f} {2:2.3f} {3:2.3f}".format(obs[0], obs[1], obs[2], obs[3]))


if __name__ == '__main__':
    main()
