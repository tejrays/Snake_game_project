import time
import subprocess
from snake_env_discrete import SnakeEnvDiscrete
from qlearning import QLearningAgent
from helper import plot


def run_qlearning(episodes=100):
    env = SnakeEnvDiscrete()
    agent = QLearningAgent()
    scores = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            action = agent.get_action(state)
            new_state, reward, done, score = env.step(action)
            agent.update(state, action, reward, new_state)
            state = new_state
            total = score
        scores.append(total)
        avg = [sum(scores[:i+1])/(i+1) for i in range(len(scores))]
        plot(scores, avg)
        print(f'Episode {ep+1}/{episodes} | Score: {total}')
    return scores


def run_dqn():
    print('Starting DQN (agent.py)...')
    subprocess.run(['python', 'agent.py'])


if __name__ == '__main__':
    qscores = run_qlearning(episodes=50)
    time.sleep(1)
    run_dqn()
