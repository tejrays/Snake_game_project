import random
import numpy as np
from collections import deque
import torch
from model import LinearQNet, QTrainer
from game import SnakeGameAI, Point, Direction
from helper import plot

MEMORY_SIZE = 100_000
BATCH_SIZE = 1000


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
        head = game.snake[0]

        # Adjacent points (using Point instead of tuples)
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        moving_left = game.direction == Direction.LEFT
        moving_right = game.direction == Direction.RIGHT
        moving_up = game.direction == Direction.UP
        moving_down = game.direction == Direction.DOWN

        danger_straight = (
            (moving_right and game.is_collision(point_r)) or
            (moving_left and game.is_collision(point_l)) or
            (moving_up and game.is_collision(point_u)) or
            (moving_down and game.is_collision(point_d))
        )
        danger_right = (
            (moving_up and game.is_collision(point_r)) or
            (moving_down and game.is_collision(point_l)) or
            (moving_left and game.is_collision(point_u)) or
            (moving_right and game.is_collision(point_d))
        )
        danger_left = (
            (moving_down and game.is_collision(point_r)) or
            (moving_up and game.is_collision(point_l)) or
            (moving_right and game.is_collision(point_u)) or
            (moving_left and game.is_collision(point_d))
        )

        food_left = game.food.x < head.x
        food_right = game.food.x > head.x
        food_up = game.food.y < head.y
        food_down = game.food.y > head.y

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(moving_left),
            int(moving_right),
            int(moving_up),
            int(moving_down),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(0, 80 - self.n_games)  # exploration
        move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()

        move[action] = 1
        return move


def train(max_games=300):
    scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while agent.n_games < max_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            scores.append(score)
            total_score += score
            avg = total_score / agent.n_games

            print(f"Game {agent.n_games} | Score: {score} | Record: {record} | Avg: {avg:.2f}")
            plot(scores, [sum(scores[:i+1])/(i+1) for i in range(len(scores))])


if __name__ == '__main__':
    train()