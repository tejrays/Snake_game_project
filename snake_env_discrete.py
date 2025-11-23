import numpy as np
from game import SnakeGameAI, Direction, Point


class SnakeEnvDiscrete:
    """Wrapper converting SnakeGameAI to a discrete state representation.

    State: (danger_forward, danger_left, danger_right, food_dir, dir_index)
    Actions: 0=straight,1=right,2=left
    """

    def __init__(self):
        self.env = SnakeGameAI()

    def reset(self):
        self.env.reset()
        return self._get_state()

    def step(self, action: int):
        one_hot = [1 if action == i else 0 for i in range(3)]
        reward, done, score = self.env.play_step(one_hot)
        state = self._get_state()
        return state, reward, done, score

    def _get_state(self):
        head = self.env.snake[0]

        left = Point(head.x - 20, head.y)
        right = Point(head.x + 20, head.y)
        up = Point(head.x, head.y - 20)
        down = Point(head.x, head.y + 20)

        dir_right = self.env.direction == Direction.RIGHT
        dir_left = self.env.direction == Direction.LEFT
        dir_up = self.env.direction == Direction.UP
        dir_down = self.env.direction == Direction.DOWN

        danger_forward = int(
            (dir_right and self.env.is_collision(right)) or
            (dir_left and self.env.is_collision(left)) or
            (dir_up and self.env.is_collision(up)) or
            (dir_down and self.env.is_collision(down))
        )

        danger_right = int(
            (dir_up and self.env.is_collision(right)) or
            (dir_down and self.env.is_collision(left)) or
            (dir_left and self.env.is_collision(up)) or
            (dir_right and self.env.is_collision(down))
        )

        danger_left = int(
            (dir_down and self.env.is_collision(right)) or
            (dir_up and self.env.is_collision(left)) or
            (dir_right and self.env.is_collision(up)) or
            (dir_left and self.env.is_collision(down))
        )

        food = self.env.food
        if food.x < head.x:
            food_dir = 0  # left
        elif food.x > head.x:
            food_dir = 1  # right
        elif food.y < head.y:
            food_dir = 2  # up
        else:
            food_dir = 3  # down

        dir_idx = {
            Direction.RIGHT: 0,
            Direction.LEFT: 1,
            Direction.UP: 2,
            Direction.DOWN: 3,
        }[self.env.direction]

        return (danger_forward, danger_left, danger_right, food_dir, dir_idx)
