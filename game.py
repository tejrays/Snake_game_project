import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', ['x', 'y'])

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 90, 255)
BLACK = (0, 0, 0)

BLOCK = 20
GAME_SPEED = 40


class SnakeGameAI:
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)

        self.snake = [
            self.head,
            Point(self.head.x - BLOCK, self.head.y),
            Point(self.head.x - 2 * BLOCK, self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (self.h - BLOCK) // BLOCK) * BLOCK
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        done = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return reward, done, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._render()
        self.clock.tick(GAME_SPEED)
        return reward, done, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt.x < 0 or pt.x > self.w - BLOCK or pt.y < 0 or pt.y > self.h - BLOCK:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def _render(self):
        self.display.fill(BLACK)

        for block in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(block.x, block.y, BLOCK, BLOCK))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(block.x + 4, block.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK, BLOCK))

        score = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score, (0, 0))

        pygame.display.flip()

    def _move(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clockwise[idx]
        elif action == [0, 1, 0]:
            new_dir = clockwise[(idx + 1) % 4]
        else:
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK
        elif self.direction == Direction.LEFT:
            x -= BLOCK
        elif self.direction == Direction.DOWN:
            y += BLOCK
        elif self.direction == Direction.UP:
            y -= BLOCK

        self.head = Point(x, y)


if __name__ == '__main__':
    game = SnakeGameAI()
    while True:
        game.play_step([1, 0, 0])
