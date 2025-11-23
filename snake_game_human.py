import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font(None, 25)

class Move(Enum):
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

CELL = 20
FPS = 10

class SnakeGameManual:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Move.RIGHT
        cx, cy = self.width // 2, self.height // 2
        self.head = Point(cx, cy)
        self.body = [
            self.head,
            Point(cx - CELL, cy),
            Point(cx - 2 * CELL, cy)
        ]
        self.score = 0
        self.food = None
        self._spawn_food()

    def _spawn_food(self):
        x = random.randint(0, (self.width - CELL) // CELL) * CELL
        y = random.randint(0, (self.height - CELL) // CELL) * CELL
        self.food = Point(x, y)
        if self.food in self.body:
            self._spawn_food()

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Move.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Move.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Move.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Move.DOWN

        self._move(self.direction)
        self.body.insert(0, self.head)

        if self._has_crashed():
            return True, self.score

        if self.head == self.food:
            self.score += 1
            self._spawn_food()
        else:
            self.body.pop()

        self._draw_frame()
        self.clock.tick(FPS)
        return False, self.score

    def _has_crashed(self):
        if self.head.x < 0 or self.head.x >= self.width or self.head.y < 0 or self.head.y >= self.height:
            return True
        if self.head in self.body[1:]:
            return True
        return False

    def _draw_frame(self):
        self.surface.fill(BLACK)
        for segment in self.body:
            pygame.draw.rect(self.surface, BLUE1, pygame.Rect(segment.x, segment.y, CELL, CELL))
            pygame.draw.rect(self.surface, BLUE2, pygame.Rect(segment.x + 4, segment.y + 4, 12, 12))
        pygame.draw.rect(self.surface, RED, pygame.Rect(self.food.x, self.food.y, CELL, CELL))
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.surface.blit(score_text, (0, 0))
        pygame.display.flip()

    def _move(self, direction):
        x, y = self.head.x, self.head.y
        if direction == Move.RIGHT:
            x += CELL
        elif direction == Move.LEFT:
            x -= CELL
        elif direction == Move.DOWN:
            y += CELL
        elif direction == Move.UP:
            y -= CELL
        self.head = Point(x, y)

if __name__ == '__main__':
    game = SnakeGameManual()
    while True:
        ended, score = game.play_step()
        if ended:
            print('Final Score:', score)
            break
    pygame.quit()
