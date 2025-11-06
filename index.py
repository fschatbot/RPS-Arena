import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

class Entity:
	def __init__(self, x:int, y:int, rps:str) -> None:
		self.x = x
		self.y = y
		self.rps = rps  # rock, paper, scissors
		assert rps in ['rock', 'paper', 'scissors'], "rps must be 'rock', 'paper', or 'scissors'"
	
	def update(self):
		...

class Universe:
	def __init__(self, display: bool = False):
		self.objects = []
		self.display = display
		if self.display:
			pygame.init()
			self.screen = pygame.display.set_mode((800, 600))
			pygame.display.set_caption("Universe Simulation")
			self.clock = pygame.time.Clock()

	def update(self):
		for obj in self.objects:
			obj.update()

	def draw(self, screen):
		for obj in self.objects:
			obj.draw(screen)