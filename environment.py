import pygame


class Obstacles:
    def __init__(self, obstacles: list, beacons: list):
        self.color = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        self.obstacles = obstacles
        self.beacons = beacons
        self.obs = []
        for i in range(len(self.obstacles)):
            self.obs.append(pygame.Rect(*self.obstacles[i]))

    def checkInCircle(self, robot):
        self.inCircle = []
        for i in range(len(self.beacons)):
            if (robot.x - self.beacons[i][0]) * (robot.x - self.beacons[i][0]) + (
                robot.y - self.beacons[i][1]
            ) * (robot.y - self.beacons[i][1]) <= 200 * 200:
                self.inCircle.append(self.beacons[i])
                self.color[i] = (255, 0, 0)
            else:
                self.inCircle.append(None)
                self.color[i] = (0, 0, 0)
        return self.inCircle

    def draw(self, screen, robot):
        for i in range(len(self.obstacles)):
            pygame.draw.rect(screen, (0, 0, 0), self.obs[i])
        self.bcns = []
        for i in range(len(self.beacons)):
            self.checkInCircle(robot)
            pygame.draw.circle(screen, self.color[i], (self.beacons[i]), 10)
            pygame.draw.circle(screen, self.color[i], (self.beacons[i]), 200, 2)
