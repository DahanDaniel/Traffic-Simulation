#Multiple beads on a loop
import numpy as np
from dataclasses import dataclass
# from os import environ
from typing import Tuple
from sys import exit

# environ["PYGAME_HIDE_SUPPORT_PROMPT"] = 'YES'

# constants of nature
TWO_PI = 2 * np.pi
RED = 255, 0, 0
YELLOW = 255, 255, 0
CYAN = 0, 255, 255
WHITE = 255, 255, 255
BLACK = 0, 0, 0

# model parameters
n = 37  # number of beads
close_dist = TWO_PI / 1.5 / n  # distance for hitting the brakes
stop_dist = 0.05
brakes = 0.2
acceleration = 0.1
max_speed = 1.0

# canvas size & other GUI parameters - purely visual


@dataclass
class _gui:

    width: int = 400
    height: int = 400
    center: Tuple[int, int] = width // 2, height // 2
    radius: int = int(0.4 * min(width, height))
    line_width: int = 1
    line_color: Tuple[int, int, int] = WHITE
    bead_colors: Tuple[Tuple[int, int, int]] = (
        (RED,) + tuple(CYAN if k % 2 else YELLOW for k in range(n - 1))
    )
    bead_radius: int = 4
    t: float = 0
    dt: float = 0
    running: bool = False
    paused: bool = True


def init(n):

    beads = np.array(
        (
            np.linspace(0, TWO_PI, n, endpoint=0),  # positions
            0.1 * np.random.random((n,)),  # velocities
        )
    )
    return beads


def spacing(beads):

    phi = beads[0]
    return (np.roll(phi, -1) - phi + TWO_PI) % TWO_PI


def update(beads, dt):

    phi, v = beads
    close, slow = spacing(beads) < close_dist, v < max_speed
    acc = -brakes * (close) + acceleration * (slow & ~close)
    v += 1e-3 * acc * dt
    v[v < 0] = 0
    phi += dt * 1e-3 * v
    v[spacing(beads) < stop_dist] = 0
    phi %= TWO_PI
    return beads


def setup_gui(**kwargs):

    gui = _gui(**kwargs)
    pygame.init()
    gui.screen = pygame.display.set_mode((gui.width, gui.height))
    pygame.display.set_caption('Beads on a Ring')
    gui.font = pygame.font.SysFont('Courier New, courier, monospace', 14, bold=True)
    gui.text = gui.font.render(
        f'N = {n}; average velocity: {beads[1].mean():.3f} rad/s', True, WHITE
    )
    gui.clock = pygame.time.Clock()
    return gui


def handle_events(gui):

    for event in pygame.event.get():
        type = event.type
        if type == pygame.QUIT:
            pygame.quit()
            exit()
        elif type == pygame.MOUSEBUTTONUP and event.button == 1:
            if not gui.running:
                gui.running = True
                gui.clock.tick()
            else:
                gui.paused = not gui.paused
                if not gui.paused:
                    gui.clock.tick()


def paint_intro(gui):

    gui.screen.fill(BLACK)
    intro_text1 = gui.font.render('Click to start simulation', True, YELLOW)
    intro_text2 = gui.font.render(
        'when running, mouse click pauses/resumes', True, YELLOW
    )
    gui.screen.blit(intro_text1, (20, gui.height // 3))
    gui.screen.blit(intro_text2, (20, gui.height * 2 // 5))
    pygame.display.flip()


def paint(beads, gui):

    gui.screen.fill(BLACK)
    if gui.t > 5e2:
        gui.text = gui.font.render(
            f'N = {n}; average velocity: {beads[1].mean():.3f} rad/s', True, WHITE
        )
        gui.time = 0
    gui.screen.blit(gui.text, (10, 10))
    pygame.draw.circle(
        gui.screen, gui.line_color, gui.center, gui.radius, gui.line_width
    )

    x = gui.center[0] + (gui.radius * np.cos(beads[0])).astype('int')
    y = gui.center[1] - (gui.radius * np.sin(beads[0])).astype('int')
    xy = np.array((x, y)).T
    for color, pos in zip(gui.bead_colors, xy):
        pygame.draw.circle(gui.screen, color, pos, gui.bead_radius)
    pygame.display.flip()


def run(beads, gui):

    while not gui.running:
        gui.clock.tick(5)
        paint_intro(gui)
        handle_events(gui)

    while gui.running:
        if not gui.paused:
            gui.t += gui.clock.get_time()
            gui.dt = gui.clock.tick(60)
            beads = update(beads, gui.dt)
        else:
            gui.clock.tick(6)
        paint(beads, gui)
        handle_events(gui)


if __name__ == '__main__':

    import pygame
    beads = init(n)
    gui = setup_gui()
    # you can make an icon if you feel creative
    # icon = pygame.image.load('multibeads_icon.png')
    # pygame.display.set_icon(icon)
    run(beads, gui)
