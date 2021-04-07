#Traffic Simulation
import numpy as np
from dataclasses import dataclass
import pygame
from typing import Tuple
from sys import exit
from os import environ

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = 'YES'

# Model parameters
n = 37  # number of cars
loop_length = 1
number_of_lanes = 4
close_dist = 0.05 #loop_length / 1.5 / n  # distance for hitting the brakes
stop_dist = 0.025
brakes = 0.2
acceleration = 0.1
max_speed = 1

# Constants of nature
RED = 255, 0, 0
YELLOW = 255, 255, 0
CYAN = 0, 255, 255
WHITE = 255, 255, 255
BLACK = 0, 0, 0

global counter # number of cars that have passed the endpoint
globals()['counter'] = 0

# Interaction sound
pygame.mixer.quit()
pygame.mixer.init()
OuchSnd = pygame.mixer.Sound('Ouch.wav')

# Canvas size & other GUI parameters - purely visual
@dataclass
class _gui:

    width: int = 800
    height: int = 800
    center: Tuple[int, int] = width // 2, height // 2
    radius: int = int(0.33 * min(width, height))
    line_width: int = 1
    line_color: Tuple[int, int, int] = WHITE
    # car_colors: Tuple[Tuple[int, int, int]] = ((RED,) + tuple(CYAN if k % 2 else YELLOW for k in range(n - 1)))
    car_radius: int = 4
    t: float = 0
    dt: float = 0
    running: bool = False
    paused: bool = True

@dataclass
class car:
    
    progress: float = 0
    velocity: float = 0
    lane: int = 0
    close_dist: float = close_dist
    acceleration: float = acceleration
    breaks: float = brakes
    max_speed: float = max_speed

def car_colors(cars):
    
    velocities = [cars[i].velocity for i in range(len(cars))]
    return (tuple( (int(255*velocities[k]/max_speed), 0, int(255*(1-velocities[k]/max_speed))) for k in range(n)))

def init(n):

    cars = np.array([car(progress = i/n*loop_length,
                         velocity = 0.1*np.random.random(),
                         lane = np.random.randint(number_of_lanes))
                         for i in range(n)])
    return cars

def dist_to_next(cars):
    
    progresses = np.array([cars[i].progress for i in range(len(cars))])
    lanes = np.array([cars[i].lane for i in range(len(cars))])
    
    ind = np.lexsort((progresses,lanes))
    Sorted = (np.array([(progresses[i],lanes[i]) for i in ind])).T
    
    same_lane = np.array(np.split(Sorted[0], np.where(np.diff(Sorted[1]))[0]+1, axis=0))
    
    distances = np.array([((np.roll(same_lane[i], -1) - same_lane[i] + loop_length) % loop_length) for i in range(np.shape(same_lane)[0])])
    dist = np.concatenate(distances)
    
    ind2 = np.argsort(ind)
    res = (np.array([(dist[i]) for i in ind2])).T
    return res

def spacing(progresses):
    
    ind = np.argsort(progresses)

    Sorted = (np.array([(progresses[i]) for i in ind])).T
    dist = (np.roll(Sorted, -1) - Sorted + loop_length) % loop_length
    ind2 = np.argsort(ind)
    res = (np.array([(dist[i]) for i in ind2])).T
    return res

def dist_top_lane(cars):
    
    progresses = np.array([cars[i].progress for i in range(len(cars))])
    lanes = np.array([cars[i].lane for i in range(len(cars))])
    
    ind = np.lexsort((progresses,lanes))
    Sorted = (np.array([(progresses[i],lanes[i]) for i in ind])).T
    
    same_lane = np.array(np.split(Sorted[0], np.where(np.diff(Sorted[1]))[0]+1, axis=0))
    
    adjacent_lanes = np.array([np.concatenate(
        [same_lane[i],
         same_lane[i-1] if i>0 else [] ]) for i in range(np.shape(same_lane)[0])])
    
    distances = np.array([spacing(adjacent_lanes[i])[:len(same_lane[i])] for i in range(np.shape(adjacent_lanes)[0])])
    dist = np.concatenate(distances)
    
    ind2 = np.argsort(ind)
    res = (np.array([(dist[i]) for i in ind2])).T
    return res

def dist_bottom_lane(cars):
    
    progresses = np.array([cars[i].progress for i in range(len(cars))])
    lanes = np.array([cars[i].lane for i in range(len(cars))])
    
    ind = np.lexsort((progresses,lanes))
    Sorted = (np.array([(progresses[i],lanes[i]) for i in ind])).T
    
    same_lane = np.array(np.split(Sorted[0], np.where(np.diff(Sorted[1]))[0]+1, axis=0))
    
    adjacent_lanes = np.array([np.concatenate(
        [same_lane[i],
         same_lane[i+1] if i<np.shape(same_lane)[0]-1 else []]) for i in range(np.shape(same_lane)[0])])
    
    distances = np.array([spacing(adjacent_lanes[i])[:len(same_lane[i])] for i in range(np.shape(adjacent_lanes)[0])])
    dist = np.concatenate(distances)
    
    ind2 = np.argsort(ind)
    res = (np.array([(dist[i]) for i in ind2])).T
    return res

# def dist_adjacent_lanes(cars):
    
#     progresses = np.array([cars[i].progress for i in range(len(cars))])
#     lanes = np.array([cars[i].lane for i in range(len(cars))])
    
#     ind = np.lexsort((progresses,lanes))
#     Sorted = (np.array([(progresses[i],lanes[i]) for i in ind])).T
    
#     same_lane = np.array(np.split(Sorted[0], np.where(np.diff(Sorted[1]))[0]+1, axis=0))
    
#     adjacent_lanes = np.array([np.concatenate(
#         [same_lane[i],
#          same_lane[i-1] if i>0 else [],
#          same_lane[i+1] if i<np.shape(same_lane)[0]-1 else []]) for i in range(np.shape(same_lane)[0])])
    
#     distances = np.array([spacing(adjacent_lanes[i])[:len(same_lane[i])] for i in range(np.shape(adjacent_lanes)[0])])
#     dist = np.concatenate(distances)
    
#     ind2 = np.argsort(ind)
#     res = (np.array([(dist[i]) for i in ind2])).T
#     return res

def change_lanes(cars):
    
    velocities = np.array([cars[i].velocity for i in range(len(cars))])
    lanes = np.array([cars[i].lane for i in range(len(cars))])
    
    dist_cars = dist_to_next(cars)
    
    switch_if = np.logical_and(np.array([dist_cars < close_dist]),
                       np.array([velocities > 0.05*max_speed]))
    
    switch_top = np.logical_and(switch_if,
        np.array([dist_cars <= dist_top_lane(cars)]))
    switch_bottom = np.logical_and(switch_if,
        np.array([dist_cars <= dist_bottom_lane(cars)]))
    both = np.logical_and(switch_top,switch_bottom)
    
    which = np.logical_or(switch_bottom,switch_top)
    
    lanes[tuple(both.tolist())] += np.random.choice((-1,1))
    lanes[tuple(np.logical_and(switch_top,np.logical_not(both)))] -= 1
    lanes[tuple(np.logical_and(switch_bottom,np.logical_not(both)))] += 1
    lanes[lanes == number_of_lanes] -= 2
    lanes[lanes == -1] += 2
    
    return lanes

def update(cars, dt):
    
    progresses = np.array([cars[i].progress for i in range(len(cars))])
    velocities = np.array([cars[i].velocity for i in range(len(cars))])
    lanes = np.array([cars[i].lane for i in range(len(cars))])
    
    v_prev = velocities.copy() # copy for sound
    dist_cars = dist_to_next(cars)
    close, slow = dist_cars < close_dist, velocities < max_speed
    acc = -brakes * (close) + acceleration * (slow & ~close)
    lanes = change_lanes(cars)
    
    velocities += 1e-3 * acc * dt
    velocities[velocities < 0] = 0
    velocities[velocities > max_speed] = max_speed
    progresses += dt * 1e-3 * velocities
    stop = np.array([dist_cars <= stop_dist])
    velocities[tuple(stop.tolist())] = 0
    globals()['counter'] += np.count_nonzero(progresses>=loop_length)
    progresses %= loop_length
    
    # # 'Ouch' Sound
    # if np.any(v_prev-velocities > 2*brakes):
    #     OuchSnd.play()
    
    for i in range(len(cars)):
        cars[i].progress = progresses[i]
        cars[i].velocity = velocities[i]
        cars[i].lane = lanes[i]
    
    # dist_cars = dist_to_next(cars)
    
    return cars

def weird_division(n, d):
    return n / d if d else 0

def setup_gui(**kwargs):

    velocities = np.array([cars[i].velocity for i in range(len(cars))])
    cars_passed = globals()['counter']
        
    gui = _gui(**kwargs)
    pygame.init()
    gui.screen = pygame.display.set_mode((gui.width, gui.height))
    pygame.display.set_caption('cars on a Ring')
    gui.font = pygame.font.SysFont('Courier New, courier, monospace', 14, bold=True)
    gui.text = gui.font.render(
        f'N = {n}; average velocity: {np.array(velocities).mean()*100:.3f} %/s; stream: {weird_division(cars_passed,gui.t)*1000:.3f} cars/s; time: {gui.t/1000} s', True, WHITE
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

# Drawing dashed lines function
def draw_line_dashed(surface, color, start_pos, end_pos, width = 1, dash_length = 10, exclude_corners = True):

    # convert tuples to numpy arrays
    start_pos = np.array(start_pos)
    end_pos   = np.array(end_pos)

    # get euclidian distance between start_pos and end_pos
    length = np.linalg.norm(end_pos - start_pos)

    # get amount of pieces that line will be split up in (half of it are amount of dashes)
    dash_amount = int(length / dash_length)

    # x-y-value-pairs of where dashes start (and on next, will end)
    dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()

    return [pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n+1]), width)
            for n in range(int(exclude_corners), dash_amount - int(exclude_corners), 2)]

def paint_intro(gui):

    gui.screen.fill(BLACK)
    intro_text1 = gui.font.render('Click to start simulation', True, YELLOW)
    intro_text2 = gui.font.render(
        'when running, mouse click pauses/resumes', True, YELLOW
    )
    gui.screen.blit(intro_text1, (20, gui.height // 3))
    gui.screen.blit(intro_text2, (20, gui.height * 2 // 5))
    pygame.display.flip()


def paint(cars, gui):

    progresses = np.array([cars[i].progress for i in range(len(cars))])
    velocities = np.array([cars[i].velocity for i in range(len(cars))])
    lanes = np.array([cars[i].lane for i in range(len(cars))])
    
    gui.screen.fill(BLACK)
    cars_passed = globals()['counter']
    if gui.t > 5e2:
        gui.text = gui.font.render(
            f'Number of cars = {n}; average velocity: {np.array(velocities).mean()*100:.3f} %/s; stream: {weird_division(cars_passed,gui.t)*1000:.3f} cars/s; time: {gui.t/1000} s', True, WHITE
        )
        gui.time = 0
    gui.screen.blit(gui.text, (10, 10))
    
    # Draw lanes
    pygame.draw.line(gui.screen,gui.line_color,(0,gui.center[1]+10*number_of_lanes),(gui.width,gui.center[1]+10*number_of_lanes),gui.line_width)
    pygame.draw.line(gui.screen,gui.line_color,(0,gui.center[1]-10*number_of_lanes),(gui.width,gui.center[1]-10*number_of_lanes),gui.line_width)
    
    for i in range(number_of_lanes):
        draw_line_dashed(gui.screen,gui.line_color,(0,gui.center[1]-10*number_of_lanes+20*i),(gui.width,gui.center[1]-10*number_of_lanes+20*i),gui.line_width)
    
    #rect(gui.screen, gui.line_color, (gui.center[0]-gui.radius,gui.center[1]-gui.radius,2*gui.radius,2*gui.radius), gui.line_width)
    # x = gui.center[0] + (gui.radius * np.cos(cars[0])/(np.maximum(np.abs(np.cos(cars[0])),np.abs(np.sin(cars[0]))))).astype('int')
    # y = gui.center[1] - (gui.radius * np.sin(cars[0])/(np.maximum(np.abs(np.cos(cars[0])),np.abs(np.sin(cars[0]))))).astype('int')
    
    #circle(gui.screen, gui.line_color, gui.center, gui.radius, gui.line_width)
    
    x = (progresses*gui.width / loop_length).astype('int')
    y = (gui.center[1]+10-10*number_of_lanes+20*lanes)
    xy = np.array((x, y)).T
    for color, pos in zip(car_colors(cars), xy):
        pygame.draw.circle(gui.screen, color, pos, gui.car_radius)
    pygame.display.flip()


def run(cars, gui):

    while not gui.running:
        gui.clock.tick(5)
        paint_intro(gui)
        handle_events(gui)

    while gui.running:
        if not gui.paused:
            gui.t += gui.clock.get_time()
            gui.dt = gui.clock.tick(60)
            cars = update(cars, gui.dt)
        else:
            gui.clock.tick(6)
        paint(cars, gui)
        handle_events(gui)


if __name__ == '__main__':

    cars = init(n)
    gui = setup_gui()
    # you can make an icon if you feel creative
    # icon = pygame.image.load('multicars_icon.png')
    # pygame.display.set_icon(icon)
    run(cars, gui)
