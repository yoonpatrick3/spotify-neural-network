import pygame, math, sys
import pygame_textinput
import pygame.freetype
import audio_neural_network as ann
from audio_neural_network import create_baseline_gender
from audio_neural_network import create_baseline_regression


features = ['Duration', 'Key', 'Mode', 'Time Signature', 'Acousticness', 'Danceability', 
                        'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Tempo']
maxis = [300000, 11, 1, 5, 1, 1, 1, 1, 1, 0, 1, 1, 250]
minis = [0, -1, 0, 0, 0, 0, 0, 0, 0, -30.0, 0, 0, 0]

pygame.init()

model = 1

X = 1120  # screen width
Y = 620  # screen height

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
TRANS = (1, 1, 1)

predict_gender = True
changed_numbers = True
initialized = False
model = ann.unpickle_gender()

def ms_secs(num):
    seconds = math.trunc(num / 1000)
    minutes = math.trunc(seconds / 60)
    leftOver = math.trunc(seconds % 60)
    text = "Duration: " + str(minutes) + " minute(s), " + str(leftOver) + " second(s)"
    return text
            
def show_text(f):
    font = pygame.font.SysFont("Verdana", 30)
    height = 40
    global changed_numbers
    global model
    if initialized:
        if predict_gender and changed_numbers:
            model = ann.unpickle_gender()
            changed_numbers = False
        elif changed_numbers:
            model = ann.unpickle_pop()
            changed_numbers = False
        outcome = ann.predict(f, model)
    else:
        outcome = "NaN"
    for i in range(len(features)):
        if i == 0:
            text = ms_secs(f[i])
        else:
            text = features[i] + ": " + str(round(f[i], 3))
        textSurface = font.render(text, True, WHITE)
        textRect = textSurface.get_rect()
        textRect.center = (X/2+ 50, height)
        height += 30
        screen.blit(textSurface, textRect)
    
    if initialized:
        if predict_gender:
            if outcome == 1:
                text = 'Male'
            else:
                text = 'Female'
        else:
            text = str(round(outcome[0], 5))
    else:
        text = outcome
    textSurface = font.render(text, True, WHITE)
    textRect = textSurface.get_rect()
    textRect.center = (X/2+ 50, height)
    height += 30
    screen.blit(textSurface, textRect)
    song_text = ann.sounds_closest_to(f)
    textSurface = font.render(song_text, True, (237, 111, 139))
    textRect = textSurface.get_rect()
    textRect.center = (X/2+ 50, height)
    screen.blit(textSurface, textRect)

class Button():
    def __init__(self, text, location, action, font_name="Verdana", font_size=30):
        self.color = (66, 164, 245)
        self.bg = (66, 164, 245)
        self.fg = WHITE
        self.font = pygame.font.SysFont(font_name, font_size)
        self.txt = text
        self.txt_surf = self.font.render(self.txt, 1, self.fg)
        text_width, text_height = self.font.size(text)
        self.size = (text_width+20, text_height+20)
        self.txt_rect = self.txt_surf.get_rect(center=[s//2 for s in self.size])
        self.surface = pygame.surface.Surface(self.size)
        self.rect = self.surface.get_rect(center=location)
        self.call_back_ = action

    def draw(self):
        self.mouseover()

        self.surface.fill(self.bg)
        self.surface.blit(self.txt_surf, self.txt_rect)
        screen.blit(self.surface, self.rect)
    
    def mouseover(self):
        self.bg = self.color

        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            self.bg = (66, 245, 236)

    def call_back(self):
        self.call_back_()

def change_outcome():
    global predict_gender
    global changed_numbers
    predict_gender = not predict_gender
    changed_numbers = True

def initialize_training():
    global initialized
    global changed_numbers
    changed_numbers = True
    initialized = True

def mousebuttondown():
    pos = pygame.mouse.get_pos()
    for button in buttons:
        if button.rect.collidepoint(pos):
            button.call_back()

class Slider():

    def __init__(self, i, val, maxi, mini, pos):
        self.i = i
        if self.i == 2 or self.i == 1 or self.i==3:
            self.val = math.trunc(val)
        else:
            self.val = val  # start value
        self.maxi = maxi  # maximum at slider position right
        self.mini = mini  # minimum at slider position left
        if i < 7:
            self.xpos = 0  # x-location on screen
            self.ypos = pos
        else:
            self.xpos = 100
            self.ypos = pos - 350
        self.surf = pygame.surface.Surface((100, 50))
        self.hit = False  # the hit attribute indicates slider movement due to mouse interaction
        name = features[i]
        if i == 7:
            font = pygame.font.SysFont("Verdana", 10)
        else:
            font = pygame.font.SysFont("Verdana", 12)
        self.txt_surf = font.render(name, 1, BLACK)
        self.txt_rect = self.txt_surf.get_rect(center=(50, 15))
        # Static graphics - slider background #
        self.surf.fill((100, 100, 100))
        pygame.draw.rect(self.surf, GREY, [0, 0, 100, 50], 3)
        pygame.draw.rect(self.surf, WHITE, [10, 30, 80, 5], 0)

        self.surf.blit(self.txt_surf, self.txt_rect)  # this surface never changes

        # dynamic graphics - button surface #
        self.button_surf = pygame.surface.Surface((20, 20))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.circle(self.button_surf, BLACK, (10, 10), 6, 0)
        pygame.draw.circle(self.button_surf, WHITE, (10, 10), 4, 0)

    def draw(self):
        """ 
        Combination of static and dynamic graphics in a copy of
        the basic slide surface
        """
        # static
        surf = self.surf.copy()

        # dynamic
        pos = (10+int((self.val-self.mini)/(self.maxi-self.mini)*80), 33)
        if self.i == 2 or self.i == 1 or self.i == 3:
            pos = (math.trunc(pos[0]), 33)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)  # move of button box to correct screen position

        # screen
        screen.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """
        The dynamic part; reacts to movement of the slider button.
        """

        val = (pygame.mouse.get_pos()[0] - self.xpos - 10) / 80 * (self.maxi - self.mini) + self.mini
        if self.i == 2 or self.i == 1 or self.i == 3:
            self.val = math.trunc(val)
        else:
            self.val = val
        if self.val < self.mini:
            self.val = self.mini
            
        if self.val > self.maxi:
            self.val = self.maxi
        song_features[self.i] = self.val

screen = pygame.display.set_mode((X, Y))
clock = pygame.time.Clock()
COLORS = [MAGENTA, RED, YELLOW, GREEN, CYAN, BLUE]

slides = []
y_pos = 0

song_features = []

for i in range(len(features)):
    song_features.append(maxis[i]/2)
    slides.append(Slider(i, maxis[i]/2, maxis[i], minis[i], y_pos))
    y_pos += 50

outcome_button = Button("Change outcome", (137, 379), change_outcome)
training_button = Button("Train data", (85, 437), initialize_training)
buttons = [outcome_button, training_button]
num = 0

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for s in slides:
                if s.button_rect.collidepoint(pos):
                    s.hit = True
            mousebuttondown()
        elif event.type == pygame.MOUSEBUTTONUP:
            for s in slides:
                s.hit = False

    # Move slides
    for s in slides:
        if s.hit:
            changed_numbers = True
            s.move()
    
    # Update screen
    screen.fill(BLACK)
    num += 2
    show_text(song_features)
    
    for s in slides:
        s.draw()
    for b in buttons:
        b.draw()

    pygame.display.flip()
    clock.tick()