
#AI playing flappy bird
#using pygame and neat-python

#neat and neat-python are diff modules

"""
NEAT (NeuroEvolution of Augmenting Topologies)

neat is used to evolve a neural network

for AI

inputs -> bird y , pipe top 
output-> jump or not
population size -> no of birds used by the NN to train 
after training these 100 (generation 0) another 100(generation 1) will be trained which will be better tha gen 0 and so on

fitness funcn -> how we grow and birds get better , how we select the best bird

Max generation -> here 30


config file -> parameters etc for the algo
in this txt file

fitness_criterion -> min max or mean ,  to get rid of the best/worst birds
fitness_threshold -> fitness level i.e score here 
pop_size -> population size
reset_on_extinction -> specied depending on the network of the bird


[DefaultGenome]
genome -> bird here(population members are alled genome)
"""


import pygame
import time
import os
import neat
import random


GEN = 0 

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800


# 3 bird images to look like bird is flapping wings
# transform.scale2x to double the size of the image

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans" , 50)

#bird class to represent bird is moving
class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25 # how much bird tilts when it goes up or down
    ROT_VEL = 20 # 
    ANIMATION_TIME = 5 # how fast or slow bird flap its wings

    def __init__(self,x,y):
        #starting position of the bird
        self.x = x 
        self.y = y
        self.tilt = 0 #how much img is tilted
        self.tick_count = 0 
        self.vel = 0 # because bird is not moving right now
        self.height = self.y 
        self.image_count = 0 # so we know which img we are on 
        self.img = self.IMGS[0] #bird images


    def jump(self):
        # upward -ve in y 
        # down +ve in y
        self.vel = -10.5
        self.tick_count = 0 #keeps tack of when we last jumped
        self.height = self.y # keep track of where bird jumped from

    
    def move(self):
        self.tick_count += 1

        #displacement i.e how many pixel we move up or down
        d = self.vel * self.tick_count + 1.5*self.tick_count**2

        #so it deos not go too up or down
        if d >=16:
            d = 16

        if d < 0:
            d-= 2 #to jump higher or lower

        self.y = self.y + d

        #tilting
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self,win):
        self.image_count += 1

        #flapping the wings
        if self.image_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.image_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.image_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.image_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.image_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.image_count = 0

        #if bird is going down it should no flap wings so
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.image_count = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img , self.tilt)
        new_rect = rotated_image.get_rect(center =self.img.get_rect(topleft = (self.x , self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mast(self):
        return pygame.mask.from_surface(self.img)
        #mask -> if the pixels of both image touches each other


class Pipe:
    GAP = 200
    VEL = 5 #speeed of pipe 

    def __init__(self,x):
        #random heights
        self.x = x
        self.height= 0

        self.top = 0
        self.bottom = 0 
        #flip pipe to make the upper pipe
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False , True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False #if bired passed the pipe
        self.set_height() 

    def set_height(self):
        #defins top bottom and height of pipe
        self.height = random.randrange(50,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        #moving pipes
        self.x -= self.VEL

    def draw(self,win):
        win.blit(self.PIPE_TOP, (self.x , self.top))
        win.blit(self.PIPE_BOTTOM, (self.x , self.bottom))
    
    #mask if the pixels of both image touches each other
    #mask returns the 2d list of pixels of the image

    def collide(self,bird):
        bird_mask = bird.get_mast()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        #offset -> how far away the masks are
        top_offset = (self.x - bird.x , self.top - round(bird.y))
        bottom_offset = (self.x - bird.x , self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask , bottom_offset) #returns none if they dont collide else return point of collision
        t_point = bird_mask.overlap(top_mask , top_offset)

        if t_point or b_point:
            return True
        
        return False


class Base:
    #background , we need class for this beacuse we have to move it
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG


    def __init__(self, y):
        self.y = y 
        self.x1 = 0
        self.x2 = self.WIDTH
    
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self,win):
        win.blit(self.IMG, (self.x1 , self.y))
        win.blit(self.IMG, (self.x2 , self.y))



def draw_window(win , birds, pipes , base,score,gen):
    win.blit(BG_IMG , (0,0))
    for pipe in pipes:
        pipe.draw(win)
    
    #score
    text = STAT_FONT.render("Score : " +str(score), 1,(255,255,255))
    win.blit(text,(WIN_WIDTH - 10 - text.get_width(), 10))

    #generation
    text = STAT_FONT.render("Gen : " +str(gen), 1,(255,255,255))
    win.blit(text,(10, 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)
    pygame.display.update()

def main(genomes , config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    #all birds play at same time
    birds = []

    #genomes is a tuple 
    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0
        ge.append(g)


    base = Base(730) #bottom of the screen
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock() #to control how fast while loop runs
    score = 0
    run = True
    while run:
        clock.tick(30) #atmost 30 ticks every seconds
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        
        pip_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pip_ind = 1
        else:
            run = False
            break
        
        for x, bird in enumerate(birds):
            bird.move()
            #add fitness
            #this loop runs 30 times a sec
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y , abs(bird.y - pipes[pip_ind].height), abs(bird.y - pipes[pip_ind].bottom)))
            #output is a list
            
            if output[0] > 0.5:
                bird.jump()



        add_pipe = False
        rem = [] #list of removed pipes
        for pipe in pipes:
            for x , bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1 # -1 from fitness score if bird hits the pipe
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

           
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed =True
                    add_pipe =True

            if pipe.x + pipe.PIPE_TOP.get_width()<0 :
                rem.append(pipe)
                #pipe is outside screen

            pipe.move()

        if add_pipe:
            score +=1
            #add 5 to bird if the pass the pipe
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for x,bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                #hit the floor
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(win,birds,pipes,base,score,GEN)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome , neat.DefaultReproduction , neat.DefaultSpeciesSet , neat.DefaultStagnation,config_path)

    p = neat.Population(config)
    #details , stats of the birds 
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50) #50 generations and the function used to calculate the fitness


#to find the path of the txt file , needed for neat
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir , "config-feedforward.txt")
    run(config_path)


        





    
