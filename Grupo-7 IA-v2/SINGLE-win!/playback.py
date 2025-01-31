import retro
import numpy as np
import cv2 
import neat
import pickle

#mesma coisa do random_agent-mario, exceto que reproduz um genoma salvo como "winner.pck" que esteja no mesmo diretorio
env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
imgarray = []
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)


ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_max = 0

done = False


while not done:
    env.render() 
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))

    imgarray = np.ndarray.flatten(ob)

            
    nnOutput = net.activate(imgarray) 
    ob, rew, done, info = env.step(nnOutput) 

    pontos = info['score']
    fim = info['endOfLevel']
    vida = info['lives']
    xPos = info['x']
  
    if xPos > xpos_max:
        counter = 0
        fitness_current = xPos
        xpos_max = xPos           
            
    if fim == 1:
        fitness_current += 40*pontos
            
    if fitness_current > current_max_fitness:
        counter = 0
        current_max_fitness = fitness_current
    else:
        counter += 1
            
    if done or counter >= 250 or vida < 4:
        done = True
                


