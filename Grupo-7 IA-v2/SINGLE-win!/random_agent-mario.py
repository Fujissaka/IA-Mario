import retro
import pickle
import cv2
import numpy as np
import neat

imgarray = []
env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset() #tela
        ac = env.action_space.sample() #acao aleatoria

        inx, iny, inc = env.observation_space.shape #resolucao da tela - x, y e cor
        inx = int(inx/8)  #dividindo por 8 para caber na rede neural
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        pontos = 0
        pontos_max = 0
        vida = 0
        vida_max = 0

        xPos = 0
        xpos_max = 0

        done = False
        
        while not done: #enquanto o jogo roda
            #env.render() #mostra tela
            frame += 1#frames do jogo
            
            #ajusta a tela
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))

            imgarray = np.ndarray.flatten(ob) #transforma o tamanho da tela num vetor
            
            nnOutput = net.activate(imgarray) #faz o output
            #print(nnOutput)
            ob, rew, done, info = env.step(nnOutput) #utiliza o imput do NEAT

            #usado para determinar se o Mario venceu o nivel e quando tempo aproximadamente ele levou
            pontos = info['score']
            fim = info['endOfLevel']
            #usado para determinar a distancia do Mario no mapa sem morrer
            vida = info['lives']
            xPos = info['x']   
            if xPos > xpos_max:
                counter = 0
                fitness_current = xPos
                xpos_max = xPos           
            
            #xPos >= 5020:
            if fim == 1:
                fitness_current += 50*pontos #A pontuacao eh baseada na distancia percorrida
            
            if fitness_current > current_max_fitness:
                counter = 0 #reinicia se mario conseguir pontuacao
                current_max_fitness = fitness_current
            else:
                counter += 1 #vai ficar contando, para caso o mario esteja parado fazendo nada
            
            if done or counter >= 250 or vida < 4:
                done = True
                print(genome_id, fitness_current)
                
            
            genome.fitness = fitness_current #atualiza o genoma.

#Obtem a configuracao do NEAT e utiliza para gerar uma populacao e treinar a IA.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')
p = neat.Population(config)

#informa os dados de cada geracao no terminal
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes) #salva o genoma do vencedor numa variavel

with open('winner.pkl', 'wb') as output: #salva o genoma do vencedor num arquivo .pck
    pickle.dump(winner, output, 1)

