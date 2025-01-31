import retro
import pickle
import cv2
import numpy as np
import neat

#aprendizado em paralelo, utiliza uma classe chamada Worker para rodar cada instancia em uma thread diferente
class Worker(object):
    def __init__ (self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):#eh o que cada thread ira rodar
        self.env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1) #jogo do mario
        self.env.reset()

        ob, _ , _ , _ = self.env.step(self.env.action_space.sample())#tela

        inx = int(ob.shape[0]/8)  #dividindo por 8 para caber na rede neural
        iny = int(ob.shape[1]/8)

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config) #cria uma IA baseada no genoma e nas configuracoes
        done = False;
        
        fitness = 0
        xPos = 0
        xPos_max = 0
        fim = 0
        pontos = 0
        counter = 0
        imgarray = []
        #igual o anterior
        while not done:
            ob = cv2.resize(ob, (inx, iny))#ajusta a tela para usar como input
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))

            imgarray = np.ndarray.flatten(ob) #transforma as informaçoes num vetor unidimensional
            
            #define as acoes da ia dado o input
            actions = net.activate(imgarray) 
            ob, rew, done, info = self.env.step(actions)

            #define o quao longe o mario chegou sem morrer
            xPos = info['x']
            vida = info['lives']
            #esses 2 eh para determinar quando terminou e o tempo que levou.
            fim = info['endOfLevel']
            pontos = info['score']
            
            #quando mais para a direita o mario for, aumenta o valor do fitness
            #se o mario ficar parado, aumenta um tempo que, caso seja maior que 250, reinicia desde o comeco
            if xPos > xPos_max:
                counter = 0
                fitness = xPos
                xPos_max = xPos
            else:
                counter += 1 #vai ficar contando, para caso o mario esteja parado fazendo nada

            if fim == 1: #caso o mario venca, verifica se tem pontos suficientes: caso o valor ultrapasse 100000 ele "eh declarado vencedor"
                fitness = 50*pontos
                done = True

            if done or counter >= 250 or vida < 4: #caso o mario morra ou fique muito tempo sem andar para frente, reinicia
                done = True
                    

        return fitness #retorna o resultado

def eval_genomes(genome, config): #utiliza a funcao acima para definir o comportamento de cada thread
    worky = Worker(genome, config)
    return worky.work()

    
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward') #as configuracoes usadas no NEAT

p = neat.Population(config) #gera uma populacao baseada nas configs

pe = neat.ParallelEvaluator(10, eval_genomes) #coloquei como 10 pois meu PC tem 12 threads

#exibe os dados de cada geracao
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

#salva o genoma do "vencedor" numa variavel
winner = p.run(pe.evaluate)

#salva o genoma do vencedor num arquivo.pck
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
