import retro
import pickle
import cv2
import numpy as np
import neat

class Worker:
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0] / 8)
        iny = int(ob.shape[1] / 8)

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        done = False

        while not done:
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            actions = net.activate(imgarray)

            ob, rew, done, info = self.env.step(actions)

            # Renderização do ambiente
            self.env.render()
            
            # Exibição usando OpenCV
            ob_rgb = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            #cv2.imshow('Super Mario World', ob_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if done:
                break

        self.env.close()
        cv2.destroyAllWindows()

def run_winner():
    config_path = 'config-feedforward'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    with open('winner.pkl', 'rb') as input_file:
        winner = pickle.load(input_file)

    worker = Worker(winner, config)
    worker.work()

if __name__ == '__main__':
    run_winner()
