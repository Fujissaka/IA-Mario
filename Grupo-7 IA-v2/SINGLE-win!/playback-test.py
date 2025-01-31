import pickle
import neat

# Carregar o genoma salvo

try:
    # Tente carregar o arquivo
    with open('winner.pkl', 'rb') as input_file:
        winner = pickle.load(input_file)
        print("Genoma carregado com sucesso!")
        print(winner)
except EOFError:
    print("Erro: O arquivo está vazio ou corrompido.")
except FileNotFoundError:
    print("Erro: O arquivo 'winner.pkl' não foi encontrado.")


# Exibir informações sobre o genoma vencedor
print("Genoma Vencedor:")

# Configuração para reconstruir a rede do genoma
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward'
)
