Como instalar:
- Primeiro, crie um env no conda (requer anaconda): conda create -n myenv python=3.7.9;
- E depois ative-o: conda activate myenv;
- Após isso, instale o gym retro - pip3 install gym-retro;
- Reinstalar gym na versão correta:
    - pip uninstall gym;
    - pip install gym==0.21.0.
- Por fim, instalar importlib na versão correta: pip install importlib-metadata==4.13.0;
- É necessário instalar o NEAT-Python também e suas dependencias:
    - pip install cloudpickle;
    - pip install graphviz;
    - pip install matplotlib;
    - pip install opencv-python. #apesar que acho que esse já estava instalado
    

Como rodar:
- Existem 2 modelos: o "SINGLE" irá rodar um jogo por sequencia, "PARALLEL" irá rodar 10 jogos em sequencia;
- Para treinar o SINGLE, rode o "random_agent-mario". Quando terminar a execução, um arquivo "winner.pck" será criado;
- Rode o "playback.py" para reproduzir a IA salva no "winner.pck";
- Para treinar o PARALLEL, rode o "random_agent-mario-parallel". Quando terminar a execução, um arquivo "winner.pck" será criado;
- Rode o "playback_2.py" para reproduzir a IA salva no "winner.pck".

Observações:
- A IA utiliza a tela como input, portanto alterar o tamanho da tela alterará o resultado. Também rodar o "playback_2.py" ao invés do "playback.py" e vice-versa também fará com que a IA tenha resultado diferente do esperado.
- Por algum motivo, não tivemos acesso a dados suficientes quando importamos a ROM, só a pontuação, vida e moeda. Portanto a gente substituiu o data.json com de outro repositorio que contem mais dados que possamos usar, como a posicao do Mario e se ele venceu ou não;
- A ROM foi obtida no Vimm' Lair usando Wayback Machine. O código foi feito seguindo um tutorial no YouTube do Lucas Thompson - https://www.youtube.com/watch?v=CFa6NhLgeL0.

