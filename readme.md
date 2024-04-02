# Visão Computacional 2024

Curso de Visão Computacional ministrado no Departamento de Computação UFSCar. 

O curso é dividio em módulos (M01, M02, ...). Cada módulo possui notebooks Jupyter numerados apresentando conceitos sobre Visão Computacional. Também são disponibilizadas [notas de aula](<Notas de Aula.pdf>) sobre os conceitos teóricos.

## Principais tópicos abordados

1. Visão geral sobre modelagem probabilística e cálculo diferencial;
2. Regressão linear e logística;
3. Diferenciação automática
4. Redes neurais convolucionais;
5. Estabilização e regularização de redes modernas;
6. Classificação de imagens naturais;
7. Segmentação de imagens naturais;
8. Autoencoders para remoção de ruído;
9. Mecanismos de atenção (transformers);
10. Busca de imagens por linguagem natural;
11. Geração de imagens artificiais
12. Detecção e casamento de pontos salientes

## Bibliografia

* Understanding Deep Learning, Simon J. D. Prince
https://udlbook.github.io/udlbook. Principal referência do curso. Possui excelentes figuras para explicar os conceitos;

* Dive into Deep Learning, Aston Zhang A. et al. https://d2l.ai. Foca na parte prática de redes neurais, com implementação de todos os conceitos discutidos;

* Deep Learning: Foundations and concepts, Chris Bishop e Hugh Bishop. https://www.bishopbook.com. Referência para a parte teórica sobre redes neurais;

* Computer Vision: Algorithms and Applications, Richard Szeliski. https://szeliski.org/Book. De longe a principal referência sobre todas as subáreas de Visão Computacional.

### Outras referências interessantes

* Curso do Justin Johnson da Universidade de Michigan: https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html

* Curso de Stanford: http://cs231n.stanford.edu/schedule.html

* Dicas importantes sobre treinamento de redes neurais - Andrej Karpathy: https://karpathy.github.io/2019/04/25/recipe/

## Materiais sobre Python e processamento de imagens

Revisão sobre conceitos Python e numpy necessários para redes neurais: https://cs231n.github.io/python-numpy-tutorial/

Livro referência para processamento digital de imagens: [Processamento Digital de Imagens, R. Gonzalez e R. Woods](https://www.amazon.com.br/Processamento-digital-imagens-Rafael-Gonzalez/dp/8576054019)

Playlist de disicplina PDI ministrada durante a pandemia em 2021: https://youtube.com/playlist?list=PLV6nBLHbx0riEUDgJrAogzpAt3qQquKVR

Slides e códigos de disciplina PDI ministrada em 2022: [https://www.dropbox.com/scl/fo/3hahb07p2ncfzhus0tpq9](https://www.dropbox.com/scl/fo/3hahb07p2ncfzhus0tpq9/AM_1eZxgn74NWp8OAFjiA6A?rlkey=xvyhhzjht6dzskmo6q5m22j8p&dl=0)

## Configuração do ambiente

`conda install -c pytorch -c nvidia -c conda-forge python pytorch torchvision torchaudio pytorch-cuda=12.1 matplotlib notebook numpy scipy transformers diffusers accelerate python-graphviz ipympl scikit-learn timm plotly`

As bibliotecas python-graphviz, ipympl, scikit-learn, timm e plotly são opcionais e serão usadas apenas uma única vez em exemplos específicos. Você pode usar os arquivos requirements.txt ou requirements.yml para criar o ambiente. Note que o pacote `-c nvidia pytorch-cuda=12.1` só deve ser instalado se o ambiente tiver acesso a uma GPU.


## GPUs online grátis

A grande maioria dos códigos da disciplina foi planejada para não necessitar de GPUs. Os notebooks que precisam de GPU estão marcados com (GPU) no nome. Os códigos foram testados em um processador Intel Core i5-3230M 2.60GHz com 2 cores e 4 threads e 8 GB de RAM, ou seja, um computador básico. Para os códigos que precisam de GPU, segue abaixo uma lista de serviços em nuvem que possibilitam execução na GPU de graça. As informações foram coletadas em 05/03/2024.

https://colab.research.google.com/
* GPU: Tesla T4 (16GB)
* CPU: Intel Xeon 2.30GHz, 1 core 2 threads
* CPU RAM: 12 GB
* Disco: 78 GB (não persistente)
* 12 horas contínuas de execução

<br/>

https://studiolab.sagemaker.aws/
* GPU: Tesla T4 (16GB)
* CPU: Intel Xeon Platinum 8259CL CPU 2.50GHz, 2 cores 4 threads
* CPU RAM: 16 GB
* Disco: 15 GB (persistente)
* 4 horas por dias de execução

<br/>

https://lightning.ai/
* GPU: Tesla T4 (16GB)
* CPU: Intel Xeon Platinum 8259CL CPU 2.50GHz, 4 cores 8 threads
* RAM: 16 GB
* Disco: 416 GB! (persistente)
* 22 horas de execução por mês

<br/>

https://www.kaggle.com/code/
* GPU: 2x Tesla T4 ou 1x Tesla P100 (pode escolher, ambas possuem 16GB)
* CPU: Intel Xeon CPU 2.00GHz, 2 cores 4 threads
* RAM: 32 GB
* Disco: 73 GB (persistente)
* 12 horas de execução contínua

<br/>

A imagem abaixo foi gerada utilizando um dos notebooks da disciplina!

![](data/leao.png)