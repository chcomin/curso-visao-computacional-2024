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
9. Busca de imagens por linguagem natural;
10. Geração de imagens artificiais
11. Detecção e casamento de pontos salientes

## Bibliografia

* Understanding Deep Learning, Simon J. D. Prince
https://udlbook.github.io/udlbook. Principal referência do curso. Possui excelentes figuras para explicar os conceitos;

* Dive into Deep Learning, Aston Zhang A. et al. https://d2l.ai. Foca na parte prática de redes neurais, com implementação de todos os conceitos discutidos;

* Deep Learning: Foundations and concepts, Chris Bishop e Hugh Bishop. https://www.bishopbook.com. Referência para a parte teórica sobre redes neurais;

* Computer Vision: Algorithms and Applications, Richard Szeliski. https://szeliski.org/Book. De longe a principal referência sobre todas as subáreas de Visão Computacional.

### Outras referências interessantes

* Curso do Justin Johnson da Universidade de Michigan: 
https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html

* Curso de Stanford: http://cs231n.stanford.edu/schedule.html

* Dicas importantes sobre treinamento de redes neurais - Andrej Karpathy:  https://karpathy.github.io/2019/04/25/recipe/

## Configuração do ambiente

`conda install -c pytorch -c nvidia -c conda-forge python pytorch torchvision torchaudio pytorch-cuda=12.1 matplotlib notebook numpy scipy transformers diffusers accelerate python-graphviz ipympl scikit-learn timm plotly`

As bibliotecas python-graphviz, ipympl, scikit-learn, timm e plotly são opcionais e serão usadas apenas uma única vez em exemplos específicos. Você pode usar os arquivos requirements.txt ou requirements.yml para criar o ambiente.


## GPUs online grátis

A grande maioria dos códigos da disciplina foi planejada para não necessitar de GPUs. Mas se for necessário executar algo em GPU, segue abaixo uma lista de serviços em nuvem que possibilitam execução na GPU de graça. As informações foram coletadas em 05/03/2024.

https://colab.research.google.com/
* GPU: Tesla T4 (16GB)
* CPU: Intel Xeon 2.30GHz, 1 core 2 threads
* CPU RAM: 12 GB
* Disco: 78 GB (não persistente)
* 12 horas contínuas de execução

<br/>

https://studiolab.sagemaker.aws/
* GPU: Tesla T4 (16GB)
* CPU: Intel Xeon Platinum 8259CL CPU 2.50GHz, 2 cores 2 threads
* CPU RAM: 16 GB
* Disco: 15 GB (persistente)
* 4 horas por dias de execução

<br/>

https://lightning.ai/
* GPU: Tesla T4 (16GB)
* CPU: Intel Xeon Platinum 8259CL CPU 2.50GHz, 4 cores 2 threads
* RAM: 16 GB
* Disco: 416 GB! (persistente)
* 22 horas de execução por mês

<br/>

https://www.kaggle.com/code/
* GPU: 2x Tesla T4 ou Tesla P100   (pode escolher, ambas possuem 16GB)
* CPU: Intel Xeon CPU 2.00GHz, 2 cores, 2 threads
* RAM: 32 GB
* Disco: 73 GB (persistente)
* 12 horas de execução contínua

<br/>

A imagem abaixo foi gerada utilizando um dos notebooks da disciplina!

![](data/leao.png)