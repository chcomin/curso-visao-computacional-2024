# Visão Computacional 2024

Curso de Visão Computacional ministrado no Departamento de Computação UFSCar. **Ainda em construção**!!

O curso é dividio em módulos (M01, M02, ...). Cada módulo possui notebooks Jupyter numerados apresentando conceitos sobre Visão Computacional.

### Bibliografia

* Understanding Deep Learning, Simon J. D. Prince
https://udlbook.github.io/udlbook.

* Dive into Deep Learning, Aston Zhang A. et al. https://d2l.ai.

* Deep Learning: Foundations and concepts, Chris Bishop e Hugh Bishop. https://www.bishopbook.com.

* Computer Vision: Algorithms and Applications, Richard Szeliski. https://szeliski.org/Book.


### Outras referências interessantes

* Curso do Justin Johnson da Universidade de Michigan: 
https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html

* Curso de Stanford: http://cs231n.stanford.edu/schedule.html

* Dicas importantes sobre treinamento de redes neurais - Andrej Karpathy:  https://karpathy.github.io/2019/04/25/recipe/

### Configuração do ambiente

`conda install -c pytorch -c nvidia -c conda-forge python pytorch torchvision torchaudio pytorch-cuda=12.1 matplotlib notebook numpy scikit-learn scipy ipympl transformers diffusers accelerate python-graphviz`

### GPUs online grátis

Lista de serviços em nuvem que possibilitam executar códigos em GPU de graça:

https://colab.research.google.com/
* GPU: Tesla T4
* CPU: Intel Xeon 2.30GHz, 1 core 2 threads
* RAM: 12 GB
* Disco: 78 GB (não persistente)
* 12 horas contínuas de execução

<br/>

https://studiolab.sagemaker.aws/
* GPU: Tesla T4
* CPU: Intel Xeon Platinum 8259CL CPU 2.50GHz, 2 cores 2 threads
* RAM: 16 GB
* Disco: 15 GB (persistente)
* 4 horas por dias de execução

<br/>

https://lightning.ai/
* GPU: Tesla T4
* CPU: Intel Xeon Platinum 8259CL CPU 2.50GHz, 4 cores 2 threads
* RAM: 16 GB
* Disco: 416 GB! (persistente)
* 22 horas de execução por semana

<br/>

https://www.kaggle.com/code/
* GPU: 2x Tesla T4 ou Tesla P100   (pode escolher)
* CPU: Intel Xeon CPU 2.00GHz, 2 cores, 2 threads
* RAM: 32 GB
* Disco: 73 GB (persistente)
* 12 horas de execução contínua

<br/>

A imagem abaixo foi gerada utilizando um dos notebooks da disciplina!

![](data/leao.png)