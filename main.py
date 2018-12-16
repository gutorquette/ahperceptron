import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
from log import Logger

#inicializando logger
sys.stdout = Logger("log.txt")

#pesos iniciais
w0 = 2
w1 = 1
w2 = -1
pesos = [w0, w1, w2]

#dados
dataset = [[0.08,0.72,1.00],
           [0.10,1.00,-1.00],
           [0.26,0.56,1.00],
           [0.32,0.97,-1.00],
           [0.44,0.14,1.00],
           [0.60,0.30,1.00],
           [0.72,0.64,-1.00],
           [0.94,0.42,-1.00]]

#tabelas de pontos [X,Y,Classe]
tabela(dataset)

#plot amostras da figura 1
plotfig1(dataset)

#predicao sem ajuste de pesos#
sem_atualizar_pesos(pesos,dataset)

#plot linha inicial
plotfig2(pesos,dataset)

#### ATUALIZACAO DOS PESOS ####

print("-"*78)
print("CALCULO DA ATUALIZACAO DE PESOS E VARIACAO DA TAXA DE APRENDIZAGEM")
print("-"*78)

#atualizacao de pesos e variacao da taxa de aprendizagem (1, 0.1 e 0.01)
taxa_aprendizagem = 1.0
for _ in range(1,4):
    atualizar_pesos(pesos,dataset,taxa_aprendizagem)
    print("-"*78)
    taxa_aprendizagem /= 10.0

print("CALCULO DA ATUALIZACAO DE PESOS ATE A CLASSIFICACAO PERFEITA")
print("-"*78)
print("-"*78)

#atualizacao de pesos ate o final com taxa de aprendizagem = 1
pesos, pesos_finais, updates = classificacao_perfeita(pesos,dataset,taxa_aprendizagem=1)

print"Numero de atualizacoes de pesos ate a classificacao perfeita: ", updates
print "Pesos da classificacao perfeita: "
print "w0 =",pesos_finais[0],"  w1 =",pesos_finais[1],"  w2 =",pesos_finais[2]
print("-"*78)

#plot linha final
plotfigfinal(pesos,pesos_finais,dataset)
