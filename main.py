import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
from log import Logger

#inicializando logger
sys.stdout = Logger("log.txt")

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
print("-"*78)
print 'Tabela de amostras:'
dspd = pd.DataFrame(dataset, index=["1","2","3","4","5","6","7","8"])
dspd.index.names = ['Amostra']
dspd.columns = ["Xi","Yi","Classe"]
#print tabela de pontos
print dspd

#pesos iniciais
w0 = 2
w1 = 1
w2 = -1
pesos = [w0, w1, w2]

#lista de coordenadas separadas por classe
x1 = [] #lista de coordenadas x da classe 1
y1 = [] #lista de coordenadas y da classe 1
x2 = [] #lista de coordenadas x da classe -1
y2 = [] #lista de coordenadas y da classe -1

#organizacao das listas
for j in range(0,len(dataset)):

    if dataset[j][2] == 1:
        x1.append(dataset[j][0])
        y1.append(dataset[j][1])
    else:
        x2.append(dataset[j][0])
        y2.append(dataset[j][1])

#plotando o grafico 1
ax1 = plt.subplot(1, 1, 1)
ax1.scatter(x2,y2,marker='o',color=['black'],edgecolors='black', label='Classe -1')
ax1.scatter(x1,y1,marker='o',color=['white'],edgecolors='black', label='Classe 1')
#ajustes
ax1.set_xlim(0, 1)
ax1.set_xlabel('i1')
plt.xticks(np.arange(0, 1.1, step=0.1))
ax1.set_ylim(0, 1.02)
ax1.set_ylabel('i2')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.legend()
plt.show()

#plotando o grafico 2
ax2 = plt.subplot(1, 1, 1)
ax2.scatter(x2,y2,marker='o',color=['black'],edgecolors='black', label='Classe -1')
ax2.scatter(x1,y1,marker='o',color=['white'],edgecolors='black', label='Classe 1')
ax2.axvline(0, color = "black")
ax2.axhline(0, color = "black")
#plotando a Linha inicial
#equacao geral da reta: a*x + b*y + c = 0
#equacao rearranjada: y = (((-a)/b)*(x)) + ((-c)/b)
xli = np.linspace(-999,999,1000)
yli = (((-w1)/w2)*(xli)) + ((-w0)/w2)
plt.plot(xli,yli,label='Linha inicial')
#ajustes
ax2.set_xlim(-2.5, 2.5)
ax2.set_xlabel('i1')
ax2.set_ylim(-2.5, 2.5)
ax2.set_ylabel('i2')
ax2.grid(True)
plt.legend()
plt.show()

#### predicao sem ajuste de pesos ####

#classificacoes incorretas
inc = 0

#contador de amostras
cp = 0

#vetor de amostras erradas
amostras_erradas = []

print("-"*78)
print "Pesos iniciais: "
print "w0 =",w0,"  w1 =",w1,"  w2 =",w2
print("-"*78)
print "PREVISOES SEM ATUALIZACAO DOS PESOS:"
print("-"*78)
for linhas_tab in dataset:
    predicao = pred(linhas_tab, pesos)
    if linhas_tab[-1] != predicao:
        inc += 1
        cp += 1
        amostras_erradas.append(cp)
    else:
        cp += 1
    print("Amostra %d:"% (cp))
    print("Esperado: %d, Previsto: %d" % (linhas_tab[-1], predicao))

print("\nNumero de classificacoes incorretas: %d" % (inc))
print "Amostras classificadas incorretas: ", str(amostras_erradas)[1:-1]

#### atualizacao de pesos ####

print("-"*78)
print("CALCULO DA ATUALIZACAO DE PESOS E VARIACAO DA TAXA DE APRENDIZAGEM")
print("-"*78)

#atualizacao de pesos e taxa de aprendizagem = 1
atualizar_pesos(x1,x2,y1,y2,pesos,dataset,taxa_aprendizagem=1)
print("-"*78)
#atualizacao de pesos e taxa de aprendizagem = 0.1
atualizar_pesos(x1,x2,y1,y2,pesos,dataset,taxa_aprendizagem=0.1)
print("-"*78)
#atualizacao de pesos e taxa de aprendizagem = 0.01
atualizar_pesos(x1,x2,y1,y2,pesos,dataset,taxa_aprendizagem=0.01)
print("-"*78)

#atualizacao de pesos ate o final com taxa de aprendizagem = 1
pesos, updates = classificacao_perfeita(pesos,dataset,taxa_aprendizagem=1)

print"Numero de atualizacoes de pesos ate a classificacao perfeita: ", updates
print "Pesos da classificacao perfeita: "
print "w0 =",pesos[0],"  w1 =",pesos[1],"  w2 =",pesos[2]
print("-"*78)

#plotando o grafico 4
ax4 = plt.subplot(1, 1, 1)
ax4.scatter(x2, y2, marker='o', color=['black'], edgecolors='black', label='Classe -1')
ax4.scatter(x1, y1, marker='o', color=['white'], edgecolors='black', label='Classe 1')
ax4.axvline(0, color="black")
ax4.axhline(0, color="black")
# plotando a Linha inicial
# equacao geral da reta: a*x + b*y + c = 0
# equacao rearranjada: y = (((-a)/b)*(x)) + ((-c)/b)
xli = np.linspace(-999, 999, 1000)
yli = (((-w1) / w2) * (xli)) + ((-w0) / w2)
plt.plot(xli, yli, label='Linha inicial')
# plotando a Linha final
yli = (((-(pesos[1])) / (pesos[2])) * (xli)) + ((-(pesos[0])) / (pesos[2]))
plt.plot(xli, yli, label='Linha final')
# ajustes
ax4.set_xlim(-2.5, 2.5)
ax4.set_xlabel('i1')
ax4.set_ylim(-2.5, 2.5)
ax4.set_ylabel('i2')
ax4.grid(True)
plt.legend()
plt.show()
