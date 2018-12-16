import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#gera a tabela formatada
def tabela(dataset):

    print("-" * 78)
    print 'Tabela de amostras:'
    dspd = pd.DataFrame(dataset, index=["1", "2", "3", "4", "5", "6", "7", "8"])
    dspd.index.names = ['Amostra']
    dspd.columns = ["Xi", "Yi", "Classe"]
    print dspd

#calculo da funcao de ativacao
def pred(linhas_tab, pesos):

    ativacao = pesos[0]

    for i in range(len(linhas_tab)-1):

        ativacao += pesos[i + 1] * linhas_tab[i]

    return 1.0 if ativacao > 0.0 else -1.0

#separacao das classes em vetores
def lista_de_classes(dataset):

    #lista de coordenadas separadas por classe
    x1 = []  #lista de coordenadas x da classe 1
    y1 = []  #lista de coordenadas y da classe 1
    x2 = []  #lista de coordenadas x da classe -1
    y2 = []  #lista de coordenadas y da classe -1

    #organizacao das listas
    for j in range(0, len(dataset)):

        if dataset[j][2] == 1:
            x1.append(dataset[j][0])
            y1.append(dataset[j][1])
        else:
            x2.append(dataset[j][0])
            y2.append(dataset[j][1])

    return x1,x2,y1,y2

#previsoes sem atualizacao dos pesos
def sem_atualizar_pesos(pesos,dataset):

    #classificacoes incorretas
    inc = 0
    # contador de amostras
    cp = 0
    #vetor de amostras erradas
    amostras_erradas = []

    print("-" * 78)
    print "Pesos iniciais: "
    print "w0 =", pesos[0], "  w1 =", pesos[1], "  w2 =", pesos[2]
    print("-" * 78)
    print "PREVISOES SEM ATUALIZACAO DOS PESOS:"
    print("-" * 78)

    #calculo das previsoes
    for linhas_tab in dataset:
        #calculo da funcao de ativacao
        predicao = pred(linhas_tab, pesos)
        #contagem de classificacoes incorretas e numero da amostra
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

#atualizacao dos pesos
def atualizar_pesos(pesos,dataset,taxa_aprendizagem):

    #pesos iniciais
    w0 = pesos[0]
    w1 = pesos[1]
    w2 = pesos[2]

    #separacao das classes em vetores
    x1,x2,y1,y2 = lista_de_classes(dataset)

    #contador de linhas
    cont = 1

    #plotando o grafico com as 6 linhas
    ax = plt.subplot(1, 1, 1)
    ax.scatter(x2, y2, marker='o', color=['black'], edgecolors='black', label='Classe -1')
    ax.scatter(x1, y1, marker='o', color=['white'], edgecolors='black', label='Classe 1')
    ax.axvline(0, color="black")
    ax.axhline(0, color="black")
    #plotando a Linha inicial
    #equacao geral da reta: a*x + b*y + c = 0
    #equacao rearranjada: y = (((-a)/b)*(x)) + ((-c)/b)
    xli = np.linspace(-999, 999, 1000)
    yli = (((-w1) / w2) * (xli)) + ((-w0) / w2)
    plt.plot(xli, yli, label='Linha inicial (1)')

    while True:

        #calculo da atualizacao dos pesos
        for linhas_tab in dataset:

            #calculo da funcao de ativacao
            predicao = pred(linhas_tab, pesos)

            #calculo do erro
            erro = linhas_tab[-1] - predicao

            #atualizacao do calculo dos pesos para w0
            pesos[0] = pesos[0] + taxa_aprendizagem * erro

            #atualizacao do calculo dos pesos para os demais pesos
            for k in range(len(linhas_tab)-1):
                pesos[k + 1] = pesos[k + 1] + taxa_aprendizagem * erro * linhas_tab[k]

            #plotando as linhas e pesos apenas das amostras classificadas erroneamente e que foram atualizadas
            if erro != 0:
                cont += 1
                yli = (((-pesos[1]) / pesos[2]) * (xli)) + ((-pesos[0]) / pesos[2])
                #incrementar label
                labelinc = 'Linha '+ str(cont)
                plt.plot(xli, yli, label=labelinc)
                # ajustes
                ax.set_xlim(-2.5, 2.5)
                ax.set_xlabel('i1')
                ax.set_ylim(-2.5, 2.5)
                ax.set_ylabel('i2')
                ax.grid(True)
                plt.title("Taxa de aprendizagem: " + str(taxa_aprendizagem))
                plt.legend()
                print 'Taxa de aprendizagem: ',taxa_aprendizagem
                print labelinc,"e seus pesos:"
                print "w0 =", pesos[0], "  w1 =", pesos[1], "  w2 =", pesos[2]
                print("-" * 78)
                if cont == 6:
                    break

        if cont == 6:
            break

    #plotar o grafico com as linhas
    plt.show()

    #reinicio com os pesos originais
    pesos[0] = w0
    pesos[1] = w1
    pesos[2] = w2

    pass

def classificacao_perfeita(pesos,dataset,taxa_aprendizagem):

    #pesos iniciais
    w0 = pesos[0]
    w1 = pesos[1]
    w2 = pesos[2]

    #lista de pesos finais
    pesos_finais = []

    #numero de passos
    passos = 0

    #numero de epocas
    epocas = 1

    #numero de atualizacoes de pesos
    updates = 0

    while True:

        #amostras erradas
        erradas = 0

        #calculo da atualizacao dos pesos
        for linhas_tab in dataset:

            #calculo da funcao de ativacao
            predicao = pred(linhas_tab, pesos)

            #contagem das predicoes erradas
            if linhas_tab[-1] != predicao:
                erradas += 1

            #calculo do erro
            erro = linhas_tab[-1] - predicao

            #atualizacao do calculo dos pesos para w0
            pesos[0] = pesos[0] + taxa_aprendizagem * erro

            #atualizacao do calculo dos pesos para os demais pesos
            for k in range(len(linhas_tab)-1):
                pesos[k + 1] = pesos[k + 1] + taxa_aprendizagem * erro * linhas_tab[k]

            #calculo do numero de atualizacoes
            if erro != 0:
                updates += 1

            passos += 1

            print"Pesos: ", pesos
            print"Passo: ", passos
            print"Numero de atualizacoes: ", updates
            print"Epoca: ", epocas
            print"Amostras erradas ate agora nessa epoca: ",erradas
            print("-" * 78)

        #calculo da acuracia
        perc = 100 - ((float(erradas)/float(len(dataset)))*100)
        print("-" * 78)
        print"Acuracia: ",perc,"%"
        print("-" * 78)
        print("-" * 78)

        epocas += 1

        if perc == 100.0:
            break

    #pesos finais
    for w in range(0,3):
        pesos_finais.append(pesos[w])

    #reinicio com os pesos originais
    pesos[0] = w0
    pesos[1] = w1
    pesos[2] = w2

    return pesos,pesos_finais,updates

def plotfig1(dataset):

    x1,x2,y1,y2 = lista_de_classes(dataset)

    # plotando o grafico 1
    ax = plt.subplot(1, 1, 1)
    ax.scatter(x2, y2, marker='o', color=['black'], edgecolors='black', label='Classe -1')
    ax.scatter(x1, y1, marker='o', color=['white'], edgecolors='black', label='Classe 1')
    # ajustes
    ax.set_xlim(0, 1)
    ax.set_xlabel('i1')
    plt.xticks(np.arange(0, 1.1, step=0.1))
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('i2')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title("Amostras da Figura 1")
    plt.legend()
    plt.show()

    return x1,x2,y1,y2

def plotfig2(pesos,dataset):

    w0 = pesos[0]
    w1 = pesos[1]
    w2 = pesos[2]

    x1,x2,y1,y2 = lista_de_classes(dataset)

    #plotando o grafico 2
    ax = plt.subplot(1, 1, 1)
    ax.scatter(x2,y2,marker='o',color=['black'],edgecolors='black', label='Classe -1')
    ax.scatter(x1,y1,marker='o',color=['white'],edgecolors='black', label='Classe 1')
    ax.axvline(0, color = "black")
    ax.axhline(0, color = "black")
    #plotando a Linha inicial
    #equacao geral da reta: a*x + b*y + c = 0
    #equacao rearranjada: y = (((-a)/b)*(x)) + ((-c)/b)
    xli = np.linspace(-999,999,1000)
    yli = (((-w1)/w2)*(xli)) + ((-w0)/w2)
    plt.plot(xli,yli,label="Linha inicial")
    #ajustes
    ax.set_xlim(-2.5, 2.5)
    ax.set_xlabel('i1')
    ax.set_ylim(-2.5, 2.5)
    ax.set_ylabel('i2')
    ax.grid(True)
    plt.title("Linha inicial     Pesos: w0=%s, w1=%s e w2=%s" % (str(pesos[0]), str(pesos[1]), str(pesos[2])))
    plt.legend()
    plt.show()

def plotfigfinal(pesos,pesos_finais,dataset):

    w0 = pesos[0]
    w1 = pesos[1]
    w2 = pesos[2]

    x1,x2,y1,y2 = lista_de_classes(dataset)

    ax = plt.subplot(1, 1, 1)
    ax.scatter(x2, y2, marker='o', color=['black'], edgecolors='black', label='Classe -1')
    ax.scatter(x1, y1, marker='o', color=['white'], edgecolors='black', label='Classe 1')
    ax.axvline(0, color="black")
    ax.axhline(0, color="black")
    # plotando a Linha inicial
    # equacao geral da reta: a*x + b*y + c = 0
    # equacao rearranjada: y = (((-a)/b)*(x)) + ((-c)/b)
    xli = np.linspace(-999, 999, 1000)
    yli = (((-w1) / w2) * (xli)) + ((-w0) / w2)
    plt.plot(xli, yli, label='Linha inicial')
    # plotando a Linha final
    w0 = pesos_finais[0]
    w1 = pesos_finais[1]
    w2 = pesos_finais[2]
    yli = (((-(pesos_finais[1])) / (pesos_finais[2])) * (xli)) + ((-(pesos_finais[0])) / (pesos_finais[2]))
    plt.plot(xli, yli, label='Linha final')
    # ajustes
    ax.set_xlim(-2.5, 2.5)
    ax.set_xlabel('i1')
    ax.set_ylim(-2.5, 2.5)
    ax.set_ylabel('i2')
    ax.grid(True)
    plt.title("Linha final     Pesos: w0=%s, w1=%s e w2=%s" % (str(pesos_finais[0]),str(pesos_finais[1]),str(pesos_finais[2])))
    plt.legend()
    plt.show()