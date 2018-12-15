import matplotlib.pyplot as plt
import numpy as np

def pred(linhas_tab, pesos):

    ativacao = pesos[0]

    for i in range(len(linhas_tab)-1):

        ativacao += pesos[i + 1] * linhas_tab[i]

    return 1.0 if ativacao > 0.0 else -1.0

def atualizar_pesos(x1,x2,y1,y2,pesos,dataset,taxa_aprendizagem):

    #pesos iniciais
    w0 = pesos[0]
    w1 = pesos[1]
    w2 = pesos[2]

    #contador
    cont = 1

    # plotando o grafico 3
    ax3 = plt.subplot(1, 1, 1)
    ax3.scatter(x2, y2, marker='o', color=['black'], edgecolors='black', label='Classe -1')
    ax3.scatter(x1, y1, marker='o', color=['white'], edgecolors='black', label='Classe 1')
    ax3.axvline(0, color="black")
    ax3.axhline(0, color="black")
    # plotando a Linha inicial
    # equacao geral da reta: a*x + b*y + c = 0
    # equacao rearranjada: y = (((-a)/b)*(x)) + ((-c)/b)
    xli = np.linspace(-999, 999, 1000)
    yli = (((-w1) / w2) * (xli)) + ((-w0) / w2)
    plt.plot(xli, yli, label='Linha inicial (1)')

    for linhas_tab in dataset:

        predicao = pred(linhas_tab, pesos)

        #calculo do erro
        erro = linhas_tab[-1] - predicao

        #atualizacao do calculo dos pesos para w0
        pesos[0] = pesos[0] + taxa_aprendizagem * erro

        #atualizacao do calculo dos pesos para os demais pesos
        for k in range(len(linhas_tab)-1):
            pesos[k + 1] = pesos[k + 1] + taxa_aprendizagem * erro * linhas_tab[k]

        # plotando as linhas e pesos apenas das amostras classificadas erroneamente e que foram atualizadas
        if erro != 0:
            cont += 1
            yli = (((-pesos[1]) / pesos[2]) * (xli)) + ((-pesos[0]) / pesos[2])
            #incrementar label
            labelinc = 'Linha '+ str(cont)
            plt.plot(xli, yli, label=labelinc)
            # ajustes
            ax3.set_xlim(-2.5, 2.5)
            ax3.set_xlabel('i1')
            ax3.set_ylim(-2.5, 2.5)
            ax3.set_ylabel('i2')
            ax3.grid(True)
            plt.legend()
            print 'Taxa de aprendizagem: ',taxa_aprendizagem
            print labelinc,"e seus pesos:"
            print "w0 =", pesos[0], "  w1 =", pesos[1], "  w2 =", pesos[2]
            print("-" * 78)

    #plotar o grafico com as linhas
    plt.show()

    #reinicio com os pesos originais
    pesos[0] = w0
    pesos[1] = w1
    pesos[2] = w2

    pass

def classificacao_perfeita(pesos,dataset,taxa_aprendizagem):

    #numero de passos
    passos = 0

    #numero de epocas
    epocas = 1

    #numero de atualizacoes de pesos
    updates = 0

    while True:

        #amostras erradas
        erradas = 0

        for linhas_tab in dataset:

            predicao = pred(linhas_tab, pesos)

            #calculo de predicoes erradas
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

    return pesos,updates


