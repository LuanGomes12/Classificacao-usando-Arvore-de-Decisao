% ********** Luan Gomes Magalhães Lima - 473008                      ********** 
% ********** Tópicos Especiais em Telecomunicações 1 - Prática 3     **********

% Inicializações
clear all;
close all;
clc;

%% Ánalise exploratória e escolha dos atributos 
% Carregar a base de dados
load("Classe1.mat");
load("Classe2.mat");

% Sinal de entrada
dados_c1 = Classe1';
dados_c2 = Classe2';

% Classes de cada sinal
classe_dados_c1 = zeros(50, 1);
classe_dados_c2 = ones(50, 1);

% Sinal de entrada com as classes
dados_c1(:, end + 1) = classe_dados_c1;
dados_c2(:, end + 1) = classe_dados_c2;

% Base de dados concatenada
base = [dados_c1; dados_c2];

% Matriz de entrada
X = base(:, 1:500);
% Matriz de saída
y = base(:, 501);

% Atributos a serem analisados
% 1º Atributo: Média
media = mean(X, 2);

% 2º Atributo: Kurtose
kurtose = kurtosis(X, 0, 2);

% 3º Atributo: Assimetria
assimetria = skewness(X, 0, 2);

% 4º Atributo: Raíz Quadrada do Erro Quadrático Médio
rms = rms(X, 2);

% 5º Atributo: Desvio Padrão
desvio_padrao = std(X, 1, 2);

% 6º Atributo: Variância
variancia = var(X, 0, 2);

% 7º Atributo: Entropia
entropia = zeros(100, 1);
for i = 1:100
    entropia(i, 1) = entropy(X(i, :));
end

% 8º Atributo: Moda
moda = zeros(100, 1);
for i = 1:100
    moda(i, 1) = mode(X(i, :));
end

% Atributos escolhidos: Assimetria, Desvio Padrão, Entropia, Kurtose
% Atributo 1: Assimetria
% Atributo 2: Desvio-Padrão
% Atributo 3: Entropia
% Atributo 4: Kurtose
X_tratado = [assimetria, desvio_padrao, entropia, kurtose];

% Cores representando cada classe -> Vermelho: 0 e Azul: 1
paleta_cores = [0 0 1; 1 0 0];
cores = paleta_cores((y == 0) + 1, :);

% Gráficos de dispersão dos 8 possíveis atributos
% Plotagem apenas dos 4 atributos escolhidos
figure;
scatter(assimetria, 0, [], cores, 'filled');
title("Assimetria");

figure;
scatter(desvio_padrao, 0, [], cores, 'filled');
title("Desvio Padrão");

figure;
scatter(entropia, 0, [], cores, 'filled');
title("Entropia");

figure;
scatter(kurtose, 0, [], cores, 'filled');
title("Kurtose");

% figure;
% scatter(media, 0, [], cores, 'filled');
% title("Média");

% figure;
% scatter(moda, 0, [], cores, 'filled');
% title("Moda");

% figure;
% scatter(rms, 0, [], cores, 'filled');
% title("RMS");

% figure;
% scatter(variancia, 0, [], cores, 'filled');
% title("Variância");

%% Criação do Classificador
% Organização da base tratada
features = X_tratado;
labels = y;
base = features;
base(:, end + 1) = labels;

% Matriz que armazena os acertos do classificador
matriz_acertos = zeros(length(base(:, 1)), 1);

% Implementação do Leave-One-Out
for i = 1 : length(base(:, 1))

    % Divisão dos índices em teste e treino
    ind_teste = i;
    ind_treino = find(1:size(base,1) ~= ind_teste);

    % Divisão da base em treinamento e teste
    % Conjunto de treinamento
    X_train =  base(ind_treino, 1 : end-1); 
    y_train = base(ind_treino, end);

    % Conjunto de Teste
    X_test = base(ind_teste, 1 : end-1);
    y_test = base(ind_teste, end);

    % Base atual que vai ser passada como parâmetro para a construção da árvore
    base_atual = X_train;
    base_atual(:, end + 1) =  y_train;
    rotulos = y_train;
    
    % Matriz que identifica a quantidade de atributos da base, os quais
    % serão utilizados para a construção da árvore
    matriz = 1 : size(base_atual, 2);

    % Construção da árvore para cada iteração do leave-one-out
    classificador = construir_arvore(base_atual, rotulos, matriz); 
    no_atual = classificador;

    % Percorre a árvore para classificar a amostra de teste
    while ~isempty(no_atual.atributo)
        amostra_teste = X_test(no_atual.atributo);
        if amostra_teste <= no_atual.valor
            no_atual = classificador.filhos{1};
        else
            no_atual = classificador.filhos{2};
        end
    end

    % Classe prevista
    predict = no_atual.classe;

    % Condicional que irá povoar a matriz de acertos
    if predict == y_test
        matriz_acertos(i) = 1;
    end
end

% Quantidade total de acertos
total_acertos = 0;
for i = 1 : length(base(:, 1))
    if matriz_acertos(i) == 1
        total_acertos = total_acertos + 1;
    end
end

% Acurácia do Classificador
qtd_total_amostras = length(base(:, 1));
acuracia = total_acertos/qtd_total_amostras;
fprintf("Acurácia do classificador: %.2f", acuracia*100);

%% Funções utilizadas
% Função que serve para construir a àrvore de decisão utilizando recursividade
function no = construir_arvore(dados, rotulos, matriz)
    no = struct('atributo', [], 'valor', [], 'filhos', [], 'classe', []);
    features = dados(:, 1 : end-1);

     % Verificar critério de parada: todos os rótulos são iguais
     if todosIguais(rotulos)
        no.classe = rotulos(1);
        return;
     end

    % Verificar critério de parada: não há mais atributos para dividir a
    % base
    if isempty(matriz)
        no.classe = mode(rotulos);
    end

    % Encontrar o melhor atributo para dividir
    [ind_atributo, valor] = melhor_divisao(dados);
    [atributo_escolhido, matriz_r] = identifica_atributo(ind_atributo, matriz);
    
    % Construção da nó atual
    no.atributo = atributo_escolhido;
    no.valor = valor;

    % Dividir os dados com base no melhor atributo
    % Filho da esquerda
    features_esq = features(features(:, ind_atributo) <= valor, :);
    rotulos_esq = rotulos(features(:, ind_atributo) <= valor, :);
    dados_esq = features_esq;
    dados_esq(:, end + 1) = rotulos_esq; 
    % Retira o melhor atributo da base para construção dos próximos nós
    dados_esq(:, ind_atributo) = [];
    
    % Filho da direita
    features_dir = features(features(:, ind_atributo) > valor, :);
    rotulos_dir = rotulos(features(:, ind_atributo) > valor, :);
    dados_dir = features_dir;
    dados_dir(:, end + 1) = rotulos_dir;
    % Retira o melhor atributo da base para construção dos próximos nós
    dados_dir(:, ind_atributo) = [];

    % Construir os nós da árvore recursivamente
    no.filhos{1} = construir_arvore(dados_esq, rotulos_esq, matriz_r); % Filho da esquerda
    no.filhos{2} = construir_arvore(dados_dir, rotulos_dir, matriz_r); % Filho da direita
end

% Verifica o melhor atributo e melhor limiar com base no ganho de informação
function [atributo, limiar] = melhor_divisao(base)

    % Entropia da base
    H_decisao = calcular_entropia(base, 0, 0);

    % Matriz ganho que vai selecionar o melhor atributo
    % Saída matriz ganho:
    % 1º Linha: ganho de informação
    % 2º Linha: limiares de cada atributo
    % As colunas referem-se aos atributos
    matriz_ganho = [];
    pos_matriz_ganho = 0;

    % Percorre os atributos
    for i = 1 : length(base(1, 1 : end-1))

        % Matriz dos possíveis limiares
        matriz_theta = [];
        pos_matriz_theta = 0;

        % Ordena a amostra
        base_ordenada = sortrows(base, i);

        % Matriz temporária para armazenar os possíveis limiares de cada atributo
        pos_H_temp = 0;
        H_temp = [];

        % Percorre as amostras do atributo i
        for j = 1 : length(base(: , 1))

            % Condição para impedir que o teste seja feito em um elemento que
            % não existe na base
            if j < length(base(:, end))

                if base_ordenada(j, end) ~= base_ordenada(j+1, end)
                    entropia = calcular_entropia(base_ordenada, j, 1);
                    
                    % Armazena as possíveis entropias
                    H_temp(pos_H_temp + 1) = entropia;
                    pos_H_temp = pos_H_temp + 1;

                    % Armazena os possíveis limiares (thetas)
                    matriz_theta(pos_matriz_theta + 1) = base_ordenada(j, i);
                    pos_matriz_theta = pos_matriz_theta + 1;
                end
            end
        end

        % Pegar dentro da matriz H_temp o limiar escolhido
        % Percorre a matriz H_temp e pega o menor valor
        H_atributo = min(H_temp);
        ind_melhor_atributo = find(H_temp == H_atributo);
        if length(ind_melhor_atributo) > 1
            ind_melhor_atributo = ind_melhor_atributo(1);
        end
        limiar_escolhido = matriz_theta(1, ind_melhor_atributo);

        ganho = calcular_ganho(H_decisao, H_atributo);
        % Ganho de informação (primeira linha)
        matriz_ganho(1, pos_matriz_ganho + 1) = ganho;
        % Limiar de cada de atributo (segunda linha)
        matriz_ganho(2, pos_matriz_ganho + 1) = limiar_escolhido;
        pos_matriz_ganho = pos_matriz_ganho + 1;

    end

    % Escolhas dos atributos
    indice = find(matriz_ganho(1, :) ==   max(matriz_ganho(1, :)));
    if length(indice) > 1
        indice = indice(1);
    end
    atributo = indice;
    limiar = matriz_ganho(2, indice);
end

% Calcula a entropia
% Paramêtro nó: 0 -> Nó pai, 1 -> Nó filho
function entropia = calcular_entropia(base, posicao_amostra, no)
    
    total_amostras = length(base(:, 1));
    % Calcular Entropia do nó pai (entropia de decisão)
    if no == 0
        amostras_c1 = length(find(base(: , end) == 0));
        amostras_c2 = length(find(base(:, end) == 1));
        entropia = - ((amostras_c1/total_amostras)*log2(amostras_c1/total_amostras)) - ((amostras_c2/total_amostras)*log2(amostras_c2/total_amostras));

    % Entropia do nó filho    
    else
        % Parte esquerda do limiar
        amostras_esq = length(base(1:posicao_amostra, end));
        amostras_c1 = length(find(base(1:posicao_amostra, end) == 0)); 
        amostras_c2 = length(find(base(1:posicao_amostra, end) == 1));
        H_esq = -((amostras_c1/amostras_esq)*log2(amostras_c1/amostras_esq)) -((amostras_c2/amostras_esq)*log2(amostras_c2/amostras_esq));
        
        if isnan(H_esq)
            H_esq = 0;
        end        

        % Parte direita do limiar
        amostras_dir = length(base(posicao_amostra+1 : end, end));
        amostras_c1 = length(find(base(posicao_amostra+1 : end, end) == 0)); 
        amostras_c2 = length(find(base(posicao_amostra+1 : end, end) == 1));
        H_dir = -((amostras_c1/amostras_dir)*log2(amostras_c1/amostras_dir)) -((amostras_c2/amostras_dir)*log2(amostras_c2/amostras_dir));        

        if isnan(H_dir)
            H_dir = 0;
        end

        entropia = (length(base(1:posicao_amostra, end))/total_amostras)*H_esq + (length(base(posicao_amostra+1:end, end))/total_amostras)*H_dir;
    end
end

% Calcula o ganho de informação
function ganho = calcular_ganho(H_classe, H_atributo)
    ganho = H_classe - H_atributo;
end

% Verifica se todos os elementos de um vetor são iguais
function iguais = todosIguais(vetor)
    iguais = all(vetor == vetor(1));
end

% Identifica o atributo dentro da matriz de atributos e exclui o atributo
% que foi utilizado no nó atual
function [atributo, matriz_r] = identifica_atributo(ind_atributo, matriz)
    atributo = matriz(ind_atributo);
    matriz_r = setdiff(matriz, matriz(ind_atributo));
end