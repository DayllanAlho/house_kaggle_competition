# Descrevendo aqui as diferenças de treinamento utilizados entre os treinamentos de Ensemble. Passando por todos os métodos.

# Desafio Kaggle - Previsão de Preços de Residências em Ames, Iowa

## Compreensão do Desafio
Ao analisar o código fornecido, fica claro que o desafio consiste em prever os preços de residências em Ames, Iowa. O conjunto de dados é composto por 79 variáveis que abrangem diversos aspectos relacionados às próprias habitações e seus arredores. Essas variáveis incluem informações como o número de quartos, o tamanho do quintal, a localização da casa, a proximidade de uma ferrovia, entre outros. Dada a complexidade desses fatores e a falta de evidência completa de seus impactos, o objetivo é criar um modelo capaz de fornecer uma faixa de valores para essas residências com base em suas características específicas.

## Métodos de Treinamento - Ensemble

O código fornecido no arquivo "houses_kaggle_competition.ipynb" apresenta oito métodos de treinamento diferentes. Abaixo, segue uma breve descrição de cada um:

### 1. Árvore de Decisão:

- Modelo que toma decisões com base em regras de decisão hierárquicas.
- Parâmetros principais: max_depth (profundidade máxima da árvore) e min_samples_leaf (número mínimo de amostras em uma folha).

### 2. KNN (K-Nearest Neighbors):
- Algoritmo de aprendizado supervisionado que classifica pontos de dados com base na proximidade com os vizinhos mais próximos.
- Não especifica hiperparâmetros diretamente.

### 3. Ridge (Regressão Linear com Regularização L2):
- Extensão da regressão linear que adiciona um termo de regularização L2 à função de custo.
- Pode ser treinado tanto com o target original quanto com o log do target.

### 4. SVM (Support Vector Machine):
- Utiliza Máquinas de Vetores de Suporte para problemas de classificação e regressão.
- Utiliza um kernel linear.

### 5. Floresta Aleatória:
- Conjunto de árvores de decisão treinadas de forma independente.
- Reduz overfitting e captura relações não lineares.
### 6. Boosted Trees (AdaBoost e Gradient Boosting):
- AdaBoost foca em exemplos mal classificados.
- Gradient Boosting ajusta os erros residuais de modelos anteriores.
### 7. Stacking:
- Combina as previsões de vários modelos de base usando um meta-modelo.
- Pode melhorar o desempenho combinando diferentes perspectivas dos modelos individuais.
### 8. XGBoost:
- Implementação otimizada de Gradient Boosting.
- Eficiente para grandes conjuntos de dados e suporta early stopping.

# Diferenças de Abordagem no Treinamento - Ensemble
Ao explorar os métodos de Ensemble no contexto do desafio da Kaggle "House Prices", é crucial compreender as nuances teóricas e práticas de treinamento de cada abordagem. Aqui estão as divergências fundamentais em termos de teoria e implementação prática:

## Diferenças de Treinamento Teóricas - Ensemble
### Diversidade dos Modelos:

- Árvores de Decisão, KNN, Ridge, SVM: Cada modelo age de maneira independente.
- Random Forest e Boosted Trees: A construção é sequencial, corrigindo os erros do modelo anterior.
- Stacking: Combinação de previsões de modelos individuais.
- XGBoost: Método de boosting, enfatizando modelos que corrigem os erros anteriores.

### Natureza do Aprendizado:

- Árvores de Decisão, KNN, Ridge, SVM: Aprendizado independente do conjunto de dados completo.
- Random Forest e Boosted Trees: Treinamento em subamostras aleatórias (bagging) ou distribuição de erros.
- Stacking: Modelos base treinados independentemente; meta-modelo aprende a combinar previsões.
- XGBoost: Algoritmo de boosting, enfocando na correção dos erros.

### Configuração dos Hiperparâmetros:
- Árvores de Decisão, KNN, Ridge, SVM: Hiperparâmetros específicos para cada modelo.
- Random Forest e Boosted Trees: Hiperparâmetros para modelos base e do ensemble.
- Stacking: Hiperparâmetros para modelos base e meta-modelo.
- XGBoost: Diversos hiperparâmetros ajustáveis para otimizar.

### Complexidade de Treinamento:
- Árvores de Decisão, KNN, Ridge, SVM: Treinamento relativamente simples e rápido.
- Random Forest e Boosted Trees: Pode ser mais lento devido à construção sequencial.
- Stacking: Pode ser mais lento devido ao treinamento de vários modelos.
- XGBoost: Pode ser mais lento devido ao algoritmo de otimização gradiente.

### Interpretabilidade:
- Árvores de Decisão, KNN, Ridge, SVM: Interpretáveis individualmente.
- Random Forest e Boosted Trees: Difíceis de interpretar devido à combinação de previsões.
- Stacking: A interpretabilidade depende do meta-modelo escolhido.
- XGBoost: Difícil de interpretar como método de boosting.

## Modificações Práticas Propostas - Ensemble
### Árvore de Decisão:

- Aplicação de seleção de atributos 'SelectKBest'.
- Busca por hiperparâmetros usando GridSearchCV.

### KNN (K-Nearest Neighbors):
- Utilização de BallTree para vizinhança.
- Seleção de atributos para redução de dimensionalidade.
- Aplicação de GridSearchCV para otimização.

### Ridge (Regressão Linear com Regularização L2):
- Utilização do algoritmo de otimização L-BFGS.
- Aplicação de técnica de regularização.
- Busca por hiperparâmetros usando GridSearchCV.

### SVM (Support Vector Machine):
- Utilização de kernel polinomial.
- Técnica de regularização aplicada.
- Busca por hiperparâmetros usando GridSearchCV.

### Floresta Aleatória:
- Utilização de um número maior de árvores.
- Otimização dos hiperparâmetros com RandomizedSearchCV.

### Boosted Trees (AdaBoost e Gradient Boosting):
- Aumento do número de árvores e redução da taxa de aprendizagem.
- Aplicação de GridSearchCV para otimização.

### Stacking:
- Utilização de um meta-modelo com regressão linear.
- Expansão do número de modelos base.

### XGBoost:
- Aumento do número de árvores e redução da taxa de aprendizagem.
- Aplicação de algoritmo de early stopping.
- Busca por hiperparâmetros usando GridSearchCV.

Essas adaptações visam aprimorar a performance de cada método e destacam a flexibilidade necessária para lidar com as peculiaridades de cada algoritmo.

## Sugestões de melhoria no código

- Árvores de decisão

Aplicação do Recursive Feature Elimination (RFE) é uma técnica que recursivamente remove as variáveis menos importantes, ajustando o modelo nas variáveis restantes. Ela pode ser explorada para melhorar a performace e a interpretabilidade do modelo.
```
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

num_features_to_select = 5

base_model = RandomForestRegressor()

selector = RFE(base_model, n_features_to_select=num_features_to_select)

model = DecisionTreeRegressor()

pipe = Pipeline([
    ('preproc', preproc),
    ('selector', selector),
    ('model', model)
])

param_grid = {
    'model__max_depth': [10, 20, 30, 40],
    'model__min_samples_leaf': [5, 10, 15, 20]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X, y_log)

selected_features = np.array(range(len(X.columns)))[selector.support_]

best_model = grid_search.best_estimator_

score = np.sqrt(-cross_val_score(best_model, X, y_log, cv=5, scoring='neg_mean_squared_error').mean())

print("Melhores parâmetros:", grid_search.best_params_)
print("Atributos selecionados:", selected_features)
print("RMSE médio:", score)

```

### Explicação da Mudança:

- Utilização do RFE: O uso de RFE pode ser preferível, pois ele seleciona as melhores features de forma recursiva, eliminando aquelas que contribuem menos para o desempenho do modelo. Isso pode ser mais eficaz do que uma abordagem fixa, pois ajusta dinamicamente o número de atributos com base na importância relativa.

- Mudança para RandomForestRegressor: A utilização de RandomForestRegressor como modelo base para a seleção de features pode ser benéfica, pois random forests tendem a capturar melhor a importância das features em comparação com um único modelo de árvore de decisão.

- Métrica de Avaliação: Substituímos o cálculo do RMSE para uma forma mais legível e adequada, usando np.sqrt(-cross_val_score()).

Essas mudanças podem proporcionar uma seleção de features mais robusta, melhorando assim o desempenho do modelo de árvore de decisão.

## KNN
Podemos explorar o RFE (Recursive Feature Elimination) para seleção de atributos.
```
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsRegressor, BallTree
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

num_features_to_select = 5

selector = RFE(KNeighborsRegressor(), n_features_to_select=num_features_to_select)

model = KNeighborsRegressor()

pipe_knn = make_pipeline(preproc, selector, BallTree(), model)

param_grid = {
    'kneighborsregressor__n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30],
    'rfe__n_features_to_select': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
}

param_grid = {f'rfe__{key}': value for key, value in param_grid.items()}

search_knn = GridSearchCV(
    pipe_knn,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=make_scorer(mean_squared_error, greater_is_better=False)
)

search_knn.fit(X, y_log)

scores = np.sqrt(-cross_val_score(search_knn.best_estimator_, X, y_log, cv=5, scoring='neg_mean_squared_error').mean())

print("RMSE médio:", scores)
print(f'Melhores parâmetros {search_knn.best_params_}')
print(f'Melhor RMSE {abs(search_knn.best_score_)}')
```
### Explicação da Mudança:

- Utilização do RFE: O uso de RFE pode ser preferível, pois ele seleciona as melhores features de forma recursiva, eliminando aquelas que contribuem menos para o desempenho do modelo.

- Mudança para BallTree: Utilizamos o uso de BallTree como o algoritmo de vizinhança, para reduzir a dimensionalidade dos dados.

- Métrica de Avaliação: Utilizamos o cálculo do RMSE para uma forma mais legível e adequada, usando np.sqrt(-cross_val_score(...)).

Essas mudanças podem proporcionar uma seleção de features mais eficaz, melhorando assim o desempenho do modelo KNN.

## Ridge
Utilizamos o modelo Ridge com o uso da validação cruzada para encontrar o melhor valor de ``alpha``. Além disso, usamos diretamente a implementação do Ridge fornecida pelo scikit-learn.

```
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

preproc = StandardScaler()
model = Ridge()

pipe_ridge = make_pipeline(preproc, model)

param_grid = {'ridge__alpha': np.linspace(0.5, 2, num=20)}

search_ridge = GridSearchCV(
    pipe_ridge,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring=rmse_neg
)

search_ridge.fit(X, y_log)

print(f'Melhores parâmetros {search_ridge.best_params_}')
print(f'Melhor Score {search_ridge.best_score_}')
```
### Explicação da Mudança:
- Uso do Ridge: Utilizamos diretamente a implementação do Ridge fornecida pelo scikit-learn.

- Pipeline Simples: O pipeline agora consiste apenas no pré-processamento (StandardScaler) e no modelo (Ridge).

- Busca em Grade com Validação Cruzada: A busca em grade é realizada diretamente no modelo Ridge para encontrar o melhor valor de alpha usando a validação cruzada.

# Descrevendo as modificações proposta