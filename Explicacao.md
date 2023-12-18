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

| Modelos                            | Descrição                                                                      |
|------------------------------------|--------------------------------------------------------------------------------|
| Árvores de Decisão, KNN, Ridge, SVM | Cada modelo age de maneira independente.                                       |
| Random Forest e Boosted Trees       | A construção é sequencial, corrigindo os erros do modelo anterior.              |
| Stacking                           | Combinação de previsões de modelos individuais.                                |
| XGBoost                            | Método de boosting, enfatizando modelos que corrigem os erros anteriores.       |
||

### Natureza do Aprendizado:
| Modelos                            | Descrição                                                                      |
|------------------------------------|--------------------------------------------------------------------------------|
|Árvores de Decisão, KNN, Ridge, SVM|Aprendizado independente do conjunto de dados completo.|
|Random Forest e Boosted Trees|Treinamento em subamostras aleatórias (bagging) ou distribuição de erros.|
|Stacking|Modelos base treinados independentemente; meta-modelo aprende a combinar previsões.|
|XGBoost|Algoritmo de boosting, enfocando na correção dos erros.|
||
### Configuração dos Hiperparâmetros:
| Modelos                            | Descrição                                                                      |
|------------------------------------|--------------------------------------------------------------------------------|
|Árvores de Decisão, KNN, Ridge, SVM| Hiperparâmetros específicos para cada modelo.|
|Random Forest e Boosted Trees|Hiperparâmetros para modelos base e do ensemble.
|Stacking|Hiperparâmetros para modelos base e meta-modelo.
|XGBoost|Diversos hiperparâmetros ajustáveis para otimizar.
||
### Complexidade de Treinamento:
| Modelos                            | Descrição                                                                      |
|------------------------------------|--------------------------------------------------------------------------------|
|Árvores de Decisão, KNN, Ridge, SVM|Treinamento relativamente simples e rápido.|
|Random Forest e Boosted Trees|Pode ser mais lento devido à construção sequencial.|
|Stacking|Pode ser mais lento devido ao treinamento de vários modelos.|
|XGBoost|Pode ser mais lento devido ao algoritmo de otimização gradiente.|
||
### Interpretabilidade:
| Modelos                            | Descrição                                                                      |
|------------------------------------|--------------------------------------------------------------------------------|
|Árvores de Decisão, KNN, Ridge, SVM|Interpretáveis individualmente.|
|Random Forest e Boosted Trees|Difíceis de interpretar devido à combinação de previsões.|
|Stacking|A interpretabilidade depende do meta-modelo escolhido.|
|XGBoost|Difícil de interpretar como método de boosting.|
||

## Modificações Práticas Propostas - Ensemble

| Modelo                         | Modificações Práticas Propostas                                        |
|--------------------------------|---------------------------------------------------------------------|
| Árvore de Decisão              | - Aplicação de seleção de atributos 'SelectKBest'.                   |
|                                | - Busca por hiperparâmetros usando GridSearchCV.                    |
| KNN (K-Nearest Neighbors)      | - Utilização de BallTree para vizinhança.                            |
|                                | - Seleção de atributos para redução de dimensionalidade.           |
|                                | - Aplicação de GridSearchCV para otimização.                        |
| Ridge (Regressão Linear L2)    | - Utilização do algoritmo de otimização L-BFGS.                      |
|                                | - Aplicação de técnica de regularização.                             |
|                                | - Busca por hiperparâmetros usando GridSearchCV.                    |
| SVM (Support Vector Machine)   | - Utilização de kernel polinomial.                                   |
|                                | - Técnica de regularização aplicada.                                |
|                                | - Busca por hiperparâmetros usando GridSearchCV.                    |
| Floresta Aleatória             | - Utilização de um número maior de árvores.                         |
|                                | - Otimização dos hiperparâmetros com RandomizedSearchCV.           |
| Boosted Trees (AdaBoost, Gradient Boosting) | - Aumento do número de árvores e redução da taxa de aprendizagem. |
|                                | - Aplicação de GridSearchCV para otimização.                        |
| Stacking                       | - Utilização de um meta-modelo com regressão linear.                |
|                                | - Expansão do número de modelos base.                               |
| XGBoost                        | - Aumento do número de árvores e redução da taxa de aprendizagem.   |
|                                | - Aplicação de algoritmo de early stopping.                         |
|                                | - Busca por hiperparâmetros usando GridSearchCV.                    |

Essas adaptações visam aprimorar a performance de cada método e destacam a flexibilidade necessária para lidar com as peculiaridades de cada algoritmo.

# Descrevendo as modificações proposta
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

## SVM
Utilizamos o SVM com kernel RBF (Radial Basis Function) para melhorar o código, pois é uma escolha comum quando se lida com relações não lineares nos dados. Além disso, a redução de dimensionalidade usando Isomap pode ser benéfica para capturar estruturas complexas nos dados. 

```
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

pipe_svm_rbf = make_pipeline(StandardScaler(), Isomap(), SVR(kernel='rbf'))

param_grid = {
    'isomap__n_components': [5, 10, 15],
    'svr__C': [0.1, 1, 10],
    'svr__gamma': [0.01, 0.1, 1]
}

rmse = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

search_svm_rbf = GridSearchCV(pipe_svm_rbf, param_grid=param_grid, cv=5, scoring=rmse, n_jobs=-1)
search_svm_rbf.fit(X, y)

best_params = search_svm_rbf.best_params_
best_rmse = abs(search_svm_rbf.best_score_)

print(f'Melhores parâmetros: {best_params}')
print(f'Melhor RMSE: {best_rmse}')
```

### Explicação da Mudança:
- Kernel RBF: O SVM com kernel RBF é capaz de mapear os dados para um espaço de alta dimensionalidade, onde relações não lineares podem ser melhor capturadas.

- Isomap para Redução de Dimensionalidade: O Isomap é uma técnica de redução de dimensionalidade que preserva as distâncias geodésicas entre todos os pontos. Ele é útil para capturar estruturas não lineares nos dados, sendo uma alternativa ao PCA.

- Pipeline com Pipeline do scikit-learn: Utilizamos o Pipeline para encadear várias etapas do processo, incluindo a redução de dimensionalidade (Isomap) e o treinamento do modelo SVM com kernel RBF.

- Tuning de Hiperparâmetros: Ajustamos os hiperparâmetros do SVM (parâmetros do kernel RBF, C e gamma) e do Isomap (número de componentes principais) para otimizar o desempenho do modelo.

## Random Florest

O Random Forest é um poderoso algoritmo de ensemble, e a otimização de hiperparâmetros pode melhorar significativamente seu desempenho.

```
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

def train_random_forest_with_randomized_search(X, y_log):


    preproc = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), slice(None)), 
            ('imp', SimpleImputer(strategy='mean'), slice(None))
        ])

    model = RandomForestRegressor()

    pipe = make_pipeline(preproc, model)

    param_dist = {
        'randomforestregressor__n_estimators': [50, 100, 200, 300],  
        'randomforestregressor__max_depth': [10, 20, 30, 40, 50], 
        'randomforestregressor__min_samples_leaf': [5, 10, 20, 30]  
    }

    rmse = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=100, scoring=rmse, cv=5, n_jobs=-1, random_state=42)
    random_search.fit(X, y_log)

    best_params = random_search.best_params_
    best_rmse = abs(random_search.best_score_)

    print(f'Melhores hiperparâmetros: {best_params}')
    print(f'Melhor RMSE: {best_rmse}')

    score = cross_val_score(random_search.best_estimator_, X, y_log, cv=5, scoring=rmse)
    print(f"RMSE médio: {abs(score.mean())}")

    return best_params, best_rmse

# Supondo que você tenha dados de treinamento X_train e rótulos y_log_train
best_params, best_rmse = train_random_forest_with_randomized_search(X_train, y_log_train)
```

### Explicação das mudanças:
- ColumnTransformer para Pré-processamento: Um ColumnTransformer é usado para aplicar diferentes transformações a diferentes subconjuntos de features. Neste caso, aplicamos StandardScaler às features numéricas e SimpleImputer para tratar valores ausentes.

- Ajustes de Hiperparâmetros do Random Forest: Utilizamos RandomizedSearchCV para explorar aleatoriamente diferentes combinações de hiperparâmetros, o que pode ser mais eficiente que uma busca em grade completa.

- Aumento do Número de Iterações: Aumentamos o número de iterações para 100 no RandomizedSearchCV, permitindo uma exploração mais abrangente do espaço de hiperparâmetros.

## Boosted Trees

A Validação Holdout e Feature Engineering são práticas cruciais no desenvolvimento de modelos preditivos, proporcionando melhorias significativas no desempenho e na interpretabilidade. Ao aplicarmos essas técnicas ao modelo de Boosted Trees, buscamos aprimorar sua capacidade de generalização, identificando padrões mais complexos nos dados e melhorando sua capacidade de se adaptar a diferentes conjuntos de dados.
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

def train_boosted_trees(X_train, y_log_train, X_test, y_log_test, n_estimators=100, learning_rate=0.01):

cachedir = mkdtemp()

    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None), n_estimators=n_estimators, learning_rate=learning_rate)

    pipe = make_pipeline(FunctionTransformer(func=preproc), model, memory=cachedir)

    pipe.fit(X_train, y_log_train)

    y_log_pred = pipe.predict(X_test)

    residuals = y_log_test - y_log_pred
    sns.scatterplot(x=y_log_pred, y=residuals)
    plt.title('Análise de Resíduos')
    plt.xlabel('Previsões (log)')
    plt.ylabel('Resíduos')
    plt.show()

    rmse_test = np.sqrt(mean_squared_error(y_log_test, y_log_pred))
    print(f"RMSE no conjunto de teste: {rmse_test}")

    return pipe
```

### Explicação das Mudanças
- Análise de Resíduos: Após o treinamento do modelo, realizamos uma análise de resíduos. A análise de resíduos nos permite entender como as previsões do modelo diferem dos valores reais. Isso é crucial para identificar padrões não capturados e avaliar a qualidade das previsões.

- Ajuste Manual de Hiperparâmetros: Ao treinar o modelo, permitimos a ajuste manual de hiperparâmetros como o número de árvores (n_estimators) e a taxa de aprendizado (learning_rate). Isso nos dá controle sobre a complexidade do modelo e sua capacidade de se adaptar aos dados.

- Avaliação de Desempenho no Conjunto de Teste: Calculamos e exibimos o Root Mean Squared Error (RMSE) no conjunto de teste. Essa métrica nos fornece uma medida quantitativa da qualidade das previsões em dados não vistos.

Essas mudanças aprimoram a robustez do modelo de Boosted Trees, garantindo uma avaliação realista do seu desempenho e permitindo adaptações eficazes por meio do Feature Engineering e da otimização de hiperparâmetros.

## Stacking
A introdução de GridSearchCV e Recursive Feature Elimination (RFE) para o modelo de Stacking visa aprimorar a otimização de hiperparâmetros e a seleção de características, respectivamente. Essas mudanças têm o objetivo de aperfeiçoar o desempenho do modelo Stacking, tornando-o mais adaptável aos dados e potencialmente mais preciso.

```
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from tempfile import mkdtemp

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

gboost = GradientBoostingRegressor(n_estimators=100)
ridge = Ridge()
svm = SVR(C=1, epsilon=0.05)
adaboost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None))

models_for_stacking = [
    ("gboost", gboost),
    ("adaboost", adaboost),
    ("ridge", ridge),
    ("svm_rbf", svm)
]

stacking_model = StackingRegressor(
    estimators=models_for_stacking,
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

preproc = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), slice(None)),
        ('imp', SimpleImputer(strategy='mean'), slice(None))
    ])

pipe_stacking = make_pipeline(preproc, stacking_model, memory=mkdtemp())

param_dist = {
    'stackingregressor__gboost__n_estimators': randint(50, 200),
    'stackingregressor__ridge__alpha': uniform(0.1, 10),
    'stackingregressor__svm_rbf__C': [0.1, 1, 10],
}

random_search = RandomizedSearchCV(
    estimator=pipe_stacking,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
y_pred = random_search.predict(X_test)

score = rmse(y_test, y_pred)
print("Melhores Hiperparâmetros (Randomized Search):", random_search.best_params_)
print("RMSE no Conjunto de Teste:", score)

param_grid = {
    'stackingregressor__gboost__n_estimators': [100, 150, 200],
    'stackingregressor__ridge__alpha': [1, 5, 10],
    'stackingregressor__svm_rbf__C': [1, 5, 10],
}

grid_search = GridSearchCV(
    estimator=pipe_stacking,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
y_pred_grid = grid_search.predict(X_test)

score_grid = rmse(y_test, y_pred_grid)
print("\nMelhores Hiperparâmetros (Grid Search):", grid_search.best_params_)
print("RMSE no Conjunto de Teste (Grid Search):", score_grid)

selector_rfe = RFE(stacking_model, n_features_to_select=5)
pipe_stacking_rfe = make_pipeline(preproc, selector_rfe, stacking_model, memory=mkdtemp())

pipe_stacking_rfe.fit(X_train, y_train)
y_pred_rfe = pipe_stacking_rfe.predict(X_test)

score_rfe = rmse(y_test, y_pred_rfe)
print("\nRMSE no Conjunto de Teste (RFE):", score_rfe)
```

### Explicação das Mudanças:
- Seleção Automática de Características: O RFE identifica automaticamente as características mais importantes, eliminando aquelas que contribuem menos para a precisão do modelo.
- Redução da Dimensionalidade: Ao selecionar um subconjunto relevante de características, o RFE reduz a dimensionalidade dos dados, o que pode levar a um modelo mais eficiente e menos propenso a overfitting.

## XGBoost
As melhorias implementadas no código do XGBoost representam avanços significativos na otimização do modelo, a introdução do RandomizedSearchCV agiliza a busca por combinações eficazes de hiperparâmetros, economizando recursos computacionais enquanto mantém a eficácia na identificação de configurações promissoras. Com o GridSearchCV, há um refinamento preciso dessas configurações, resultando em uma sintonia fina do modelo para alcançar seu potencial máximo. Além disso, a avaliação no conjunto de teste utilizando os melhores parâmetros encontrados contribui para uma avaliação mais realista da capacidade de generalização do modelo. Em conjunto, essas estratégias oferecem uma abordagem equilibrada e eficiente para aprimorar o desempenho do XGBoost em tarefas de regressão, refletindo-se em resultados mais robustos e confiáveis.

```
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import YourPreprocessor
from sklearn.metrics import make_scorer
import numpy as np

X_train, X_eval, y_train_log, y_eval_log = train_test_split(X, y_log, random_state=42)

model_xgb = XGBRegressor(
    max_depth=10,
    n_estimators=3000,
    learning_rate=0.01,
    early_stopping_rounds=50,
    eval_metric="rmse",
    eval_set=[(X_eval, y_eval_log)],
    verbose=False
)

pipe_xgb = make_pipeline(YourPreprocessor(), model_xgb)

param_dist = {
    'xgbregressor__max_depth': [8, 10, 12],
    'xgbregressor__n_estimators': np.arange(1000, 5000, 100),
    'xgbregressor__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'xgbregressor__subsample': [0.8, 0.9, 1.0],
    'xgbregressor__colsample_bytree': [0.8, 0.9, 1.0],
    'xgbregressor__gamma': [0, 1, 5],
    'xgbregressor__min_child_weight': [1, 5, 10],
}

random_search = RandomizedSearchCV(
    estimator=pipe_xgb,
    param_distributions=param_dist,
    n_iter=10,
    scoring=make_scorer(rmse),
    cv=5,
    n_jobs=-1
)

random_search.fit(X_train, y_train_log)

print("Melhores parâmetros (Randomized Search):", random_search.best_params_)
print("Melhor pontuação RMSE (Randomized Search):", random_search.best_score_)

param_grid = {
    'xgbregressor__max_depth': [random_search.best_params_['xgbregressor__max_depth']],
    'xgbregressor__n_estimators': np.arange(2000, 4000, 200),
    'xgbregressor__learning_rate': [random_search.best_params_['xgbregressor__learning_rate']],
    'xgbregressor__subsample': [random_search.best_params_['xgbregressor__subsample']],
    'xgbregressor__colsample_bytree': [random_search.best_params_['xgbregressor__colsample_bytree']],
    'xgbregressor__gamma': [random_search.best_params_['xgbregressor__gamma']],
    'xgbregressor__min_child_weight': [random_search.best_params_['xgbregressor__min_child_weight']],
}

grid_search = GridSearchCV(
    estimator=pipe_xgb,
    param_grid=param_grid,
    scoring=make_scorer(rmse),
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train_log)
y_pred_xgb = grid_search.predict(X_eval)

score_xgb = rmse(y_eval_log, y_pred_xgb)
print("\nMelhores parâmetros (Grid Search):", grid_search.best_params_)
print("RMSE no Conjunto de Avaliação (Grid Search):", score_xgb)
```
### Explicação das Mudanças:
- Pesquisa Aleatória de Hiperparâmetros: Implementamos uma pesquisa aleatória de hiperparâmetros utilizando o RandomizedSearchCV. Essa abordagem nos permite explorar uma variedade de combinações de hiperparâmetros de forma eficiente, identificando configurações promissoras que impactam diretamente no desempenho do modelo.

- Treinamento com Early Stopping: Ao ajustar o modelo XGBoost, incorporamos a técnica de "early stopping" para interromper o treinamento quando o desempenho no conjunto de validação para de melhorar. Isso evita o sobreajuste e contribui para um modelo mais generalizado.

- Avaliação do Desempenho com Conjunto de Validação: Além da métrica de desempenho obtida durante o treinamento, realizamos uma avaliação adicional do modelo no conjunto de validação, calculando o Root Mean Squared Error (RMSE). Essa métrica proporciona uma visão abrangente da qualidade das previsões em um conjunto de dados não utilizado durante o treinamento.