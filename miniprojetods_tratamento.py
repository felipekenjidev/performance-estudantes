# ====================================================================================
# ====================================================================================
# =================== TRATAMENTO E ANÁLISES DO DATASET ===============================
# ====================================================================================
# ====================================================================================

# =================================== CÉLULA 1.2
# Importar Bibliotecas do Python
import pandas as pd
import os

# Abrir o Arquivo Dataset
df = pd.read_csv("student_data.csv")

# Exibir o DataFrame
print('--- EXIBIÇÃO DO DATASET EM FORMATO DE TABELA ---')
df.head()

# =================================== CÉLULA 1.3
# Verificar a Quantidade de Linhas por Colunas
print('--- QUANTIDADE DE LINHAS/COLUNAS ---')
print(df.shape)

# Exibir Nome de Todas as Colunas Presentes
print('\n--- NOME DAS COLUNAS ---')
print(df.columns)

# Exibir as Colunas e a Quantidade de Valores Presentes
print('\n--- COLUNAS E SEUS VALORES ---')
print(df.info())

# Contar o Número de Linhas com Valores Ausentes
print('\n--- DADOS DAS LINHAS ---')
print(f'Número de linhas do dataset: {len(df)}')
print(f'Número de linhas com pelo menos um valor ausente: {df.isna().sum().sum()}')

# =================================== CÉLULA 1.4
# Identificar Linhas Duplicadas no Dataframe
num_duplicadas = df.duplicated().sum()

# Exibir o Número de Linhas Duplicadas
print('--- EXIBIÇÃO DE LINHAS DUPLICADAS ---')
if num_duplicadas > 0:
    print('Primeiras 5 linhas duplicadas (mantendo a primeira ocorrência):')
    df[df.duplicated()].head()
else:
    print('Não há linhas duplicadas para exibir.')

# =================================== CÉLULA 1.5
# Declarar Variável das Colunas Categóricas
colunas_alvo = ['Mjob', 'Fjob', 'reason', 'guardian']

# Exibir a Frequência de Cada Categoria das Colunas Categóricas
print('--- EXIBIÇÃO DE FREQUÊNCIA DE CATEGORIAS ---')
for coluna in colunas_alvo:
  print(f'{df[coluna].value_counts(normalize=True)}\n')

# =================================== CÉLULA 1.6
print('--- ANÁLISE DE TENDÊNCIAS E PROPORÇÃO DE RESPOSTAS (TODAS AS COLUNAS) ---\n')

# Vamos percorrer todas as colunas do dataset automaticamente
for coluna in df.columns:
    print(f"==================================================")
    print(f"COLUNA: {coluna.upper()}")
    print(f"==================================================")

    # 1. Conta a quantidade absoluta de cada resposta
    contagem = df[coluna].value_counts()

    # 2. Calcula a porcentagem de cada resposta (multiplicamos por 100 e arredondamos)
    porcentagem = df[coluna].value_counts(normalize=True) * 100

    # 3. Junta as duas informações em um novo DataFrame para exibir bonitinho lado a lado
    analise_coluna = pd.DataFrame({
        'Quantidade': contagem,
        'Porcentagem (%)': porcentagem.round(2)
    })

    analise_coluna
    print("\n") # Dá um espaço entre as colunas para não virar bagunça

# =================================== CÉLULA 1.7
# Importar Bibliotecas do Python
import numpy as np
from scipy import stats

# Criar Nova Coluna de Média de Notas
df['média'] = df[['G1', 'G2', 'G3']].mean(axis=1)

# Normalização dos Dados para Uma Escala Normal (0 a 1)
dados_normalizados = (df['média'] - df['média'].min()) / (df['média'].max() - df['média'].min())

# Criação dos Dados Puramente Aleatórios
dados_aleatorios = np.random.uniform(0, 1, len(df))

# Comparar Ambos os Valores
stat, p_value = stats.ks_2samp(dados_normalizados, dados_aleatorios)
alpha = 0.05

# Exibir os Resultados
print('--- VERIFICAÇÃO DE HIPÓTESE NULA ---')
print(f'O resultado do p-value foi de aproximadamente: {p_value:.6f}')

if p_value < alpha:
  print("De acordo com o valor alpha definido, REJEITAMOS a hipotése nula de que todos os dados são puramente aleatórios.")
else:
  print("De acordo com o valor alpha definido, NÃO SABEMOS se a hipotése nula realmente existe e se todos os dados são puramente aleatórios.")

# =================================== CÉLULA 1.8
df.describe()

print("ANÁLISE DE ZEROS")

colunas_zeros = ['absences', 'failures', 'G1', 'G2', 'G3']

resultado_zeros = []

for coluna in colunas_zeros:
    qtd_zeros = (df[coluna] == 0).sum()
    percentual = (qtd_zeros / len(df)) * 100

    resultado_zeros.append({
        'Variável': coluna,
        'Quantidade de zeros': qtd_zeros,
        'Percentual (%)': round(percentual, 2)
    })

resultado_zeros = pd.DataFrame(resultado_zeros)

print(resultado_zeros.to_string(index=False))

zeros_g3 = (df['G3'] == 0).sum()

print(f'Quantidade de alunos com G3 = 0: {zeros_g3}')
print(f'Percentual: {zeros_g3/len(df)*100:.2f}%')

g3_zero = df[df['G3'] == 0]

print(g3_zero[['G1','G2','G3','studytime','failures','absences']].describe())

g3_nao_zero = df[df['G3'] > 0]

comparacao = pd.DataFrame({
    'G3=0': g3_zero[['studytime','failures','absences', 'G1', 'G2']].mean(),
    'G3>0': g3_nao_zero[['studytime','failures','absences', 'G1', 'G2']].mean()
})

print(comparacao)

# =================================== CÉLULA 1.9
# Para cada coluna, serão exibidas as categorias existentes

print('Valores únicos por coluna categórica:')
for col in df.select_dtypes(include='object').columns.tolist() + df.select_dtypes(include='string').columns.tolist():
    print(f'  {col}: {list(df[col].unique())}')

#Encoding de todas as colunas binárias: school, sex, address, famsize, Pstatus, schoolsup, famsup, paid, activities, nursery, higher,  internet, romantic

mappings = {
    'school':     {'GP': 0, 'MS': 1},
    'sex':        {'F': 0, 'M': 1},
    'address':    {'U': 0, 'R': 1},
    'famsize':    {'LE3': 0, 'GT3': 1},
    'Pstatus':    {'T': 0, 'A': 1},
    'schoolsup':  {'no': 0, 'yes': 1},
    'famsup':     {'no': 0, 'yes': 1},
    'paid':       {'no': 0, 'yes': 1},
    'activities': {'no': 0, 'yes': 1},
    'nursery':    {'no': 0, 'yes': 1},
    'higher':     {'no': 0, 'yes': 1},
    'internet':   {'no': 0, 'yes': 1},
    'romantic':   {'no': 0, 'yes': 1}
}

df_encoded = df.copy()

for col, mapping in mappings.items():
    df_encoded[col] = df_encoded[col].map(mapping)

for col in mappings.keys():
    print(f"\n{col}:")
    print(df_encoded[col].value_counts())

#Encoding de todas as colunas com + de duas categorias

df_encoded = pd.get_dummies(
    df_encoded,
    columns=['Mjob', 'Fjob', 'reason', 'guardian'],
    dtype=int)

ohe_cols = [
    c for c in df_encoded.columns
    if c.startswith('Mjob_')
    or c.startswith('Fjob_')
    or c.startswith('reason_')
    or c.startswith('guardian_')]

for col in ohe_cols:
    print(f"\n{col}:")
    print(df_encoded[col].value_counts())

# =================================== CÉLULA 1.10
import pandas as pd
import numpy as np

print('==================================================================')
print('TRIAGEM: RANKING OFICIAL DE CORRELAÇÕES COM A NOTA')
print('==================================================================\n')

# 1. Garantir que a G_Media está criada
if 'G_Media' not in df_encoded.columns:
    df_encoded['G_Media'] = (df_encoded['G1'] + df_encoded['G2'] + df_encoded['G3']) / 3

# 2. Calcular a matriz de correlação (apenas para colunas numéricas/encodadas)
# Nota: Já ignora as colunas que decidiste excluir (higher, Pstatus, school)
df_analise = df_encoded.drop(columns=['higher', 'Pstatus', 'school'], errors='ignore')
matriz_corr = df_analise.corr(numeric_only=True)

# 3. Isolar a correlação com a G_Media e remover as próprias notas do ranking
ranking = matriz_corr['G_Media'].drop(['G_Media', 'G1', 'G2', 'G3'], errors='ignore')

# 4. Criar um DataFrame para organizar o ranking por força de impacto (Valor Absoluto)
df_ranking = pd.DataFrame({
    'Correlação Real': ranking,
    'Força do Impacto (Absoluto)': ranking.abs()
}).sort_values(by='Força do Impacto (Absoluto)', ascending=False)

# 5. Exibir o Ranking Estruturado para o Grupo
print(f"{'Posição':<7} | {'Variável':<15} | {'Correlação':<11} | {'Tipo de Impacto'}")
print('-' * 65)

posicao = 1
for variavel, linhas in df_ranking.iterrows():
    corr_real = linhas['Correlação Real']

    # Classificação do impacto para orientar os teus próximos gráficos
    if corr_real > 0.05:
        sinal = "🟢 POSITIVO (Alavanca a nota)"
    elif corr_real < -0.05:
        sinal = "🔴 NEGATIVO (Prejudica a nota)"
    else:
        sinal = "⚪ ZERO (Sem relação linear)"

    print(f"#{posicao:<5} | {variavel:<15} | {corr_real:+11.4f} | {sinal}")
    posicao += 1

print('\n📌 Como usar este resultado para as próximas análises:')
print('1. As variáveis no topo (#1, #2, #3...) são as mais importantes do projeto.')
print('2. Variáveis com sinal 🟢 devem ser cruzadas para mostrar o que ajuda o aluno (ex: studytime).')
print('3. Variáveis com sinal 🔴 devem ser cruzadas para mostrar o que destrói a nota (ex: failures).')
print('4. Variáveis com sinal ⚪ (perto de 0.00) são as que podes sugerir excluir da modelagem por falta de relação.')

# =================================== CÉLULA 2.1
print('TRIAGEM: RANKING DE CORRELAÇÕES DE FAILURE')

# Calcular a matriz de correlação (apenas para colunas numéricas/encodadas)
# Nota: Já ignora as colunas que decidiste excluir (higher, Pstatus, school)
df_analise = df_encoded.drop(columns=['higher', 'Pstatus', 'school'], errors='ignore')
matriz_corr = df_analise.corr(numeric_only=True)

# 3. Isolar a correlação com a G_Media e remover as próprias notas do ranking
ranking = matriz_corr['failures'].drop(['failures', 'G_Media', 'G1', 'G2', 'G3'], errors='ignore')

# 4. Criar um DataFrame para organizar o ranking por força de impacto (Valor Absoluto)
df_ranking = pd.DataFrame({
    'Correlação Real': ranking,
    'Força do Impacto (Absoluto)': ranking.abs()
}).sort_values(by='Força do Impacto (Absoluto)', ascending=False)

# 5. Exibir o Ranking Estruturado para o Grupo
print(f"{'Posição':<7} | {'Variável':<15} | {'Correlação':<11} | {'Tipo de Impacto'}")
print('-' * 65)

posicao = 1
for variavel, linhas in df_ranking.iterrows():
    corr_real = linhas['Correlação Real']

    # Classificação do impacto para orientar os teus próximos gráficos
    if corr_real > 0.05:
        sinal = "🟢 POSITIVO"
    elif corr_real < -0.05:
        sinal = "🔴 NEGATIVO"
    else:
        sinal = "⚪ ZERO"

    print(f"#{posicao:<5} | {variavel:<15} | {corr_real:+11.4f} | {sinal}")
    posicao += 1

print('TRIAGEM: RANKING DE CORRELAÇÕES DE GOOUT')


# Calcular a matriz de correlação (apenas para colunas numéricas/encodadas)
# Nota: Já ignora as colunas que decidiste excluir (higher, Pstatus, school)
df_analise = df_encoded.drop(columns=['higher', 'Pstatus', 'school'], errors='ignore')
matriz_corr = df_analise.corr(numeric_only=True)

# 3. Isolar a correlação com a G_Media e remover as próprias notas do ranking
ranking = matriz_corr['goout'].drop(['goout', 'G_Media', 'G1', 'G2', 'G3'], errors='ignore')

# 4. Criar um DataFrame para organizar o ranking por força de impacto (Valor Absoluto)
df_ranking = pd.DataFrame({
    'Correlação Real': ranking,
    'Força do Impacto (Absoluto)': ranking.abs()
}).sort_values(by='Força do Impacto (Absoluto)', ascending=False)

# 5. Exibir o Ranking Estruturado para o Grupo
print(f"{'Posição':<7} | {'Variável':<15} | {'Correlação':<11} | {'Tipo de Impacto'}")
print('-' * 65)

posicao = 1
for variavel, linhas in df_ranking.iterrows():
    corr_real = linhas['Correlação Real']

    # Classificação do impacto para orientar os teus próximos gráficos
    if corr_real > 0.05:
        sinal = "🟢 POSITIVO (Alavanca a nota)"
    elif corr_real < -0.05:
        sinal = "🔴 NEGATIVO (Prejudica a nota)"
    else:
        sinal = "⚪ ZERO (Sem relação linear)"

    print(f"#{posicao:<5} | {variavel:<15} | {corr_real:+11.4f} | {sinal}")
    posicao += 1

print('TRIAGEM: RANKING DE CORRELAÇÕES DE SCHOOLSUP')


# Calcular a matriz de correlação (apenas para colunas numéricas/encodadas)
# Nota: Já ignora as colunas que decidiste excluir (higher, Pstatus, school)
df_analise = df_encoded.drop(columns=['higher', 'Pstatus', 'school'], errors='ignore')
matriz_corr = df_analise.corr(numeric_only=True)

# 3. Isolar a correlação com a G_Media e remover as próprias notas do ranking
ranking = matriz_corr['schoolsup'].drop(['schoolsup', 'G_Media', 'G1', 'G2', 'G3'], errors='ignore')

# 4. Criar um DataFrame para organizar o ranking por força de impacto (Valor Absoluto)
df_ranking = pd.DataFrame({
    'Correlação Real': ranking,
    'Força do Impacto (Absoluto)': ranking.abs()
}).sort_values(by='Força do Impacto (Absoluto)', ascending=False)

# 5. Exibir o Ranking Estruturado para o Grupo
print(f"{'Posição':<7} | {'Variável':<15} | {'Correlação':<11} | {'Tipo de Impacto'}")
print('-' * 65)

posicao = 1
for variavel, linhas in df_ranking.iterrows():
    corr_real = linhas['Correlação Real']

    # Classificação do impacto para orientar os teus próximos gráficos
    if corr_real > 0.05:
        sinal = "🟢 POSITIVO (Alavanca a nota)"
    elif corr_real < -0.05:
        sinal = "🔴 NEGATIVO (Prejudica a nota)"
    else:
        sinal = "⚪ ZERO (Sem relação linear)"

    print(f"#{posicao:<5} | {variavel:<15} | {corr_real:+11.4f} | {sinal}")
    posicao += 1

print('TRIAGEM: RANKING DE CORRELAÇÕES DE STUDY TIME')


# Calcular a matriz de correlação (apenas para colunas numéricas/encodadas)
# Nota: Já ignora as colunas que decidiste excluir (higher, Pstatus, school)
df_analise = df_encoded.drop(columns=['higher', 'Pstatus', 'school'], errors='ignore')
matriz_corr = df_analise.corr(numeric_only=True)

# 3. Isolar a correlação com a G_Media e remover as próprias notas do ranking
ranking = matriz_corr['studytime'].drop(['studytime', 'G_Media', 'G1', 'G2', 'G3'], errors='ignore')

# 4. Criar um DataFrame para organizar o ranking por força de impacto (Valor Absoluto)
df_ranking = pd.DataFrame({
    'Correlação Real': ranking,
    'Força do Impacto (Absoluto)': ranking.abs()
}).sort_values(by='Força do Impacto (Absoluto)', ascending=False)

# 5. Exibir o Ranking Estruturado para o Grupo
print(f"{'Posição':<7} | {'Variável':<15} | {'Correlação':<11} | {'Tipo de Impacto'}")
print('-' * 65)

posicao = 1
for variavel, linhas in df_ranking.iterrows():
    corr_real = linhas['Correlação Real']

    # Classificação do impacto para orientar os teus próximos gráficos
    if corr_real > 0.05:
        sinal = "🟢 POSITIVO (Alavanca a nota)"
    elif corr_real < -0.05:
        sinal = "🔴 NEGATIVO (Prejudica a nota)"
    else:
        sinal = "⚪ ZERO (Sem relação linear)"

    print(f"#{posicao:<5} | {variavel:<15} | {corr_real:+11.4f} | {sinal}")
    posicao += 1

# =================================== CÉLULA 2.2
import matplotlib.pyplot as plt
import seaborn as sns

df['G_mean'] = df[['G1', 'G2', 'G3']].mean(axis=1)

df_pivot_mean = df.groupby(['freetime', 'goout'])['G_mean'].mean().unstack()
df_pivot_count = df.groupby(['freetime', 'goout'])['G_mean'].count().unstack()

MIN_N = 5
mask_baixa_amostra = df_pivot_count < MIN_N

labels = df_pivot_mean.round(1).astype(str) + '\n(n=' + df_pivot_count.astype(str) + ')'

fig, ax = plt.subplots(figsize=(8, 6))

# Camada 1: células com amostra suficiente (coloridas normalmente)
sns.heatmap(
    df_pivot_mean,
    ax=ax,
    cmap='RdYlGn',
    annot=labels,
    fmt='',
    linewidths=0.5,
    cbar_kws={'label': 'Média de notas'},
    mask=mask_baixa_amostra,
    annot_kws={'fontsize': 9}
)

# Camada 2: células com poucos alunos (cinza, pouco confiáveis)
sns.heatmap(
    df_pivot_mean,
    ax=ax,
    cmap=['lightgray'],
    cbar=False,
    mask=~mask_baixa_amostra,
    annot=labels,
    fmt='',
    linewidths=0.5,
    annot_kws={'fontsize': 9, 'color': 'gray'}
)

ax.set_title(f'Média de notas (e tamanho da amostra)\nSair com amigos vs Tempo livre\nCélulas em cinza: n < {MIN_N}')
ax.set_xlabel('Sair com amigos (goout)')
ax.set_ylabel('Tempo livre (freetime)')

plt.tight_layout()
 

# =================================== CÉLULA 2.3
# Importar Biblioteca de Python
import matplotlib.pyplot as plt

# Calcular as Médias das Notas de Cada Prova
medias = df_encoded[['G1', 'G2', 'G3']].mean()
cores = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Exibir Gráfico de Barras
plt.figure(figsize=(6, 5))
plt.bar(medias.index, medias.values, color=cores, width=0.5, edgecolor='black', alpha=0.8)

plt.title('Média Geral de Notas por Prova', fontsize=16, fontweight='bold')
plt.xlabel('Provas', fontsize=12, fontweight='bold')
plt.ylabel('Média das Notas', fontsize=12, fontweight='bold')

for i, valor in enumerate(medias.values):
  plt.text(i, valor + 0.2, f'{valor:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Dar um pequeno espaço no topo do gráfico para o texto não cortar
plt.ylim(0, medias.max() + 2)

# Adicionar linhas de grade apenas no eixo Y
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Exibir o resultado
 

# Definir o Intervalo de Notas Possíveis
nota_minima = 0
nota_maxima = int(df_encoded[['G1', 'G2', 'G3']].max().max())
range_notas = np.arange(nota_minima, nota_maxima + 1)

# Contar a Quantidade de Notas para Cada Prova
frequencia_g1 = df_encoded['G1'].value_counts().reindex(range_notas, fill_value=0)
frequencia_g2 = df_encoded['G2'].value_counts().reindex(range_notas, fill_value=0)
frequencia_g3 = df_encoded['G3'].value_counts().reindex(range_notas, fill_value=0)

# Exibir Gráfico de Frequência
print('\n--- EXIBIÇÃO DO GRÁFICO DE FREQUÊNCIA DE NOTAS ---')
plt.figure(figsize=(17, 6))
plt.plot(range_notas, frequencia_g1, marker='o', label='G1 (Prova 1)', color='#1f77b4', linewidth=2)
plt.plot(range_notas, frequencia_g2, marker='o', label='G2 (Prova 2)', color='#ff7f0e', linewidth=2)
plt.plot(range_notas, frequencia_g3, marker='o', label='G3 (Prova 3)', color='#2ca02c', linewidth=2)

plt.title('Distribuição de Frequência das Notas (G1, G2 e G3)', fontsize=16, fontweight='bold')
plt.xlabel('Notas das Provas', fontsize=12, fontweight='bold')
plt.ylabel('Quantidade de Estudantes', fontsize=12, fontweight='bold')
plt.xticks(range_notas)

plt.legend(fontsize=11, title="Provas")
plt.grid(True, linestyle='--', alpha=0.5)
 

# Importar Biblioteca do Python
import seaborn as sns

# Exibir Matriz de PairPlot
pp = sns.pairplot(df_encoded, vars=['G1', 'G2', 'G3'], diag_kind='kde', plot_kws={'alpha': 0.2})
pp.fig.suptitle('Matriz de Correlação e Distribuição: G1, G2 e G3', y=1.05, fontsize=16, fontweight='bold')
 

# Importar Bibliotecas do Python
import plotly.graph_objects as go

# Função para Calcular as Frequências das Notas de Cada Prova
def calcular_frequencias(dataframe_filtrado):
  g1_freq = dataframe_filtrado['G1'].value_counts().reindex(range_notas, fill_value=0).values
  g2_freq = dataframe_filtrado['G2'].value_counts().reindex(range_notas, fill_value=0).values
  g3_freq = dataframe_filtrado['G3'].value_counts().reindex(range_notas, fill_value=0).values
  return g1_freq, g2_freq, g3_freq

# Colocar as Informações Iniciais do Gráfico
fig = go.Figure()
f_g1, f_g2, f_g3 = calcular_frequencias(df_encoded)
fig.add_trace(go.Scatter(x=range_notas, y=f_g1, mode='lines+markers', name='G1', line=dict(color='#1f77b4', width=3)))
fig.add_trace(go.Scatter(x=range_notas, y=f_g2, mode='lines+markers', name='G2', line=dict(color='#ff7f0e', width=3)))
fig.add_trace(go.Scatter(x=range_notas, y=f_g3, mode='lines+markers', name='G3', line=dict(color='#2ca02c', width=3)))

# Criar os Eventos do Slider Interativo
steps = []
for nota_limite in range_notas:
  # Filtrar o Dataframe e Recalcular a Frequência de Notas
  df_filtrado = df_encoded[df_encoded['G1'] <= nota_limite]
  freq_g1, freq_g2, freq_g3 = calcular_frequencias(df_filtrado)

  # Atualizar as Informações do Gráfico
  step = dict(
    method='update',
    label=str(nota_limite),
    args=[{'y': [freq_g1, freq_g2, freq_g3]},
      {'title': f'<b>Distribuição de Frequência de Notas para Alunos com Nota G1 <= {nota_limite} ({len(df_filtrado)} alunos) </b>'}],
  )
  steps.append(step)

# Exibir o Gráfico Interativo
sliders = [dict(
    active=len(range_notas) - 1,
    currentvalue={'prefix': 'Filtrar alunos com nota em G1 menor ou igual a: ', 'font': {'size': 14, 'color': 'black'}},
    pad={'t': 60},
    steps=steps
)]

fig.update_layout(
  sliders=sliders,
  title=dict(
    text=f'<b>Distribuição de Frequência de Notas para Alunos com Nota G1 <= {nota_limite} ({len(df_filtrado)} alunos) </b>',
    font=dict(
        family='Arial, sans-serif',
        size=18,
        color='black'
    ),
    x=0.5,
    y=0.95
  ),
  xaxis=dict(
    title=dict(
      text='<b>Notas das Provas</b>',
      font=dict(
        family='Arial, sans-serif',
        size=16,
        color='black'
      ),
    ),
    tickmode='linear',
    tick0=0,
    dtick=1
  ),
  yaxis=dict(
    title=dict(
      text='<b>Quantidade de Estudantes</b>',
      font=dict(
        family='Arial, sans-serif',
        size=16,
        color='black'
      ),
    ),
    hoverformat='d'
  ),
  hovermode='x unified',
  legend_title='Provas'
)

# Colocar as Informações Iniciais do Gráfico
fig = go.Figure()
f_g1, f_g2, f_g3 = calcular_frequencias(df_encoded[df_encoded['G1'] == nota_maxima])
fig.add_trace(go.Scatter(x=range_notas, y=f_g1, mode='lines+markers', name='G1', line=dict(color='#1f77b4', width=3)))
fig.add_trace(go.Scatter(x=range_notas, y=f_g2, mode='lines+markers', name='G2', line=dict(color='#ff7f0e', width=3)))
fig.add_trace(go.Scatter(x=range_notas, y=f_g3, mode='lines+markers', name='G3', line=dict(color='#2ca02c', width=3)))

# Criar os Eventos do Slider Interativo
steps = []
for nota_limite in range_notas:
  # Filtrar o Dataframe e Recalcular a Frequência de Notas
  df_filtrado = df_encoded[df_encoded['G1'] == nota_limite]
  freq_g1, freq_g2, freq_g3 = calcular_frequencias(df_filtrado)

  # Atualizar as Informações do Gráfico
  step = dict(
    method='update',
    label=str(nota_limite),
    args=[{'y': [freq_g1, freq_g2, freq_g3]},
      {'title': f'<b>Distribuição de Frequência de Notas para Alunos com Nota G1 = {nota_limite} ({len(df_filtrado)} alunos) </b>'}],
  )
  steps.append(step)

# Exibir o Gráfico Interativo
sliders = [dict(
    active=len(range_notas) - 1,
    currentvalue={'prefix': 'Filtrar alunos com nota em G1 igual a: ', 'font': {'size': 14, 'color': 'black'}},
    pad={'t': 60},
    steps=steps
)]

fig.update_layout(
  sliders=sliders,
  title=dict(
    text=f'<b>Distribuição de Frequência de Notas para Alunos com Nota G1 <= {nota_limite} ({len(df_filtrado)} alunos) </b>',
    font=dict(
        family='Arial, sans-serif',
        size=18,
        color='black'
    ),
    x=0.5,
    y=0.95
  ),
  xaxis=dict(
    title=dict(
      text='<b>Notas das Provas</b>',
      font=dict(
        family='Arial, sans-serif',
        size=16,
        color='black'
      ),
    ),
    tickmode='linear',
    tick0=0,
    dtick=1
  ),
  yaxis=dict(
    title=dict(
      text='<b>Quantidade de Estudantes</b>',
      font=dict(
        family='Arial, sans-serif',
        size=16,
        color='black'
      ),
    ),
    hoverformat='d'
  ),
  hovermode='x unified',
  legend_title='Provas'
)

# =================================== CÉLULA 2.4
# Gerar Gráfico de Média de Notas por Profissão
print('\n--- EXIBIÇÃO DO GRÁFICO DE MÉDIA DE NOTAS POR PROFISSÃO DOS PAIS ---')
df['average'] = df[['G1', 'G2', 'G3']].mean(axis=1)
plt.figure(figsize=(8, 5))

sns.barplot(data=df, x='Mjob', y='average', hue='Fjob', palette='mako')

plt.legend(title='Profissão do Pai', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Gráfico de Média de Notas x Profissão dos Pais', fontsize=16, fontweight='bold')
plt.xlabel('Profissão da Mãe', fontsize=12, fontweight='bold')
plt.ylabel('Média de Notas', fontsize=12, fontweight='bold')
 

# Gerar Gráfico de Média de Notas por Profissão (Guardião da Família MÃE)
print('\n--- EXIBIÇÃO DO GRÁFICO DE MÉDIA DE NOTAS POR PROFISSÃO DOS PAIS COM MÃE GUARDIÃ ---')
df['average'] = df[['G1', 'G2', 'G3']].mean(axis=1)
plt.figure(figsize=(8, 5))

sns.barplot(data=df[df['guardian']=='mother'], x='Mjob', y='average', hue='Fjob', palette='mako')

plt.legend(title='Profissão do Pai', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Gráfico de Média de Notas x Profissão dos Pais (MÃE GUARDIÃ)', fontsize=16, fontweight='bold')
plt.xlabel('Profissão da Mãe', fontsize=12, fontweight='bold')
plt.ylabel('Média de Notas', fontsize=12, fontweight='bold')
 

# Gerar Gráfico de Média de Notas por Profissão (Guardião da Família PAI)
print('\n--- EXIBIÇÃO DO GRÁFICO DE MÉDIA DE NOTAS POR PROFISSÃO DOS PAIS COM PAI GUARDIÃO ---')
df['average'] = df[['G1', 'G2', 'G3']].mean(axis=1)
plt.figure(figsize=(8, 5))

sns.barplot(data=df[df['guardian']=='father'], x='Mjob', y='average', hue='Fjob', palette='mako')

plt.legend(title='Profissão do Pai', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Gráfico de Média de Notas x Profissão dos Pais (PAI GUARDIÃO)', fontsize=16, fontweight='bold')
plt.xlabel('Profissão da Mãe', fontsize=12, fontweight='bold')
plt.ylabel('Média de Notas', fontsize=12, fontweight='bold')
 

# Contar Linhas por Guardião
print('\n--- CONTAGEM DE LINHAS POR GUARDIÃO ---')
print(f'Quantidade de linhas com Mãe Guardiã: {(df['guardian'] == 'mother').sum()}')
print(f'Quantidade de linhas com Pai Guardião: {(df['guardian'] == 'father').sum()}')

# Gerar Matriz de Correlação de Nota por Educação dos Pais
print('\n--- EXIBIÇÃO DA MATRIZ DE CORRELAÇÃO MÉDIA x EDUCAÇÃO ---')
df_correlacao = df[['Medu', 'Fedu', 'average']]
plt.figure(figsize=(5, 5))

sns.heatmap(df_correlacao.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)

plt.title('Matriz de Correlação Média x Escolaridade dos Pais', fontsize=16, fontweight='bold')
 

# Gerar Gráfico de Média de Notas por Educação dos Pais
print('\n--- EXIBIÇÃO DO GRÁFICO DE MÉDIA DE NOTAS POR ESCOLARIDADE DOS PAIS ---')
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Medu', y='average', hue='Fedu', palette='rocket')

plt.legend(title='Educação do Pai', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Gráfico de Média de Notas x Educação dos Pais', fontsize=16, fontweight='bold')
plt.xlabel('Educação da Mãe', fontsize=12, fontweight='bold')
plt.ylabel('Média de Notas', fontsize=12, fontweight='bold')
 

# Contar Linhas por Guardião
print('\n--- CONTAGEM DE LINHAS POR GUARDIÃO ---')
print(f'Quantidade de Linhas por Medu: { df['Medu'].value_counts().sort_index() }')
print(f'Quantidade de Linhas por Fedu: { df['Fedu'].value_counts().sort_index() }')

# Gerar Gráfico de Média de Notas por Relação dos Pais
print('\n--- EXIBIÇÃO DO GRÁFICO DE MÉDIA DE NOTAS POR RELAÇÃO FAMILIAR E STATUS DOS PAIS ---')
plt.figure(figsize=(9, 6))

sns.barplot(data=df, x='famrel', y='average', hue='Pstatus', palette='pastel')

plt.legend(title='Status dos Pais', labels=['Juntos (T)', 'Separados (A)'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Gráfico de Média de Notas x Relação Familiar x Status dos Pais', fontsize=14, fontweight='bold')
plt.xlabel('Qualidade da Relação Familiar', fontsize=12, fontweight='bold')
plt.ylabel('Média de Notas', fontsize=12, fontweight='bold')
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['Péssima', 'Ruim', 'Regular', 'Boa', 'Excelente'])
 

# Contar Linhas por Guardião
print('\n--- CONTAGEM DE LINHAS POR RELAÇÃO FAMILIAR E STATUS DOS PAIS ---')
print(f'Quantidade de Linhas por Qualidade da Relação (famrel): { df['famrel'].value_counts().sort_index() }')
print(f'\nQuantidade de Linhas por Status de União dos Pais (Pstatus): { df['Pstatus'].value_counts() }')

# Exibir o Resumo das Médias de Notas para Cada Variável
print("--- RESUMO DAS MÉDIAS DAS VARIÁVEIS ---")

# Criar a Coluna de Média de Notas
if 'G_Media' not in df.columns:
    df['G_Media'] = df[['G1', 'G2', 'G3']].mean(axis=1)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Exibir Profissão dos Pais vs Média de Notas
df_jobs = df.groupby(['Mjob', 'Fjob'])['G_Media'].mean().unstack()
sns.heatmap(df_jobs, annot=True, fmt=".2f", cmap='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Média de Notas pelas Profissões da Mãe e do Pai')

# Exibir Escolaridade dos Pais vs Média
df_edu_notes = df.groupby(['Medu', 'Fedu'])['G_Media'].mean().unstack()
sns.heatmap(df_edu_notes, annot=True, fmt=".2f", cmap='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Média de Notas pelo Nível de Instrução dos Pais')

# Exibir Pstatus vs Famrel vs G_Media
sns.boxenplot(data=df, x='famrel', y='G_Media', hue='Pstatus', ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('Qualidade da Relação Familiar (famrel) vs Nota Média por Pstatus')

plt.tight_layout()
 

# =================================== CÉLULA 2.5
# Gerar Gráfico de Média de Notas por Profissão
print('\n--- EXIBIÇÃO DO GRÁFICO DE MÉDIA DE NOTAS POR GEOLOCALIZAÇÃO ---')
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='address', y='average', hue='internet', palette='plasma')

plt.legend(title='Possui internet?', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Gráfico de Média de Notas x Geolocalização', fontsize=16, fontweight='bold')
plt.xlabel('Espaço Urbano x Rural', fontsize=12, fontweight='bold')
plt.ylabel('Média de Notas', fontsize=12, fontweight='bold')
 

# =================================== CÉLULA 2.6
# Exibir Gráfico de Apoio Educacional x Média de Notas
print("--- GRÁFICO DE APOIO EDUCACIONAL POR MÉDIA DE NOTAS ---")

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='schoolsup', y='G_Media', hue='paid', palette='viridis', errorbar=None)
plt.title('Comparação de Desempenho: Apoio da Escola (Gratuito) vs. Aulas Particulares (Pagas)')
plt.ylabel('Média das Notas (G1, G2, G3)')
 

# Importar Bibliotecas do Python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Gerar Estatísticas sobre as Notas Finais
print("="*60)
print("DISTRIBUIÇÃO DOS ALUNOS")
print("="*60)

print(df['paid'].value_counts())

print("\n" + "="*60)
print("ESTATÍSTICAS DAS NOTAS FINAIS (G3)")
print("="*60)

estatisticas = df.groupby('paid')['G3'].describe()
print(estatisticas)

print("\n" + "="*60)
print("MÉDIAS DAS NOTAS")
print("="*60)

medias = df.groupby('paid')[['G1','G2','G3']].agg(['mean','median','count'])
print(medias)

grupo_paid = df[df['paid'] == 'yes']['G3']
grupo_nao_paid = df[df['paid'] == 'no']['G3']

t_stat, p_value = ttest_ind(grupo_paid, grupo_nao_paid)

print("\n" + "="*60)
print("TESTE T DE STUDENT")
print("="*60)

print(f"T-statistic = {t_stat:.4f}")
print(f"P-value = {p_value:.6f}")

if p_value < 0.05:
    print("\nResultado: Rejeita-se H0")
    print("Existe diferença estatisticamente significativa entre os grupos.")
else:
    print("\nResultado: Não se rejeita H0")
    print("Não há evidências suficientes de diferença significativa.")

# Exibir Gráfico das Notas Finais em relação a Suportes Pagos
plt.figure(figsize=(6,4))
df.groupby('paid')['G3'].mean().plot(kind='bar')

plt.title('Média das Notas Finais (G3)')
plt.xlabel('Aulas Pagas Extras')
plt.ylabel('Média da Nota')
plt.xticks(rotation=0)

 

plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='paid', y='G3')

plt.title('Distribuição das Notas Finais (G3)')
plt.xlabel('Aulas Pagas Extras')
plt.ylabel('Nota Final')

 

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='G3', hue='paid', kde=True, bins=10)

plt.title('Distribuição das Notas Finais por Grupo')
plt.xlabel('Nota Final (G3)')
plt.ylabel('Frequência')

 

media_no = df[df['paid']=='no']['G3'].mean()
media_yes = df[df['paid']=='yes']['G3'].mean()

print("CONCLUSÃO")

print(
    f"Alunos com suporte pago obtiveram média de {media_yes:.2f} pontos, "
    f"enquanto alunos sem suporte pago obtiveram média de {media_no:.2f} pontos."
)

print(
    f"A diferença observada foi de {media_yes - media_no:.2f} ponto(s)."
)

if p_value < 0.05:
    print(
        f"O p-value ({p_value:.4f}) indica que a diferença é estatisticamente significativa."
    )
else:
    print(
        f"O p-value ({p_value:.4f}) indica que a diferença não é estatisticamente significativa."
    )

# =================================== CÉLULA 2.7
# Gerar Resultados sobre as Personas
print("==================================================")
print("GERANDO PERSONAS DO PROJETO (ALTO VS BAIXO RENDIMENTO)")
print("==================================================")

alunos_alto = df[df['G_Media'] >= 15]
alunos_baixo = df[df['G_Media'] < 10]

def extrair_perfil(dataframe, nome_perfil):
    perfil = {
        'Tamanho do Grupo': len(dataframe),
        'Tempo de Estudo (Média)': round(dataframe['studytime'].mean(), 2),
        'Histórico de Reprovações (Média)': round(dataframe['failures'].mean(), 2),
        'Falta às Aulas (Média)': round(dataframe['absences'].mean(), 2),
        'Tempo Livre (Moda)': dataframe['freetime'].mode()[0],
        'Sair com Amigos (Moda)': dataframe['goout'].mode()[0],
        'Profissão da Mãe (Moda)': dataframe['Mjob'].mode()[0],
        'Profissão do Pai (Moda)': dataframe['Fjob'].mode()[0],
        'Acesso à Internet (%)': f"{round((dataframe['internet'] == 'yes').mean() * 100, 2)}%"
    }
    return pd.DataFrame([perfil], index=[nome_perfil]).T

persona_alto = extrair_perfil(alunos_alto, 'Persona: Alto Rendimento (Nota >= 15)')
persona_baixo = extrair_perfil(alunos_baixo, 'Persona: Baixo Rendimento (Nota < 10)')

quadro_comparativo = pd.concat([persona_alto, persona_baixo], axis=1)
print(quadro_comparativo.to_string())

# =================================== CÉLULA 2.8
# Analisar Estudantes que possuem Notas Diferentes apesar de Contextos Semelhantes
print("==================================================")
print("ANALISANDO OUTLIERS: ALUNOS RESILIENTES")
print("==================================================")

# Contexto desfavorável: Baixa escolaridade dos pais (Medu<=1 e Fedu<=1) OU Relação familiar ruim (famrel<=2) OU sem internet
contexto_desfavoravel = df[(df['Medu'] <= 2) & (df['Fedu'] <= 2) & ((df['internet'] == 'no') | (df['famrel'] <= 2))]
alunos_resilientes = contexto_desfavoravel[contexto_desfavoravel['G_Media'] >= 12]

print(f"Quantidade de alunos encontrados em vulnerabilidade: {len(contexto_desfavoravel)}")
print(f"Quantidade de Alunos Resilientes (Nota Média >= 12 nesse grupo): {len(alunos_resilientes)}")

if len(alunos_resilientes) > 0:
    print("\nO que eles têm em comum em comparação ao grupo geral vulnerável?")
    caracteristicas_resilientes = pd.DataFrame({
        'Resilientes (Média)': alunos_resilientes[['studytime', 'failures', 'absences', 'goout', 'freetime']].mean(),
        'Geral Vulnerável (Média)': contexto_desfavoravel[['studytime', 'failures', 'absences', 'goout', 'freetime']].mean()
    })
    print(caracteristicas_resilientes.round(2))

    # Gráfico explicativo sobre os Resilientes
    plt.figure(figsize=(8, 5))
    sns.barplot(x=['Vulneráveis em Geral', 'Resilientes'],
                y=[contexto_desfavoravel['studytime'].mean(), alunos_resilientes['studytime'].mean()],
                hue=[contexto_desfavoravel['studytime'].mean(), alunos_resilientes['studytime'].mean()], palette='viridis')
    plt.title('Fator de Superação: Tempo de Estudo dos Alunos Resilientes')
    plt.ylabel('Média de Horas de Estudo (Escala de 1 a 4)')
     
else:
    print("Nenhum aluno atendeu rigorosamente aos critérios extremos de resiliência. Considere flexibilizar os filtros.")

# =================================== CÉLULA 2.9
# Remove colunas descartadas e isola correlações numéricas significativas
plt.figure(figsize=(14, 10))
df_corr_geral = df_encoded.drop(columns=['higher', 'Pstatus', 'school'], errors='ignore')

# Selecionar apenas as top 15 variáveis de maior impacto com a nota para não poluir visualmente o mapa
top_vars = df_corr_geral.corr()['G_Media'].abs().sort_values(ascending=False).head(15).index
matriz_reduzida = df_corr_geral[top_vars].corr()

sns.heatmap(matriz_reduzida, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar=True)
plt.title('Matriz de Correlação Geral (Top 15 Variáveis de Impacto Acadêmico)')
 

# =================================== CÉLULA 3.1
# Importar Biblioteca do Python
from sklearn.preprocessing import StandardScaler

# Inicializar df_scaled como uma cópia de df_encoded
df_scaled = df_encoded.copy()

# Calcular a média e o desvio padrão de cada coluna a partir dos dados
scaler = StandardScaler()
escalas_colunas = ['age', 'absences', 'G1', 'G2', 'G3']
valores_escalados = scaler.fit_transform(df_scaled[escalas_colunas])

for i, col in enumerate(escalas_colunas):
    df_scaled[f'{col}_scaled'] = valores_escalados[:, i]

print(df_scaled[['age', 'age_scaled', 'absences', 'absences_scaled']].head())

# =================================== CÉLULA 3.2
# Importar Bibliotecas do Python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Excluímos as notas individuais G1, G2 e G3 para evitar vazamento de dados, focando nas características do aluno
X = df_encoded.drop(columns=['higher', 'Pstatus', 'school', 'G1', 'G2', 'G3', 'G_Media', 'média'], errors='ignore')
y = df_encoded['G_Media']

# =================================== CÉLULA 3.3
# Divisão de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização/Escalonamento das Variáveis Numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento do Modelo (Random Forest Regressor)
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train_scaled, y_train)

# =================================== CÉLULA 3.4
# Realizar Previsões
y_pred = modelo_rf.predict(X_test_scaled)

# Calcular Métricas de Avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Exibir Resultados
print('--- EXIBIÇÃO DAS MÉTRICAS DE AVALIAÇÃO ---')
print(f'Coeficiente de determinação (R²): {r2:.4f}')
print(f'Erro quadrático médio (MSE): {mse:.2f}')

# =================================== CÉLULA 3.5
# Verificar Importância das Variáveis no Aprendizado de Máquina
importancia = pd.Series(modelo_rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=importancia.values, y=importancia.index, hue=importancia.index, palette='viridis')
plt.title('Top 10 Variáveis Mais Importantes para o Modelo de Machine Learning')
plt.xlabel('Grau de Importância Relativa')
 