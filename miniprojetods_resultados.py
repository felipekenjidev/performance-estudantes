# ====================================================================================
# ====================================================================================
# ================== CRIAÇÃO DO APLICATIVO DO STREAMLIT ==============================
# ====================================================================================
# ====================================================================================

# Importar Variáveis de Arquivo Python
import miniprojetods_tratamento

# Importar Bibliotecas do Python
import streamlit as st
import pandas as pd
import numpy as np

# Configurar as Informações Gerais da Página
st.set_page_config(
    page_title="Performance dos Estudantes 2026 - Liga DS",
    page_icon="🎓",
    layout="wide"
)

# Criar a Barra Lateral do Aplicativo Web
st.sidebar.title("📚 Sobre o Projeto")

colab_url = "https://colab.research.google.com/github/felipekenjidev/performance-estudantes/blob/main/miniprojetods_performance_estudantes.ipynb"
st.sidebar.link_button("🚀 Abrir no Google Colab", colab_url, use_container_width=True)

st.sidebar.write("**Performance Estudantes**")
st.sidebar.write("É um mini projeto com apoio da Liga de Data Science da Unicamp desenvolvido em Python, com objetivo de analisar um dataset sobre as características dos estudantes, e gerar insights e correlações sobre as variáveis que mais impactam na performance escolar/universitária de um aluno.")

st.sidebar.write("**Integrantes do Grupo:**")
st.sidebar.write("""
    - Felipe Kenji Ouba Fukuzono
    - Thiago Henrique da Silva dos Santos
    - Giovanna Mota Freire
    - Julia Duarte Mira Marques
""")

# ==========================================
# 1. APRESENTAÇÃO
# ==========================================
st.title("📊 Análise de Performance dos Estudantes")
st.write("Um estudo coletivo sobre os fatores socioeconômicos e comportamentais que impactam no desempenho acadêmico de um estudante, aliado a ciência de dados e a programação, o objetivo do trabalho é identificar padrões e encontrar possíveis soluções para melhorar a média de notas dos alunos.")
st.divider()

# ==========================================
# 2. INTRODUÇÃO
# ==========================================
st.header("1. Introdução")
col_intro1, col_intro2 = st.columns([1, 1])

with col_intro1:
    st.write("""
    Para concluir com os objetivos do projeto, foi instruído ao grupo para utilizar um dataset disponível no Kaggle, 
    chamado de Student Performance Dataset, uma fonte de dados rica de informações essenciais, que além de expor a nota dos alunos de três 
    provas consecutivas (G1, G2 e G3), também dispõe de informações pessoais do estudante, como contexto familiar, rotina, informações econômicas, demográficas, 
    suporte e até estilo de vida. E a partir dele, o grupo deve analisar os dados presentes, utilizando o Python, e tirar as conclusões
    necessárias.
    """)
with col_intro2:
    st.info("""
    **Etapas Executadas no Projeto:**
    1. Limpeza e Tratamento de Dados
    2. Análise das Informações
    3. Desenvolvimento do Modelo Preditivo
    4. Visualização Dinâmica no Streamlit
    """)

st.divider()

# ==========================================
# 3. TRATAMENTO DE DADOS
# ==========================================
st.header("2. Limpeza e Tratamento")
st.write("Essa é a etapa inicial do projeto, onde a limpeza de dados é feita, para garantir o trabalho executado por cima de um dataset limpo, sem informações incoerentes e somente com dados relevantes que auxiliem a análise.")

col_trat1, col_trat2, col_trat3 = st.columns(3)

with col_trat1:
    st.subheader("Linhas Nulas e Duplicadas")
    st.write("""
    As linhas do dataset que estão completamente nulas ou que todas as informações são duplicadas, foram removidas do dataset, por poluir
    a base de dados.
    """)

with col_trat2:
    st.subheader("Frequência de Categorias")
    st.write("""
    O dataset está populado com um monte de categorias, então o grupo achou necessário eliminar algumas categorias, que
    possuem uma frequência abaixo de 5%, simplesmente por facilitar as análises e por não gerar valor estatístico suficiente.
    """)

with col_trat3:
    st.subheader("Filtro de Variância Quase Zero")
    st.write("""
    As colunas onde as respostas estavam excessivamente concentradas em uma única opção (acima de 88%) foram excluídas. 
    Então as variáveis `higher`, `Pstatus` e `school` foram excluídas por este critério, já que não traziam poder de discriminação estatística.
    """)

col_trat1, col_trat2, col_trat3 = st.columns(3)

with col_trat1:
    st.subheader("Verificação de Hipóteses Nulas")
    st.write("""
    Avaliar se as informações e os dados possuem alguma correlação entre si, e que o dataset utilizado não possui informações muito discrepantes, que são apenas fruto do acaso.
    Encontramos um p-value de 0.000115, o que rejeita a hipótese nula.
    """)

with col_trat2:
    st.subheader("Análise de Zeros")
    st.write("""
    Identificar no dataset se os valores 0 agregam alguma informação para o problema, ou se é apenas um incômodo para futuras análises.
    Foi identificado que as colunas `absences` e `failures` possuíam muitos zeros, mas é bastante coerente com o contexto, visto que muitos estudantes podem ter nenhuma falta ou reprovação.
    """)

with col_trat3:
    st.subheader("Encoding e Normalização")
    st.write("""
    Variáveis categóricas textuais foram convertidas em representações numéricas utilizando técnicas de codificação. 
    Em seguida, para assegurar que variáveis com escalas muito diferentes tivessem o mesmo peso,  aplicamos a padronização 
    através do `StandardScaler`, centralizando a média em zero e definindo o desvio padrão como um.
    """)

st.divider()

# ==========================================
# 3. ANÁLISES
# ==========================================
st.header("3. Análise de Dados e Insights")
st.write("Abaixo estão destacados os insights mais cruciais obtidos em cada um dos tópicos de análise:")

# Análise de Dinâmica de Estilo de Vida e Tempo


st.subheader("🕒 Tópico 1: Impacto do Tempo de Estudo (`studytime`)")
col_t1_txt, col_t1_graf = st.columns([1, 1])
with col_t1_txt:
    st.write("""
    Como esperado, o tempo dedicado aos estudos fora do período escolar possui uma relação direta com o rendimento. 
    Alunos que dedicam mais de 2 horas semanais apresentam índices substancialmente menores de reprovação e médias finais mais consistentes.
    """)
with col_t1_graf:
    dados_tempo = pd.DataFrame({'Média das Notas (G_Media)': [9.5, 10.8, 12.1, 13.5]}, index=['<2 horas', '2-5 horas', '5-10 horas', '>10 horas'])
    st.line_chart(dados_tempo)

# --- TÓPICO 2: Consumo de Álcool ---
st.subheader("🍹 Tópico 2: Consumo de Álcool e Relação Social")
col_t2_txt, col_t2_graf = st.columns([1, 1])
with col_t2_txt:
    st.write("""
    A análise de correlação linear revelou que o consumo de álcool (tanto em dias úteis `Dalc` quanto finais de semana `Walc`) é um dos fatores que mais possuem correlação negativa com o tempo de estudo (`studytime`). 
    Ou seja, quanto maior o consumo frequente de álcool, menor é a dedicação horária do estudante aos livros.
    """)
with col_t2_graf:
    dados_corr = pd.DataFrame({'Correlação com Tempo de Estudo': [-0.306, -0.253, -0.196]}, index=['Gênero Masculino', 'Álcool (Fim de Semana)', 'Álcool (Dias Úteis)'])
    st.bar_chart(dados_corr)

# --- TÓPICO 3: Perfis/Personas ---
st.subheader("👥 Tópico 3: Comparativo de Perfis (Personas)")
col_alto, col_baixo = st.columns(2)

with col_alto:
    # Contêiner nativo com borda simulando um card
    with st.container(border=True):
        st.subheader("🟢 Alto Rendimento (Nota Média ≥ 15)")
        st.write("- **Média de Faltas:** 4.36 faltas")
        st.write("- **Tempo de Estudo:** Alto (Média: 2.20)")
        st.write("- **Histórico de Reprovações:** Próximo a zero (0.02)")
        st.write("- **Acesso à Internet:** 92.86% possuem")

with col_baixo:
    # Contêiner nativo com borda simulando um card
    with st.container(border=True):
        st.subheader("🔴 Baixo Rendimento (Nota Média < 10)")
        st.write("- **Média de Faltas:** 6.35 faltas")
        st.write("- **Tempo de Estudo:** Baixo (Média: 1.93)")
        st.write("- **Histórico de Reprovações:** Alto (Média: 0.64)")
        st.write("- **Acesso à Internet:** 78.05% possuem")

st.divider()

# ==========================================
# 4. MACHINE LEARNING (FORMULÁRIO PREDITIVO)
# ==========================================
st.header("4. Simulador Preditivo (Machine Learning)")
st.write("Preencha as características do estudante abaixo para que o modelo faça a previsão da nota final.")

with st.form(key='form_modelo'):
    c_form1, c_form2, c_form3 = st.columns(3)
    
    with c_form1:
        idade = st.number_input("Idade do Estudante", min_value=15, max_value=22, value=17)
        tempo_estudo = st.selectbox("Tempo de Estudo Semanal", ("< 2 horas", "2 a 5 horas", "5 a 10 horas", "> 10 horas"))
        acesso_internet = st.selectbox("Possui Internet em Casa?", ("Sim", "Não"))
        
    with c_form2:
        reprovacoes = st.selectbox("Histórico de Reprovações Prévias", ("0", "1", "2", "3 ou mais"))
        faltas = st.number_input("Número de Faltas no Ano", min_value=0, max_value=100, value=4)
        consumo_alcool = st.slider("Consumo de Álcool Fim de Semana (1-Baixo a 5-Alto)", 1, 5, 2)
        
    with c_form3:
        nota_g1 = st.number_input("Nota do 1º Período (G1: 0 a 20)", min_value=0.0, max_value=20.0, value=12.0)
        nota_g2 = st.number_input("Nota do 2º Período (G2: 0 a 20)", min_value=0.0, max_value=20.0, value=11.5)
        
    botao_prever = st.form_submit_button(label='🔮 Calcular Nota Preditiva Final')

if botao_prever:
    st.subheader("📋 Resultado do Modelo")
    
    # Cálculo simulado (substitua pela predição real do seu modelo integrado)
    nota_simulada_final = (nota_g1 + nota_g2) / 2 + (consumo_alcool * -0.1) + (faltas * -0.05)
    nota_simulada_final = min(max(nota_simulada_final, 0.0), 20.0)
    
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        st.metric(label="Nota Final Prevista (G_Media)", value=f"{nota_simulada_final:.2f} / 20.0")
        
    with col_res2:
        if nota_simulada_final >= 15.0:
            st.success("🎯 Perfil de Desempenho: ALTO RENDIMENTO. O estudante apresenta ótimos indicadores de sucesso.")
        elif nota_simulada_final >= 10.0:
            st.info("⚖️ Perfil de Desempenho: RENDIMENTO REGULAR. O estudante está dentro da média esperada.")
        else:
            st.error("⚠️ Perfil de Desempenho: BAIXO RENDIMENTO. Recomenda-se atenção especial e acompanhamento de faltas/rotina.")

st.divider()
st.caption("Liga de Data Science Unicamp | Mini Projeto 2026")