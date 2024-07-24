
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
from datetime import datetime, timedelta
import google.generativeai as genai
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px 
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import colorsys
from plotly.subplots import make_subplots
onehot = OneHotEncoder(handle_unknown='ignore')
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import matplotlib
import geodatasets
import PyPDF2
import io
import logging
import difflib
logging.basicConfig(level=logging.INFO)
matplotlib.use('Agg')
import difflib
from fuzzywuzzy import fuzz
from collections import deque
# Define as global variables if needed
n_neighbors = 3 
if 'informacoes_edital' not in st.session_state:
    st.session_state.informacoes_edital = None
segmentos_map = {
    "Segmento": [
        "Mineração", "Outras Industrias", "Rodovias", "Barragens e Canais", "Fotovoltaico",
        "Metrôs", "Ferrovias", "Papel e Celulose", "Água e Esgoto", "Aeroportos",
        "Hidrelétrica", "Fertilizantes", "Óleo e Gás", "Macrodrenagem", "Transmissão",
        "Termoelétrica", "Usinas Eólicas", "Equipamentos Públicos", "Portos", "Edificações"
    ],
    "ID": list(range(1, 22))
}

cenario_map = {
    'Base': 0,
    'Conservador': 1,
    'Agressivo': 2,
    'Mapeamento': 3
}
PESOS_FONTES_DE_DADOS = {
    "base": 0.2,  # Peso da base de conhecimento-2024-07-05-13-12-28.xlsx
    "atual": 0.5,  # Peso da SF.xlsx
    "ganha": 0.8   # Peso da base de conhecimento-ganha.xlsx
}
cbm_info = {
    "nome": "Construtora Barbosa Mello",
    "setores_atuacao": ["Mineração", "Infraestrutura", "Energia", "Saneamento"],
    "especialidades": [
        "Obras rodoviárias", "Obras ferroviárias", "Obras de arte especiais",
"Obras de saneamento", "Barragens", "Usinas hidrelétricas",
"Obras portuárias", "Obras aeroportuárias", "Terraplenagem de grande porte",
"Manutenção mecânica", "Obras civis industriais em concreto armado",
"Obras de drenagens superficiais e profundas"
    ],
    "certificacoes": ["ISO 9001", "ISO 14001", "ISO 45001"],
    "capacidade_financeira": 2000000000, 
    
    "principais_clientes": [
        "Vale", "Anglo American", "Samarco", "ArcelorMittal",
        "Governo Federal", "Governos Estaduais"
    ],
    "experiencia_anos": 67,
    "regiao_atuacao": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],

    "equipamentos_proprios": True,

}
relacionamento_empresas = {
    'Vale': 10,
    'ArcelorMittal': 9,
    'Outra' : 2,
    'Samarco': 8,
    'Anglo American': 8,
    'Usiminas': 7,
    'EcoRodovias': 3,
    'CSN': 6,
    'BAMIM - Bahia Mineração': 7,
    'Rumo SA': 7,
    'Copasa': 7,
    'Suzano Papel e Celulose': 3,
    'Heineken Brasil': 3,
    'CBA - Companhia Brasileira de Alumínio': 8,
    'Zurich': 8,
    'Alcoa': 3,
    'Cemig - Companhia Energética de Minas Gerais': 5,
    'Aura Minerals': 3,
    'Via Bahia': 3,
    'BRK Ambiental': 3,
    'Governo de Minas Gerais': 3,
    'Equinox Gold': 3,
    'AB Nascentes das Gerais': 3,
    'Promon Engenharia': 3,
    'Sabes': 3,
    'TAG - Transportadora Associada de Gás S.A.': 7,
    'Gerdau': 3,
    'Correias Mercúrio': 3,
    'MUSA - Mineração Usiminas S.A.': 3,
    'Vale S.A. - Ferrolvia': 6,
    'TESC- TERMINAL PORTUÁRIO SANTA CATARINA': 6,
    'Serra Verde Pesquisa e Mineração (SVPM)': 3,
    'Fundacao Renova': 3,
    'Prefeitura Municipal de Mairiporã': 5,
    'Hydro Paragominas': 5,
    'Petrobras': 5,
    'ENESA Engenharia LTDA.': 5,
    'Governo do estado do Amapá': 5,
    'Hydro': 5,
    'Governo do Estado - Rio Grande do Sul': 5,
    'Aena Internacional': 6,
    'Sinoma': 3,
    'Nexa Resources': 3,
    'Albion a Codo ra Energia': 2,
    'Prefeitura Municipal de Nova Serrana': 2,
    'GeoFix': 2,
    'CEI - Companhia Energética Integrada': 2,
    'Energisa': 5,
    'Presidência da República - PPI': 2,
    'Pátria Investimentos': 2,
    'Morro do Pilar Minerais S.A.': 2,
    'Mineração Morro do Ipê': 2,
    'PQ Corporation': 2,
    'G-Mining Ventures': 2,
    'Enel Green Power': 3,
    'Lundin Mining': 3,
    'MRN - Mineração Rio do Norte': 7,
    'Cagece': 3,
    'Governo do Estado do Mato Grosso': 3,
    'Prefeitura de Contagem': 3,
    'Arena MRV': 6,
    'Amazon Energy': 3,
    'GE': 3,
    'Amarelo Gold': 3,
    'AMG Mineração': 3,
    'AngloGold Ashanti': 5,
    'JMN Mineração': 3,
    'Largo Resources': 3,
    'Sigma Lithium': 3,
    'CMOC - China Molybdenum Company': 3,
    'Mineração COONEMP': 3,
    'Imetame Energia': 3,
    'ENEVA SA': 3,
    'Engie': 8,
    'Voltalia Energia do Brasil': 3,
    'Shell Energy Brasil': 3,
    'VLI Logística': 3,
    'New Fortress Energy': 3,
    'Azul Linhas Aéreas': 3,
    'IBS Energy': 3,
    'Prumo Logística Global': 3,
    'MRS Logística S.A.': 3,
    'Votorantim Energia': 3,
    'Scatec': 3,
    'CIP': 3,
    'Auren Energia': 3,
    'Vinci Airports': 3,
    'Horizonte Minerals': 3,
    'Samotracia': 3,
    'Ib Lercon Engenharia': 3,
    'CBM SA': 6,
    'CINMED': 3,
    'Docol': 3,
    'Bemisa': 3,
    'Atlas Energia Renovável': 3,
    'Grupo Energia': 3,
    'J.Mendes': 3,
    'MetrôRio': 3,
    'Metrô São Paulo': 3,
    'CS Grãos do Piauí': 3,
    'Qair': 3,
    'CPFL': 3,
    'Rio Energy': 3,
    'Draft Solutions': 3,
    'Ultracargo': 3,
    'Prefeitura de Belo Horizonte': 3,
    'Royry Tecnologia': 3,
    'Renovias': 3,
    'GRUAiport': 3,
    'Ambev': 3,
    'Echoenergia': 3,
    'Galp': 3,
    'Kinross': 3,
    'Ero Copper': 3,
    'Terna Plus': 3,
    'Elera Renováveis': 3,
    'Grupo Avante': 3,
    'Ero Brasil': 3,
    'Komatsu': 3,
    'Ipiranga': 3,
    'Eletrobras Furnas': 3,
    'RHI Magnesita': 3,
    'XP Infra': 3,
    'ANAC - Agência Nacional de Aviação Civil': 3,
    'GNA': 3,
    'Casa dos Ventos': 6,
    'Votorantim Cimentos': 3,
    'Fortes Engenharia': 3,
    'CRO': 3,
    'Gas Brasiliano Distribuidora S.A.': 3,
    'Everest Empreendimentos': 3,
    'Aperam': 3,
    'Companhia de Gás de Minas Gerais - GASMIG': 3,
    'Fraport': 7,
    'Omega Engenharia': 3,
    'Rota do Oeste': 7,
    'Sulgás': 3,
    'Tecnored Desenvolvimento Tecnológico': 3,
    'Cimento Tupi': 3,
    'Aracuo': 3,
    'Jaguar Mining': 3,
    'Vale Base Metals': 7,
    'Fomento do Brasil': 3,
    'APM Terminals': 3,
    'Vallourec': 3,
    'DNIT - Departamento Nacional de Estradas e Rodagens': 6,
    'AG - Andrade Gutierrez': 3,
    'Sudecap': 3,
    'Logum Logística': 3,
    'Lenzing': 3,
    'Eldorado Gold': 3,
    'Belo Sun Mining': 3,
    'LAP - Lima Airport': 3,
    'Porto Consult': 3,
    'Noxis Energy Participações': 3,
    'Artesp': 3,
    'Socicam Serviços Urbanos': 3,
    'DEER MG': 3,
    'Portocem': 3,
    'Geramar III': 3,
    'CELBA - Centrais Elétricas de Barcarena S.A.': 3,
    'Comgás': 3,
    'CSPAR': 3,
    'Copelmi Mineração': 3,
    'Vanádio de Maracás S.A.': 3,
    'McCain Foods': 3,
    'Igua SA': 3,
    'Parabél S.A.': 3,
    'Ageo Terminais': 3,
    'Seinfra': 3,
    'PetroCity Portos S.A.': 3,
    'Atiaia Energia': 3,
    'Infraero Aeroportos': 3,
    'Fundação FEAC': 3,
    'Brado Logística': 3,
    'SAAE Sete Lagoas': 3,
    'AB Concessões': 3,
    'K-Infra Rodovia do Aço': 3,
    'Ight S/A': 3,
    'Governo do Peru': 3,
    'Ivepar': 3,
    'Sociedade de Abastecimento de Água e Saneamento S/A- SANASA': 3,
    'Governo de São Paulo': 3,
    'SEMAE': 3
}
relacionamento_estados = {
        'AC': 4,
    'AL': 3,
    'AP': 4,
    'AM': 4,
    'BA': 5,
    'CE': 4,
    'ES': 4,
    'GO': 4,
    'MA': 4,
    'MT': 8,
    'MS': 6,
    'MG': 9,
    'PA': 10,
    'PB': 5,
    'PR': 5,
    'PE': 5,
    'PI': 5,
    'RJ': 5,
    'RN': 8,
    'RS': 6,
    'RO': 4,
    'RR': 4,
    'SC': 4,
    'SP': 6,
    'SE': 4,
    'TO': 4,
    'DF': 4
}
# Configurações do Gemini
genai.configure(api_key="APIKEY")  
MODELO_NOME = "gemini-1.5-pro"
model = genai.GenerativeModel(MODELO_NOME)

def encontrar_propostas_similares(X, X_scaled, tfidf_matrix, nova_proposta, scaler, tfidf, 
                                  feature_columns, features_numericas, cat_features, onehot, 
                                  n_neighbors=3, peso_numerico=0.7, peso_texto=0.3):
    """
    Encontra propostas similares com base em características numéricas, categóricas e textuais.

    Parâmetros:
    - X: DataFrame com todas as features
    - X_scaled: Array com features numéricas e categóricas escalonadas
    - tfidf_matrix: Matriz TF-IDF dos textos dos objetos
    - nova_proposta: Dicionário com os dados da nova proposta
    - scaler: Objeto StandardScaler ajustado aos dados de treinamento
    - tfidf: Objeto TfidfVectorizer ajustado aos dados de treinamento
    - feature_columns: Lista de todas as colunas de features
    - features_numericas: Lista de features numéricas
    - cat_features: Lista de features categóricas
    - onehot: Objeto OneHotEncoder ajustado aos dados de treinamento
    - n_neighbors: Número de vizinhos similares a retornar
    - peso_numerico: Peso para similaridade baseada em features numéricas/categóricas
    - peso_texto: Peso para similaridade baseada no texto do objeto

    Retorna:
    - indices_similares: Índices das propostas mais similares
    - similaridades: Scores de similaridade correspondentes
    """
    
    # Preparar nova proposta
    nova_proposta_df = pd.DataFrame([nova_proposta])
    
    # Processar features categóricas
    nova_proposta_df[cat_features] = nova_proposta_df[cat_features].astype(str).fillna("Desconhecido")
    cat_encoded = onehot.transform(nova_proposta_df[cat_features])
    
    # Criar DataFrame com todas as features
    nova_proposta_encoded = pd.DataFrame(cat_encoded, 
                                         columns=onehot.get_feature_names_out(cat_features),
                                         index=nova_proposta_df.index)
    
    # Adicionar features numéricas
    for feature in features_numericas:
        nova_proposta_encoded[feature] = nova_proposta_df.get(feature, 0)
    
    # Alinhar colunas com o conjunto de treinamento
    nova_proposta_encoded = nova_proposta_encoded.reindex(columns=X.columns, fill_value=0)
    
    # Aplicar escalonamento
    nova_proposta_scaled = scaler.transform(nova_proposta_encoded)
    
    # Calcular similaridade baseada em features numéricas e categóricas
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(X_scaled)
    distancias, indices = nn.kneighbors(nova_proposta_scaled)
    
    # Calcular similaridade baseada no texto do Objeto
    nova_proposta_tfidf = tfidf.transform([str(nova_proposta.get('Objeto', ''))])
    similaridades_texto = cosine_similarity(nova_proposta_tfidf, tfidf_matrix).flatten()
    
    # Combinar similaridades
    similaridades_combinadas = np.zeros(X.shape[0])
    similaridades_combinadas[indices[0]] = peso_numerico * (1 - distancias.flatten())
    similaridades_combinadas += peso_texto * similaridades_texto
    
    # Encontrar as propostas mais similares
    indices_similares = similaridades_combinadas.argsort()[-n_neighbors:][::-1]
    similaridades = similaridades_combinadas[indices_similares]
    
    return indices_similares, similaridades

def carregar_e_tratar_dados(caminho_base, caminho_atual, caminho_ganha):
    df_base = pd.read_excel(caminho_base)
    df_atual = pd.read_excel(caminho_atual)
    df_ganha = pd.read_excel(caminho_ganha)
    
    df_base['Peso'] = PESOS_FONTES_DE_DADOS['base']
    df_atual['Peso'] = PESOS_FONTES_DE_DADOS['atual']
    df_ganha['Peso'] = PESOS_FONTES_DE_DADOS['ganha']

    df_combinado = pd.concat([df_base, df_atual, df_ganha], ignore_index=True)
    
    df_combinado['Valor CBM.amount'] = pd.to_numeric(df_combinado['Valor CBM.amount'], errors='coerce')
    df_combinado['Data de Assinatura'] = pd.to_datetime(df_combinado['Data de Assinatura'], errors='coerce')
    df_combinado['Data Inicio Obra'] = pd.to_datetime(df_combinado['Data Inicio Obra'], errors='coerce')
    
    
    
    # Preencher valores nulos
    colunas_numericas = ['Valor CBM.amount']
    for coluna in colunas_numericas:
        df_combinado[coluna].fillna(df_combinado[coluna].mean(), inplace=True)
    
    df_combinado['Data de Assinatura'].fillna(df_combinado['Data de Assinatura'].mode()[0], inplace=True)
    
    return df_combinado
def mapear_relacionamento(df):
    df['Relacionamento_Empresa'] = df['Empresa'].map(relacionamento_empresas).fillna(0)
    df['Relacionamento_Estado'] = df['Estado'].map(relacionamento_estados).fillna(0)
    return df


def preparar_dados_ml(df):
    df = mapear_relacionamento(df)
    
    # Normalização do Valor CBM
    df['Valor CBM Normalizado'] = (df['Valor CBM.amount'] - df['Valor CBM.amount'].min()) / (df['Valor CBM.amount'].max() - df['Valor CBM.amount'].min())
    
    # Preparação de features numéricas e categóricas
    features_numericas = ['Valor CBM Normalizado', 'Relacionamento_Empresa', 'Relacionamento_Estado']
    cat_features = ['Setor', 'Segmento', 'Fase', 'Cenário', 'Estado']
    
    # Preencher valores NaN em cat_features
    df[cat_features] = df[cat_features].fillna('Desconhecido')

    # One-hot encoding para features categóricas
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_encoded = onehot.fit_transform(df[cat_features])
    cat_columns = onehot.get_feature_names_out(cat_features)

    # Criar DataFrame com as features codificadas
    df_encoded = pd.DataFrame(cat_encoded, columns=cat_columns, index=df.index)
    df_encoded = pd.concat([df[features_numericas], df_encoded], axis=1)
    
    # Combinar features numéricas e categóricas
    features = df_encoded.columns.tolist()
    X = df_encoded
    
    # Normalização das features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Processamento do texto do Objeto
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Objeto'].fillna(''))
    
    return X, X_scaled, tfidf_matrix, features, scaler, tfidf, cat_features, features_numericas, onehot


def avaliar_modelo(X, y, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    scores = cross_val_score(nn, X, y, cv=5, scoring='neg_mean_squared_error')
    return np.sqrt(-scores.mean())
def encontrar_empresa_similar(empresa_digitada, empresas_conhecidas, cutoff=60, max_matches=3):
    """
    Encontra empresas similares à empresa digitada dentro de uma lista de empresas conhecidas.

    Parâmetros:
    - empresa_digitada (str): O nome da empresa a ser pesquisada.
    - empresas_conhecidas (list): Lista de nomes de empresas conhecidas.
    - cutoff (int): Pontuação mínima de similaridade (0-100). Padrão é 60.
    - max_matches (int): Número máximo de correspondências a retornar. Padrão é 3.

    Retorna:
    - list: Lista de tuplas (nome_empresa, pontuacao) ordenada por pontuação decrescente.
    """
    # Normaliza a entrada
    empresa_digitada = empresa_digitada.lower().strip()
    
    # Usa get_close_matches para uma primeira filtragem
    matches_iniciais = difflib.get_close_matches(empresa_digitada, empresas_conhecidas, n=10, cutoff=0.5)
    
    # Se não houver correspondências iniciais, retorna uma lista vazia
    if not matches_iniciais:
        return []
    
    # Usa fuzzywuzzy para uma comparação mais precisa
    matches_refinados = []
    for empresa in matches_iniciais:
        ratio = fuzz.ratio(empresa_digitada, empresa.lower())
        if ratio >= cutoff:
            matches_refinados.append((empresa, ratio))
    
    # Ordena os resultados por pontuação e limita ao número máximo de correspondências
    matches_refinados.sort(key=lambda x: x[1], reverse=True)
    return matches_refinados[:max_matches]
def contar_projetos_no_mes(df, data_assinatura):
    mes_ano = data_assinatura.strftime('%Y-%m')
    return df[df['Data de Assinatura'].dt.strftime('%Y-%m') == mes_ano].shape[0]

def pontuar_proposta(nova_proposta, empresas_conhecidas, df_combinado):
    pontuacao = 0
    peso_total = 0

    # Dicionários de pontuação
    pontuacao_cenario = {
        'Conservador': 10,
        'Agressivo': 6,
        'Base': 4,
        'Mapeamento': 4
    }

    pontuacao_setor = {
        'Transporte': 8,
        'Água e Saneamento': 6,
        'Mineração': 10,
        'Energia': 8,
        'Industrias': 6
    }

    # Função para pontuar o valor CBM
    def pontuar_valor_cbm(valor):
        if 100_000_000 <= valor < 200_000_000:
            return 3
        elif 200_000_000 <= valor < 400_000_000:
            return 7
        elif 400_000_000 <= valor < 600_000_000:
            return 8
        elif 600_000_000 <= valor < 1_000_000_000:
            return 9
        elif valor >= 1_000_000_000:
            return 10
        else:
            return 0  # para valores abaixo de 100 milhões

    # Função para pontuar com base no número de projetos no mês
    def pontuar_projetos_no_mes(num_projetos):
        if 1 <= num_projetos <= 3:
            return 10
        elif num_projetos == 4:
            return 8
        elif num_projetos == 5:
            return 6
        elif num_projetos == 6:
            return 4
        else:
            return 2

    # Pesos
    pesos = {
        'valor_cbm': 7,
        'relacionamento': 4,
        'setor': 5,
        'cenario': 3,
        'empresa': 5,
        'projetos_no_mes': 6  # Novo peso para projetos no mês
    }

    # Pontuação do Valor CBM
    valor_cbm = nova_proposta['Valor CBM.amount']
    pontuacao += pontuar_valor_cbm(valor_cbm) * pesos['valor_cbm']
    peso_total += pesos['valor_cbm']

    # Pontuação do Relacionamento (empresa e estado)
    if 'Relacionamento_Empresa' in nova_proposta and 'Relacionamento_Estado' in nova_proposta:
        pontuacao_relacionamento = (nova_proposta['Relacionamento_Empresa'] + nova_proposta['Relacionamento_Estado']) / 2
        pontuacao += pontuacao_relacionamento * pesos['relacionamento']
        peso_total += pesos['relacionamento']

    # Pontuação do Setor
    setor = nova_proposta['Setor']
    if setor in pontuacao_setor:
        pontuacao += pontuacao_setor[setor] * pesos['setor']
    peso_total += pesos['setor']

    # Pontuação do Cenário
    cenario = nova_proposta['Cenário']
    if cenario in pontuacao_cenario:
        pontuacao += pontuacao_cenario[cenario] * pesos['cenario']
    peso_total += pesos['cenario']

    # Pontuação da Empresa
    empresa_digitada = nova_proposta['Empresa']
    empresa_similar = encontrar_empresa_similar(empresa_digitada, empresas_conhecidas)
    if empresa_similar:
        pontuacao += 10 * pesos['empresa']  # Pontuação máxima se a empresa for identificada
    else:
        pontuacao += 1 * pesos['empresa']  # Pontuação mínima se a empresa não for identificada
    peso_total += pesos['empresa']

    # Pontuação baseada no número de projetos no mês de assinatura
    data_assinatura = nova_proposta['Data de Assinatura']
    num_projetos_no_mes = contar_projetos_no_mes(df_combinado, data_assinatura)
    pontuacao += pontuar_projetos_no_mes(num_projetos_no_mes) * pesos['projetos_no_mes']
    peso_total += pesos['projetos_no_mes']

    # Cálculo da pontuação final normalizada (0-100)
    pontuacao_final = (pontuacao / peso_total) * 10  # Multiplica por 10 para escala de 0-100

    return pontuacao_final, empresa_similar, num_projetos_no_mes

# Função para classificar a proposta
def classificar_proposta(pontuacao):
    if pontuacao >= 90:
        return "Excelente"
    elif pontuacao >= 85:
        return "Muito Boa"
    elif pontuacao >= 80:
        return "Boa"
    elif pontuacao >= 75:
        return "Razoável"
    elif pontuacao >= 70:
        return "Regular"
    else:
        return "Precisa de Atenção"

# Função para inicializar o histórico de chat na sessão, se ainda não existir
def init_chat_history():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = deque(maxlen=10)  # Mantém as últimas 10 interações

# Função para adicionar mensagens ao histórico
def add_to_chat_history(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

# Função para formatar o histórico de chat para o prompt
def format_chat_history():
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])

def perguntar_sobre_edital(pergunta, edital_texto, cbm_info, info_sf, model):
    init_chat_history()
    add_to_chat_history("Humano", pergunta)
    
    chat_history = format_chat_history()
    
    prompt = f"""
    Você é um assistente especializado em análise de editais de licitação para a Construtora Barbosa Mello (CBM). Sua tarefa é responder perguntas sobre o edital de forma precisa, clara e concisa, sempre fornecendo a página de referência no edital quando possível.

    Contexto:

    Texto completo do edital:
    {edital_texto}

    Informações sobre a CBM:
    - Setores de atuação: {', '.join(cbm_info['setores_atuacao'])}
    - Especialidades: {', '.join(cbm_info['especialidades'])}
    - Certificações: {', '.join(cbm_info['certificacoes'])}
    - Capacidade financeira: R$ {cbm_info['capacidade_financeira']:,}
    - Principais clientes: {', '.join(cbm_info['principais_clientes'])}
    - Regiões de atuação: {', '.join(cbm_info['regiao_atuacao'])}
    - Equipamentos próprios: {'Sim' if cbm_info['equipamentos_proprios'] else 'Não'}

    Dados atuais de projetos em andamento da CBM (baseados na planilha SF):
    - Média de valor dos projetos: R$ {info_sf['media_valor_cbm']:,.2f}
    - Total de projetos em andamento: {info_sf['total_projetos']}
    - Setores de atuação atual: {', '.join(f"{k} ({v} projetos)" for k, v in info_sf['setores'].items())}
    - Segmentos de atuação atual: {', '.join(f"{k} ({v} projetos)" for k, v in info_sf['segmentos'].items())}
    - Estados de atuação atual: {', '.join(f"{k} ({v} projetos)" for k, v in info_sf['estados'].items())}
    - Valor total dos projetos em andamento: R$ {info_sf['total_valor_cbm']:,.2f}
    - Maior projeto atual: {info_sf['maior_projeto']['Nome da oportunidade']} (R$ {info_sf['maior_projeto']['Valor CBM.amount']:,.2f})

    Histórico de chat:
    {chat_history}

    Com base nessas informações e no histórico de chat, por favor responda à seguinte pergunta:

    {pergunta}

    Instruções para a resposta:
    1. Forneça uma resposta clara, concisa e relevante.
    2. Base sua resposta nas informações do edital, nos dados da CBM e da planilha SF.
    3. Sempre que possível, cite a página específica do edital onde a informação foi encontrada.
    4. Se a informação estiver em múltiplas páginas, cite todas as páginas relevantes.
    5. Se a pergunta não puder ser respondida com as informações disponíveis, indique isso claramente.
    6. Se a resposta envolver uma análise comparativa entre o edital e as capacidades da CBM, explique claramente a relação.
    7. Quando relevante, mencione como a informação se relaciona com os projetos atuais ou a experiência da CBM.
    8. Se a pergunta for sobre um aspecto técnico, financeiro ou legal específico, forneça detalhes precisos conforme o edital.
    9. Caso a pergunta requeira uma recomendação, baseie-a nos dados disponíveis e justifique sua sugestão.
    Formato da resposta:
    - Resposta: [Sua resposta detalhada aqui]
    - Referência no Edital: [Página(s) X, Y, Z] ou [Não especificado no edital]
  

    Lembre-se: precisão, clareza e relevância são cruciais. Sempre relacione sua resposta às necessidades e capacidades específicas da CBM quando apropriado.
    """

    response = model.generate_content(prompt)
    resposta = response.text
    add_to_chat_history("Assistente", resposta)
    return resposta
    
def calcular_meses_com_muitos_projetos(df_sf, coluna_data='Data de Assinatura', limite=5, periodo_analise=None):
    """
    Calcula o número de meses com mais projetos simultâneos que o limite especificado,
    considerando apenas os dados da planilha SF.

    Parâmetros:
    - df_sf (pd.DataFrame): DataFrame contendo os dados da planilha SF.
    - coluna_data (str): Nome da coluna que contém as datas de assinatura dos projetos.
    - limite (int): Número limite de projetos simultâneos.
    - periodo_analise (tuple): Tupla com (data_inicio, data_fim) para restringir a análise.
                               Se None, considera todo o período dos dados.

    Retorna:
    - int: Número de meses com mais projetos simultâneos que o limite.
    """
    if df_sf is None or df_sf.empty:
        return 0

    # Verifica se a coluna de data existe
    if coluna_data not in df_sf.columns:
        raise ValueError(f"A coluna '{coluna_data}' não existe no DataFrame.")

    # Converte a coluna de data para datetime
    df_sf[coluna_data] = pd.to_datetime(df_sf[coluna_data], errors='coerce')
    df_sf = df_sf.dropna(subset=[coluna_data])

    # Aplica o filtro de período, se especificado
    if periodo_analise:
        data_inicio, data_fim = periodo_analise
        df_sf = df_sf[(df_sf[coluna_data] >= data_inicio) & (df_sf[coluna_data] <= data_fim)]

    if df_sf.empty:
        return 0

    # Calcula o número de projetos por mês
    projetos_por_mes = df_sf.groupby(df_sf[coluna_data].dt.to_period('M')).size().reset_index(name='Projetos Simultâneos')

    # Conta os meses com mais projetos que o limite
    meses_acima_do_limite = sum(projetos_por_mes['Projetos Simultâneos'] > limite)

    return meses_acima_do_limite


def gerar_prompt_gemini(nova_proposta, df_combinado, indices_propostas_similares, df_mercado, projetos_simultaneos, cbm_info, limite_projetos=5):
    """
    Gera um prompt detalhado para análise de uma nova proposta de projeto.

    Parâmetros:
    - nova_proposta (dict): Dicionário contendo os detalhes da nova proposta.
    - df_combinado (pd.DataFrame): DataFrame com todas as propostas.
    - indices_propostas_similares (list): Índices das propostas similares.
    - df_mercado (pd.DataFrame): DataFrame com dados de mercado.
    - projetos_simultaneos (pd.DataFrame): DataFrame com informações sobre projetos simultâneos.
    - cbm_info (dict): Informações sobre a Construtora Barbosa Mello.
    - limite_projetos (int): Limite de projetos simultâneos para análise.

    Retorna:
    - str: Prompt formatado para o modelo Gemini.
    """
    try:
        # Extrair propostas similares
        propostas_similares = df_combinado.iloc[indices_propostas_similares]

        # Análise de mercado
        mercado_setor = df_mercado[df_mercado['Setor'] == nova_proposta['Setor']]
        media_valor_mercado = mercado_setor['Valor CBM.amount'].mean()
        total_projetos_setor = len(mercado_setor)
        total_valor_setor = mercado_setor['Valor CBM.amount'].sum()

        # Análise de projetos simultâneos
        meses_com_muitos_projetos = calcular_meses_com_muitos_projetos(projetos_simultaneos, limite=limite_projetos)

        # Informações do edital
        resumo_edital_info = get_resumo_edital_info()

        # Formatação da nova proposta
        nova_proposta_formatada = format_nova_proposta(nova_proposta)

        # Formatação das propostas similares
        propostas_similares_formatadas = format_propostas_similares(propostas_similares)

        prompt = f"""
        Analise a seguinte nova proposta de projeto para a Construtora Barbosa Mello (CBM):

        Nova Proposta:
        {nova_proposta_formatada}

        Informações sobre a CBM:
        {format_cbm_info(cbm_info)}

        Top {len(indices_propostas_similares)} Propostas Similares:
        {propostas_similares_formatadas}

        Análise de Mercado:
        - Média de valor de projetos no setor: R$ {media_valor_mercado:.2f}
        - Total de projetos no setor: {total_projetos_setor}
        - Valor total de projetos no setor: R$ {total_valor_setor:.2f}

        Análise de Projetos Simultâneos:
        - Número de meses com mais de {limite_projetos} projetos simultâneos: {meses_com_muitos_projetos}
        - Nota: Um número elevado de meses com mais de {limite_projetos} projetos simultâneos pode indicar sobrecarga na capacidade de execução da empresa.

        {resumo_edital_info}

        Com base nessas informações, forneça uma análise detalhada sobre:
        1. Similaridade entre a nova proposta e as propostas anteriores, com ênfase no Valor CBM e outros aspectos relevantes.
        2. Cenário competitivo para este tipo de projeto, considerando os dados de mercado fornecidos.
        3. Pontos fortes e fracos da nova proposta, incluindo prazo, valor CBM e índices de relacionamento.
        4. Comparação do valor da proposta com a média do mercado e impacto na competitividade.
        5. Recomendações sobre prosseguir ou não com a proposta, considerando todos os fatores relevantes.
        6. Riscos potenciais associados ao projeto, incluindo prazo, valor e relações com a empresa e o estado.
        7. Estratégias para melhorar a competitividade da proposta, se aplicável.
        8. Impacto dos projetos simultâneos na capacidade de execução da CBM.
        9. Alinhamento da proposta com as capacidades e experiência da CBM.
        10. Considerações sobre sustentabilidade e responsabilidade social corporativa, se relevante.

        Apresente a análise de forma clara, concisa e estruturada, destacando os pontos mais críticos para a tomada de decisão.
        """

        return prompt

    except Exception as e:
        return f"Erro ao gerar o prompt: {str(e)}"

def format_nova_proposta(nova_proposta):
    """Formata as informações da nova proposta."""
    return "\n".join([f"- {k}: {v}" for k, v in nova_proposta.items()])

def format_propostas_similares(propostas_similares):
    """Formata as informações das propostas similares."""
    colunas = ['Setor', 'Segmento', 'Valor CBM.amount', 'Data de Assinatura', 'Empresa', 'Estado', 'Relacionamento_Empresa', 'Relacionamento_Estado']
    return propostas_similares[colunas].to_string(index=False)

def format_cbm_info(cbm_info):
    """Formata as informações da CBM."""
    return "\n".join([f"- {k}: {v}" for k, v in cbm_info.items() if k in ['setores_atuacao', 'especialidades', 'experiencia_anos', 'capacidade_financeira']])

def get_resumo_edital_info():
    """Obtém informações do resumo do edital, se disponível."""
    if 'resumo_edital' in st.session_state and st.session_state.resumo_edital:
        return f"""
        Informações adicionais do edital analisado:
        {st.session_state.informacoes_edital if st.session_state.informacoes_edital else "Nenhuma informação específica do edital disponível."}

        Análise do edital:
        {st.session_state.resumo_edital}

        Considere como esta nova proposta se relaciona ou pode ser influenciada pelo edital analisado acima.
        """
    return ""

def calcular_meses_com_muitos_projetos(projetos_simultaneos, limite=5):
    """Calcula o número de meses com mais projetos simultâneos que o limite especificado."""
    if projetos_simultaneos is None or projetos_simultaneos.empty:
        return 0
    return sum(projetos_simultaneos['Projetos Simultâneos'] > limite)

def criar_animacao_processamento():
    animation_html = """
    <style>
    .processing-animation {
        width: 100%;
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #f0f0f0;
        overflow: hidden;
    }
    .processing-bar {
        width: 20px;
        height: 100px;
        background-color: #294E88;
        margin: 0 5px;
        animation: process 1.5s ease-in-out infinite;
    }
    @keyframes process {
        0%, 100% { transform: scaleY(0.3); }
        50% { transform: scaleY(1); }
    }
    .processing-bar:nth-child(2) { animation-delay: 0.1s; }
    .processing-bar:nth-child(3) { animation-delay: 0.2s; }
    .processing-bar:nth-child(4) { animation-delay: 0.3s; }
    .processing-bar:nth-child(5) { animation-delay: 0.4s; }
    </style>
    <div class="processing-animation">
        <div class="processing-bar"></div>
        <div class="processing-bar"></div>
        <div class="processing-bar"></div>
        <div class="processing-bar"></div>
        <div class="processing-bar"></div>
    </div>
    <p style="text-align: center; color: #294E88;">Processando dados...</p>
    """
    return animation_html

def criar_animacao_dados():
    animation_html = """
    <style>
    .data-animation {
        width: 100%;
        height: 200px;
        background-color: #f0f0f0;
        overflow: hidden;
        position: relative;
    }
    .data-point {
        position: absolute;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #294E88;
        opacity: 0;
        animation: dataFlow 3s linear infinite;
    }
    @keyframes dataFlow {
        0% { transform: translate(0, 0); opacity: 0; }
        20% { opacity: 1; }
        80% { opacity: 1; }
        100% { transform: translate(calc(100vw - 10px), 190px); opacity: 0; }
    }
    </style>
    <div class="data-animation">
        <div class="data-point" style="animation-delay: 0s;"></div>
        <div class="data-point" style="animation-delay: 0.5s;"></div>
        <div class="data-point" style="animation-delay: 1s;"></div>
        <div class="data-point" style="animation-delay: 1.5s;"></div>
        <div class="data-point" style="animation-delay: 2s;"></div>
    </div>
    <p style="text-align: center; color: #294E88;">Analisando dados...</p>
    """
    return animation_html
def processar_dados_sf(df):
    # Filtrar apenas os projetos em andamento
    df_andamento = df[df['Fase'].isin(['Proposta', 'Execução', 'Mobilização'])]
    
    info_sf = {
        "media_valor_cbm": df_andamento["Valor CBM.amount"].mean(),
        "total_projetos": len(df_andamento),
        "setores": df_andamento["Setor"].value_counts().to_dict(),
        "segmentos": df_andamento["Segmento"].value_counts().to_dict(),
        "estados": df_andamento["Estado"].value_counts().to_dict(),
        "fases": df_andamento["Fase"].value_counts().to_dict(),
        "cenarios": df_andamento["Cenário"].value_counts().to_dict(),
        "total_valor_cbm": df_andamento["Valor CBM.amount"].sum(),
        "maior_projeto": df_andamento.loc[df_andamento["Valor CBM.amount"].idxmax(), ["Nome da oportunidade", "Valor CBM.amount"]].to_dict(),
        "projetos_foco": df_andamento[df_andamento["Projeto foco?"] == "VERDADEIRO"]["Nome da oportunidade"].tolist(),
    }
    return info_sf
def resumir_edital(file, model, cbm_info, info_sf):
    logging.info("Iniciando resumo do edital")
    
    if isinstance(file, io.StringIO):  # Se for entrada de texto
        texto_completo = file.getvalue()
    else:  # Se for upload de PDF
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        texto_completo = ""
        for page in pdf_reader.pages:
            texto_completo += page.extract_text()

    prompt = f"""
    Você é um especialista em análise de editais de licitação para a Construtora Barbosa Mello (CBM). Sua tarefa é resumir e analisar o seguinte edital, considerando as capacidades da CBM e seus dados atuais de projetos.

    Edital de Licitação:
    {texto_completo}

    Informações da Construtora Barbosa Mello (CBM):
    - Setores de atuação: {', '.join(cbm_info['setores_atuacao'])}
    - Especialidades: {', '.join(cbm_info['especialidades'])}
    - Certificações: {', '.join(cbm_info['certificacoes'])}
    - Capacidade financeira: R$ {cbm_info['capacidade_financeira']:,}
    - Principais clientes: {', '.join(cbm_info['principais_clientes'])}
    - Regiões de atuação: {', '.join(cbm_info['regiao_atuacao'])}
    - Equipamentos próprios: {'Sim' if cbm_info['equipamentos_proprios'] else 'Não'}

    Dados atuais de projetos em andamento da CBM (baseados na planilha SF):
    - Média de valor dos projetos: R$ {info_sf['media_valor_cbm']:,.2f}
    - Total de projetos em andamento: {info_sf['total_projetos']}
    - Setores de atuação atual: {', '.join(f"{k} ({v} projetos)" for k, v in info_sf['setores'].items())}
    - Segmentos de atuação atual: {', '.join(f"{k} ({v} projetos)" for k, v in info_sf['segmentos'].items())}
    - Estados de atuação atual: {', '.join(f"{k} ({v} projetos)" for k, v in info_sf['estados'].items())}
    - Fases dos projetos: {', '.join(f"{k} ({v})" for k, v in info_sf['fases'].items())}
    - Cenários dos projetos: {', '.join(f"{k} ({v})" for k, v in info_sf['cenarios'].items())}
    - Valor total dos projetos em andamento: R$ {info_sf['total_valor_cbm']:,.2f}
    - Maior projeto atual: {info_sf['maior_projeto']['Nome da oportunidade']} (R$ {info_sf['maior_projeto']['Valor CBM.amount']:,.2f})
    - Projetos foco: {', '.join(info_sf['projetos_foco'])}

    Por favor, analise o edital e forneça um resumo detalhado seguindo esta estrutura caso seja possível:

    1. Informações Gerais:
       a) Objeto da licitação (Página X)
       b) Órgão licitante (Página X)
       c) Modalidade de licitação (Página X)
       d) Número do edital (Página X)

    2. Detalhes Financeiros:
       a) Valor estimado do contrato (Página X)
       b) Forma de pagamento (Página X)
       c) Reajustes e revisões contratuais (Página X)

    3. Prazos e Datas Importantes:
       a) Data de publicação do edital (Página X)
       b) Prazo para esclarecimentos e impugnações (Página X)
       c) Data da abertura das propostas (Página X)
       d) Prazo de execução do objeto (Página X)

    4. Requisitos de Participação:
       a) Habilitação jurídica (Página X)
       b) Qualificação técnica (Página X)
       c) Qualificação econômico-financeira (Página X)
       d) Regularidade fiscal e trabalhista (Página X)

    5. Critérios de Julgamento:
       a) Tipo de licitação (menor preço, técnica e preço, etc.) (Página X)
       b) Critérios de pontuação técnica (se aplicável) (Página X)
       c) Critérios de desempate (Página X)

    6. Especificações Técnicas:
       a) Resumo do escopo do trabalho (Página X)
       b) Principais requisitos técnicos (Página X)
       c) Normas e padrões aplicáveis (Página X)

    7. Análise de Adequação da CBM:
       a) Compatibilidade do objeto com os setores e especialidades da CBM (Página X)
       b) Alinhamento com a capacidade financeira da CBM (Página X)
       c) Conformidade das certificações e qualificações da CBM (Página X)
       d) Relevância da experiência atual da CBM para esta licitação (Página X)
       e) Vantagens competitivas da CBM para este projeto (Página X)
       f) Possíveis desafios ou áreas de preocupação (Página X)

    8. Análise de Risco e Oportunidade:
       a) Principais riscos identificados (Página X)
       b) Oportunidades potenciais (Página X)
       c) Impacto no portfólio atual de projetos da CBM (Página X)

    9. Recomendação:
       Baseado na análise, forneça uma recomendação clara sobre a participação da CBM nesta licitação, justificando sua posição.
       Eventualmente você poderá receber editais que não contenham todas as informações, mas sempre se atente as datas
    10. Resumo Executivo:
        Apresente um resumo executivo conciso (máximo de 5 pontos-chave) que capture as informações mais críticas e a recomendação final.

    Observações importantes:
    - Se o edital exigir que a empresa seja inidônea, a CBM não poderá participar da licitação.
    - Destaque quaisquer requisitos ou condições que possam impedir a participação da CBM.
    - Considere o impacto deste projeto potencial no portfólio atual da CBM.

    Apresente a análise de forma clara, objetiva e estruturada, enfatizando os pontos mais relevantes para a tomada de decisão da CBM.
    """

    informacoes_edital = {
        "objeto": "Objeto da licitação extraído do edital",
        "valor_estimado": "Valor estimado extraído do edital",
        "prazo_execucao": "Prazo de execução extraído do edital",
    }

    response = model.generate_content(prompt)
    resumo = response.text

    return resumo, informacoes_edital
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def gerar_cores_intermediarias(cor1, cor2, n):
    rgb1 = hex_to_rgb(cor1)
    rgb2 = hex_to_rgb(cor2)
    cores = []
    for i in range(n):
        r = rgb1[0] + (rgb2[0] - rgb1[0]) * i / (n-1)
        g = rgb1[1] + (rgb2[1] - rgb1[1]) * i / (n-1)
        b = rgb1[2] + (rgb2[2] - rgb1[2]) * i / (n-1)
        cores.append(f'rgb({int(r)},{int(g)},{int(b)})')
    return cores


def init_chat_history():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_message(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def display_chat():
    for message in st.session_state.chat_history:
        if message["role"] == "Humano":
            st.markdown(f'<div style="background-color: #E3F2FD; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right;"><strong>Você:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #F0F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px;"><strong>Assistente:</strong> {message["content"]}</div>', unsafe_allow_html=True)




def gerar_sugestoes_perguntas(resumo_edital, model):
    prompt = f"""
    Com base no seguinte resumo de um edital de licitação:

    {resumo_edital}

    Gere 5 perguntas relevantes e específicas que um analista poderia fazer para obter mais informações cruciais sobre este edital. As perguntas devem ser variadas, cobrindo diferentes aspectos do edital, como requisitos técnicos, financeiros, prazos, critérios de avaliação, etc.

    Formato da resposta:
    1. [Pergunta 1]
    2. [Pergunta 2]
    3. [Pergunta 3]
    4. [Pergunta 4]
    5. [Pergunta 5]
    """

    response = model.generate_content(prompt)
    sugestoes = response.text.split('\n')
    return [s.split('. ', 1)[1] for s in sugestoes if s.strip()]



def gerar_graficos_mercado(nova_proposta, df_mercado):
    if df_mercado is None or df_mercado.empty:
        st.warning("Dados de mercado não disponíveis ou vazios.")
        return None, None, None, None

    cor_principal = "#294E88"
    cor_secundaria = "#547EA7"
    cores_cbm = gerar_cores_intermediarias(cor_principal, cor_secundaria, 8)

    tema_cbm = dict(
        font=dict(family="Arial, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_font=dict(size=24, color=cor_principal),
        legend=dict(
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor=cor_secundaria,
            borderwidth=1
        )
    )

    largura = 800
    altura = 500

    try:
        mercado_setor = df_mercado[df_mercado['Setor'] == nova_proposta['Setor']]

        # Scatter plot of market projects
        fig1 = px.scatter(mercado_setor, x='Valor CBM.amount', y='LOCAL_TIV', 
                          color='Estado', hover_data=['Nome da oportunidade'],
                          title=f"Projetos de {nova_proposta['Setor']} no Mercado",
                          color_discrete_sequence=cores_cbm,
                          width=largura, height=altura)
        fig1.add_trace(go.Scatter(x=[nova_proposta['Valor CBM.amount']], y=[0], 
                                  mode='markers', name='Nova Proposta',
                                  marker=dict(color='red', size=15, symbol='star-diamond')))
        fig1.update_layout(**tema_cbm)
        fig1.update_xaxes(title="Valor do Projeto (R$)", tickformat=",.0f")
        fig1.update_yaxes(title="TIV Local")

        # Histogram of project values
        fig2 = px.histogram(mercado_setor, x='Valor CBM.amount', nbins=30,
                            title=f"Distribuição de Valores de Projetos - {nova_proposta['Setor']}",
                            color_discrete_sequence=[cor_principal],
                            width=largura, height=altura)
        fig2.add_vline(x=nova_proposta['Valor CBM.amount'], line_dash="dash", line_color="red",
                       annotation_text="Nova Proposta", annotation_position="top right")
        fig2.update_layout(**tema_cbm)
        fig2.update_xaxes(title="Valor do Projeto (R$)", tickformat=",.0f")
        fig2.update_yaxes(title="Número de Projetos")

        # Box plot of project values by state
        fig3 = px.box(mercado_setor, x='Estado', y='Valor CBM.amount',
                      title=f"Distribuição de Valores por Estado - {nova_proposta['Setor']}",
                      color='Estado', color_discrete_sequence=cores_cbm,
                      width=largura, height=altura)
        fig3.add_trace(go.Scatter(x=[nova_proposta['Estado']], y=[nova_proposta['Valor CBM.amount']],
                                  mode='markers', name='Nova Proposta',
                                  marker=dict(color='red', size=15, symbol='star-diamond')))
        fig3.update_layout(**tema_cbm)
        fig3.update_xaxes(title="Estado")
        fig3.update_yaxes(title="Valor do Projeto (R$)", tickformat=",.0f")

        # Radar chart comparing new proposal to market averages
        mercado_medias = mercado_setor[['Valor CBM.amount', 'LOCAL_TIV']].mean()
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatterpolar(
            r=[nova_proposta['Valor CBM.amount'], nova_proposta.get('LOCAL_TIV', 0)],
            theta=['Valor do Projeto', 'TIV Local'],
            fill='toself',
            name='Nova Proposta'
        ))
        fig4.add_trace(go.Scatterpolar(
            r=[mercado_medias['Valor CBM.amount'], mercado_medias['LOCAL_TIV']],
            theta=['Valor do Projeto', 'TIV Local'],
            fill='toself',
            name='Média do Mercado'
        ))
        fig4.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(nova_proposta['Valor CBM.amount'], mercado_medias['Valor CBM.amount'])])),
            showlegend=True,
            title=f"Comparação da Proposta com Médias do Mercado - {nova_proposta['Setor']}",
            width=largura, height=altura,
            **tema_cbm
        )

        return fig1, fig2, fig3, fig4
    except Exception as e:
        st.error(f"Erro ao gerar gráficos: {str(e)}")
        return None, None, None, None

def calcular_projetos_simultaneos(df):
    """Calcula o número de projetos simultâneos por mês.

    Args:
        df (pd.DataFrame): O DataFrame contendo 'Data de Assinatura'.
    Returns:
        pd.DataFrame: Um DataFrame com 'Mês/Ano' e 'Projetos Simultâneos'.
    """
    if df.empty or 'Data de Assinatura' not in df.columns:
        st.warning(
            "Dados insuficientes para calcular projetos simultâneos. "
            "Verifique se a coluna 'Data de Assinatura' existe."
        )
        return None

    try:
        df['Data de Assinatura'] = pd.to_datetime(
            df['Data de Assinatura'], errors='coerce'
        )
        df.dropna(subset=['Data de Assinatura'], inplace=True)

        if not df['Data de Assinatura'].empty:
            # Encontra a data mínima e máxima de assinatura
            data_min = df['Data de Assinatura'].min()
            data_max = df['Data de Assinatura'].max()

            # Cria uma lista para armazenar os dados de projetos simultâneos
            dados_projetos = []

            # Itera pelos meses entre a data mínima e máxima
            data_atual = data_min
            while data_atual <= data_max:
                mes_ano = data_atual.strftime('%Y-%m')
                projetos_no_mes = 0

                # Verifica cada oportunidade
                for _, row in df.iterrows():
                    data_inicio = row['Data de Assinatura'] - timedelta(days=240)
                    data_fim = row['Data de Assinatura']

                    # Verifica se a oportunidade está ativa no mês atual
                    if data_inicio <= data_atual < data_fim:
                        projetos_no_mes += 1

                dados_projetos.append({'Mês/Ano': mes_ano, 'Projetos Simultâneos': projetos_no_mes})
                data_atual += timedelta(days=31)  # Avança para o próximo mês

            return pd.DataFrame(dados_projetos)
        else:
            st.warning("Não há datas válidas na coluna 'Data de Assinatura'.")
            return None

    except Exception as e:
        st.error(f"Erro ao calcular projetos simultâneos: {str(e)}")
        return None
    
def exibir_animacoes_e_gerar_analise(prompt):
    # Container para as animações
    animation_container = st.empty()
    
    # Exibir a primeira animação
    animation_container.markdown(criar_animacao_processamento(), unsafe_allow_html=True)
    
    
    # Exibir a segunda animação
    animation_container.markdown(criar_animacao_dados(), unsafe_allow_html=True)
    
    # Gerar a análise
    response = model.generate_content(prompt)
    analise = response.text
    
    # Remover a animação
    animation_container.empty()
    return analise

# Paleta de cores CBM
cor_principal = "#294E88"
cor_secundaria = "#547EA7"
cor_fundo = "#F5F5F5"

# Caminho para o logo
logo_path = r"C:\Users\070283\Downloads\Logo - fundo transparente - preto.png"

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Avaliador de Propostas CBM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    body {{
        font-family: 'Roboto', sans-serif;
        background-color: {cor_fundo};
        color: #333;
    }}

    .main {{
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}

    h1, h2, h3 {{
        color: {cor_principal};
        font-weight: 500;
    }}

    .stButton > button {{
        background-color: {cor_principal};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        background-color: #1E3A66;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }}

    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input {{
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }}

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stDateInput > div > div > input:focus {{
        border-color: {cor_secundaria};
        box-shadow: 0 0 0 2px rgba(84, 126, 167, 0.2);
    }}

    .user-message, .assistant-message {{
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        max-width: 80%;
    }}

    .user-message {{
        background-color: #E3F2FD;
        color: #1E3A66;
        float: right;
    }}

    .assistant-message {{
        background-color: {cor_principal};
        color: white;
        float: left;
    }}

    .dataframe {{
        border: none;
        border-radius: 5px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}

    .dataframe th {{
        background-color: {cor_secundaria};
        color: white;
        padding: 0.75rem;
        font-weight: 500;
    }}

    .dataframe td {{
        padding: 0.75rem;
        border-bottom: 1px solid #E0E0E0;
    }}

    .dataframe tr:nth-child(even) {{
        background-color: #F5F5F5;
    }}

    .dataframe tr:hover {{
        background-color: #E3F2FD;
    }}

    .stPlotlyChart {{
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}

    .metric-container {{
        background-color: {cor_principal};
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }}

    .metric-label {{
        font-size: 0.9rem;
        opacity: 0.8;
    }}

    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
    }}
    </style>
    """, 
    unsafe_allow_html=True
)

# --- Barra Lateral e Logo ---
st.sidebar.image(logo_path, use_column_width=True)

    # Criar abas
tab1, tab2 = st.tabs(["Análise de Propostas", "Resumo de Editais"])

with tab1:
    st.markdown("<h2 style='text-align: center; color: #294E88;'>Análise de Propostas</h2>", unsafe_allow_html=True)

# --- Cabeçalho ---
st.markdown(
    f"""
    <div style="background-color:{cor_principal};padding:1.5rem;border-radius:10px;margin-bottom:2rem;">
        <h1 style="color:white;text-align:center;margin:0;">Avaliador de Propostas CBM</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Funções auxiliares ---
@st.cache_data
def carregar_dados_mercado(caminho):
    try:
        df = pd.read_excel(caminho)
        colunas_necessarias = ['Setor', 'Valor CBM.amount', 'Estado', 'Fase', 'LOCAL_TIV', 'Nome da oportunidade', 'Objeto']
        df = df[colunas_necessarias]
        df['Valor CBM.amount'] = pd.to_numeric(df['Valor CBM.amount'], errors='coerce')
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados de mercado: {str(e)}")
        return None

@st.cache_data
def load_data():
    df_combinado = carregar_e_tratar_dados(
        "base de conhecimento-2024-07-05-13-12-28.xlsx",
        "SF.xlsx",
        "base de conhecimento-ganha.xlsx"
    )
    
    df_mercado = carregar_dados_mercado(
        "Industrial Info.xlsx"
    )
    
    info_sf = processar_dados_sf(df_combinado)
    
    return df_combinado, df_mercado, info_sf

# --- Carregamento dos dados ---
if 'df_combinado' not in st.session_state or 'df_mercado' not in st.session_state or 'info_sf' not in st.session_state:
    st.session_state.df_combinado, st.session_state.df_mercado, st.session_state.info_sf = load_data()

# Preparar dados ML uma vez
if 'ml_data' not in st.session_state:
    X, X_scaled, tfidf_matrix, features, scaler, tfidf, cat_features, features_numericas, onehot = preparar_dados_ml(st.session_state.df_combinado)
    st.session_state.ml_data = {
        'X': X, 'X_scaled': X_scaled, 'tfidf_matrix': tfidf_matrix,
        'features': features, 'scaler': scaler, 'tfidf': tfidf,
        'cat_features': cat_features, 'features_numericas': features_numericas, 'onehot': onehot
    }

# --- Interface principal ---
def main():
    
    st.markdown("<h2 style='text-align: center; color: #294E88;'>Nova Proposta</h2>", unsafe_allow_html=True)

    with st.form("nova_proposta_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            setor = st.selectbox("Setor", options=st.session_state.df_combinado['Setor'].unique())
            segmento = st.selectbox("Segmento", options=st.session_state.df_combinado['Segmento'].unique())
            valor_cbm = st.number_input("Valor CBM (R$)", min_value=0.0, format="%.2f")

        with col2:
            fase = st.selectbox("Fase", options=st.session_state.df_combinado['Fase'].unique())
            data_assinatura = st.date_input("Data de Assinatura")
            data_inicio_obra = st.date_input("Data de Início da Obra")

        with col3:
            cenario = st.selectbox("Cenário", options=st.session_state.df_combinado['Cenário'].unique())
            estado = st.selectbox("Estado", options=st.session_state.df_combinado['Estado'].unique())
            empresa = st.selectbox("Empresa", options=st.session_state.df_combinado['Empresa'].unique())

        objeto = st.text_area("Objeto", "")
        submitted = st.form_submit_button("Analisar Proposta")

    if submitted:
            # Criar um placeholder para as animações
        animation_placeholder = st.empty()
        
        # Exibir a primeira animação imediatamente
        animation_placeholder.markdown(criar_animacao_processamento(), unsafe_allow_html=True)

        # Processar a proposta e encontrar propostas similares
        nova_proposta = pd.Series({
            'Setor': setor,
            'Segmento': segmento,
            'Valor CBM.amount': valor_cbm,
            'Data de Assinatura': data_assinatura,
            'Data Inicio Obra': data_inicio_obra,
            'Fase': fase,
            'Cenário': cenario,
            'Estado': estado,
            'Empresa': empresa,
            'Objeto': objeto,
        })

         # Resto do processamento
        empresas_conhecidas = st.session_state.df_combinado['Empresa'].unique().tolist()
        pontuacao, empresa_similar, num_projetos_no_mes = pontuar_proposta(nova_proposta, empresas_conhecidas, st.session_state.df_combinado)
        classificacao = classificar_proposta(pontuacao)

        # Atualizar relacionamentos com base na empresa identificada
        if empresa_similar:
            nova_proposta['Relacionamento_Empresa'] = relacionamento_empresas.get(empresa_similar, 0)
        else:
            nova_proposta['Relacionamento_Empresa'] = 0
        nova_proposta['Relacionamento_Estado'] = relacionamento_estados.get(nova_proposta['Estado'], 0)

        with st.spinner("Analisando a proposta..."):
            ml_data = st.session_state.ml_data

            indices_propostas_similares, similaridades_propostas_similares = encontrar_propostas_similares(
                ml_data['X'],
                ml_data['X_scaled'],
                ml_data['tfidf_matrix'],
                nova_proposta,
                ml_data['scaler'],
                ml_data['tfidf'],
                ml_data['features'],
                ml_data['features_numericas'],
                ml_data['cat_features'],
                ml_data['onehot'],
                n_neighbors=5
            )

            propostas_similares = st.session_state.df_combinado.iloc[indices_propostas_similares].copy()
            propostas_similares['Similaridade'] = similaridades_propostas_similares

        projetos_simultaneos = calcular_projetos_simultaneos(st.session_state.df_combinado)

        with st.spinner("Gerando análise..."):
            prompt = gerar_prompt_gemini(nova_proposta, st.session_state.df_combinado, indices_propostas_similares, st.session_state.df_mercado, projetos_simultaneos)
            response = model.generate_content(prompt)
            analise = response.text if response else "Não foi possível gerar a análise"
        # Exibição dos resultados
        st.markdown("<h2 style='text-align: center; color: #294E88;'>Análise da Proposta</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Pontuação</div>
                    <div class="metric-value">{pontuacao:.2f}/100</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Classificação</div>
                    <div class="metric-value">{classificacao}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Projetos no Mês</div>
                    <div class="metric-value">{num_projetos_no_mes}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Exibir informação sobre a empresa identificada e projetos no mês
        if empresa_similar:
            st.info(f"Empresa identificada: {empresa_similar}")
        else:
            st.warning("Empresa não identificada na base de dados. A pontuação foi ajustada de acordo.")
        
        st.info(f"Número de projetos no mês de assinatura: {num_projetos_no_mes}")

        # Mudar para a segunda animação
        animation_placeholder.markdown(criar_animacao_dados(), unsafe_allow_html=True)

        # Remover a animação e exibir a análise
        animation_placeholder.empty()
        st.markdown("### Análise Detalhada")
        if analise:
            st.write(analise)
        else:
            st.warning("Não foi possível gerar a análise.")
        if not propostas_similares.empty:
            st.markdown("### Propostas Similares")
            st.dataframe(
                propostas_similares[['Nome da oportunidade', 'Setor', 'Segmento', 'Valor CBM.amount', 'Data de Assinatura', 'Objeto', 'Similaridade']],
                height=300
            )

            # Gráficos
            st.markdown("### Análise Visual")

            # Gráfico de Radar
            categorias = ['Valor CBM', 'Relacionamento_Empresa', 'Relacionamento_Estado']
            nova_proposta_valores = [nova_proposta['Valor CBM.amount'], 
                                     nova_proposta['Relacionamento_Empresa'], 
                                     nova_proposta['Relacionamento_Estado']]
            media_similares = propostas_similares[['Valor CBM.amount', 'Relacionamento_Empresa', 'Relacionamento_Estado']].mean().values

            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                r=nova_proposta_valores,
                theta=categorias,
                fill='toself',
                name='Nova Proposta'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=media_similares,
                theta=categorias,
                fill='toself',
                name='Média das Similares'
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(max(nova_proposta_valores), max(media_similares))]
                    )),
                showlegend=True,
                    title="Comparação com Propostas Similares",
                    height=500,  # Definindo altura fixa
                    width=700    # Definindo largura fixa
                    )
            
            st.plotly_chart(fig_radar, use_container_width=True)

            # Gráfico de Distribuição de Valores CBM
            fig_dist = px.histogram(propostas_similares, x="Valor CBM.amount", 
                        title="Distribuição de Valores CBM",
                        labels={"Valor CBM.amount": "Valor CBM"},
                        nbins=20,
                        height=400,  # Definindo altura fixa
                        width=700)
            fig_dist.add_vline(x=nova_proposta['Valor CBM.amount'], line_dash="dash", line_color="red",
                               annotation_text="Nova Proposta", annotation_position="top right")
            
            st.plotly_chart(fig_dist, use_container_width=True)

            # Gráfico de Dispersão Tridimensional
            fig_3d = px.scatter_3d(propostas_similares, 
                                   x='Valor CBM.amount', 
                                   y='Relacionamento_Empresa', 
                                   z='Relacionamento_Estado',
                                   color='Setor',
                                   title="Visualização 3D das Propostas",
                                   labels={"Valor CBM.amount": "Valor CBM", 
                                           "Relacionamento_Empresa": "Rel. Empresa", 
                                           "Relacionamento_Estado": "Rel. Estado"},
                                           
                                    height=600,  # Definindo altura fixa
                                    width=800)  
            
            fig_3d.add_scatter3d(x=[nova_proposta['Valor CBM.amount']], 
                                 y=[nova_proposta['Relacionamento_Empresa']], 
                                 z=[nova_proposta['Relacionamento_Estado']],
                                 mode='markers',
                                 marker=dict(size=10, color='red'),
                                 name='Nova Proposta')

            st.plotly_chart(fig_3d, use_container_width=True)

            # Timeline de Propostas
            fig_timeline = px.timeline(propostas_similares, 
                                       x_start="Data de Assinatura", 
                                       x_end="Data Inicio Obra", 
                                       y="Nome da oportunidade",
                                       color="Setor",
                           title="Timeline de Projetos Similares",
                           height=500,  # Definindo altura fixa
                           width=900) 
            
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Gráfico de dispersão interativo
            st.subheader("Distribuição de Propostas Similares")
            fig_scatter = px.scatter(
                propostas_similares,
                x="Data de Assinatura",
                y="Valor CBM.amount",
                color="Setor",
                size="Similaridade",
                hover_data=['Segmento', 'Fase', 'Cenário', 'Estado'],
                 title="Distribuição de Propostas Similares",
                 height=500,  # Definindo altura fixa
                 width=800    # Definindo largura fixa
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[nova_proposta['Data de Assinatura']],
                    y=[nova_proposta['Valor CBM.amount']],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15,
                        symbol='star'
                    ),
                    name='Nova Proposta'
                )
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            # Gerar gráficos de comparação com o mercado
            fig1, fig2, fig3, fig4 = gerar_graficos_mercado(nova_proposta, st.session_state.df_mercado)

            # Exibir os gráficos
            st.markdown("### Comparação com o Mercado")
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig3, use_container_width=True)
            st.plotly_chart(fig4, use_container_width=True)
with tab2:
    st.markdown("<h2 style='text-align: center; color: #294E88;'>Resumo de Editais</h2>", unsafe_allow_html=True)
    
    input_method = st.radio("Escolha o método de entrada:", ["Upload de PDF", "Texto"])
    
    if input_method == "Upload de PDF":
        uploaded_file = st.file_uploader("Faça o upload do edital (PDF)", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Resumir Edital (PDF)"):
                with st.spinner("Analisando o edital..."):
                    animation_placeholder = st.empty()
                    animation_placeholder.markdown(criar_animacao_dados(), unsafe_allow_html=True)
                    
                    resumo, informacoes_edital = resumir_edital(uploaded_file, model, cbm_info, st.session_state.info_sf)
                    
                    animation_placeholder.empty()
                    
                    st.session_state.resumo_edital = resumo
                    st.session_state.informacoes_edital = informacoes_edital
    
    else:  # Entrada de texto
        edital_text = st.text_area("Cole o texto do edital aqui:", height=300)
        
        if st.button("Resumir Edital (Texto)"):
            if edital_text:
                with st.spinner("Analisando o edital..."):
                    animation_placeholder = st.empty()
                    animation_placeholder.markdown(criar_animacao_dados(), unsafe_allow_html=True)
                    
                    # Cria um objeto similar a um arquivo com o texto fornecido
                    text_file = io.StringIO(edital_text)
                    
                    resumo, informacoes_edital = resumir_edital(text_file, model, cbm_info, st.session_state.info_sf)
                    
                    animation_placeholder.empty()
                    
                    st.session_state.resumo_edital = resumo
                    st.session_state.informacoes_edital = informacoes_edital
            else:
                st.warning("Por favor, insira o texto do edital antes de resumir.")
st.markdown("<h2 style='text-align: center; color: #294E88;'>Chat sobre o Edital</h2>", unsafe_allow_html=True)

# Inicializa o histórico de chat
init_chat_history()

# Exibe o histórico de chat
display_chat()

# Container para a interface de chat
chat_interface = st.container()

with chat_interface:
    # Gera sugestões de perguntas usando o Google Gemini
    if 'resumo_edital' in st.session_state and st.session_state.resumo_edital:
        if 'sugestoes_perguntas' not in st.session_state:
            with st.spinner("Gerando sugestões de perguntas..."):
                st.session_state.sugestoes_perguntas = gerar_sugestoes_perguntas(st.session_state.resumo_edital, model)
        
        # Dropdown para selecionar uma pergunta sugerida
        pergunta_selecionada = st.selectbox("Selecione uma pergunta sugerida ou escreva sua própria:", 
                                            [""] + st.session_state.sugestoes_perguntas)

        # Área de texto para digitar a pergunta
        pergunta_edital = st.text_area("Sua pergunta:", 
                                       value=pergunta_selecionada if pergunta_selecionada else "",
                                       height=100)

        # Botão para enviar a pergunta
        if st.button("Enviar Pergunta", key="enviar_pergunta"):
            if pergunta_edital.strip():  # Verifica se a pergunta não está vazia
                add_message("Humano", pergunta_edital)
                with st.spinner("Processando sua pergunta..."):
                    resposta = perguntar_sobre_edital(
                        pergunta_edital,
                        st.session_state.resumo_edital,
                        cbm_info,
                        st.session_state.info_sf,
                        model
                    )
                add_message("Assistente", resposta)
                st.experimental_rerun()  # Recarrega a página para mostrar a nova mensagem
            else:
                st.warning("Por favor, digite uma pergunta antes de enviar.")
    else:
        st.warning("Por favor, faça o upload e resuma um edital antes de fazer perguntas.")

# Botão para limpar o histórico do chat
if st.button("Limpar Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

    # Adicione esta seção após a exibição dos outros gráficos
    st.markdown("### Projetos Simultâneos ao Longo do Tempo")
    
    # Calcula os projetos simultâneos
    projetos_simultaneos = calcular_projetos_simultaneos(st.session_state.df_combinado)
    
    if projetos_simultaneos is not None and not projetos_simultaneos.empty:
        # Cria o gráfico de linha
        fig_projetos_simultaneos = px.line(
            projetos_simultaneos, 
            x='Mês/Ano', 
            y='Projetos Simultâneos',
            title='Número de Projetos Simultâneos por Mês',
            labels={'Mês/Ano': 'Data', 'Projetos Simultâneos': 'Número de Projetos'},
            height=500,
            width=800
        )
        
        # Personaliza o layout do gráfico
        fig_projetos_simultaneos.update_layout(
            xaxis_title="Data",
            yaxis_title="Número de Projetos Simultâneos",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        # Exibe o gráfico
        st.plotly_chart(fig_projetos_simultaneos, use_container_width=True)
    else:
        st.warning("Não foi possível gerar o gráfico de projetos simultâneos.")

    if st.checkbox("Mostrar Dados Brutos"):
        st.subheader("Dados Brutos")
        st.write(st.session_state.df_combinado)
if __name__ == "__main__":
    main()
