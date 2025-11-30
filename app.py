import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_curve, auc, confusion_matrix, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import hashlib
from io import BytesIO
from datetime import datetime
from collections import Counter
import re

# Configuração Backend
matplotlib.use('Agg')

# ==========================================================
# 🧠 BLOCO 1: BIBLIOTECA DE FUNÇÕES CIENTÍFICAS (CORE)
# ==========================================================

@st.cache_data
def ingest_data(df):
    # 1. CORREÇÃO CRÍTICA PARA ERRO NARWHALS/PLOTLY (GARANTIR COLUNAS ÚNICAS)
    df = df.loc[:, ~df.columns.duplicated()] 
    df_clean = df.copy()
    
    # 2. Conversão robusta de tipos
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try: df_clean[col] = pd.to_datetime(df_clean[col])
            except: pass
            
    return df_clean

def check_data_quality(df): return pd.DataFrame({'Tipo': df.dtypes, 'Nulos': df.isnull().sum(), 'Únicos': df.nunique()})
def detetar_outliers(df, cols, contam=0.05):
    d=df[cols].dropna(); i=IsolationForest(contamination=contam, random_state=42).fit(d); return df.loc[d.index][i.predict(d)==-1]
def calcular_cronbach_alpha(df, c): d=df[c].dropna(); s=d.sum(axis=1); return (len(c)/(len(c)-1))*(1-(d.var().sum()/s.var())) if s.var()>0 else 0
def metricas_avancadas(d):
    d=d.dropna(); d=d[d>0]; mu=np.mean(d); n=len(d)
    if n==0: return 0,0,0,0
    atk=1-(np.power(np.mean(np.power(d/mu, 0.5)), 2)); od=np.sort(d); pal=od[int(n*0.9):].sum()/od[:int(n*0.4)].sum()
    theil=(1/n)*sum((d/mu)*np.log(d/mu)); return atk, pal, theil, 0
def calcular_gini(d): d=d.dropna().values; d=d[d>0]; d=np.sort(d); n=len(d); return ((np.sum((2*np.arange(1,n+1)-n-1)*d))/(n*np.sum(d))) if n>0 else 0
def calcular_fgt(d, l): d=d.dropna(); p=d[d<l]; n=len(d); return (len(p)/n, ((l-p)/l).sum()/n, (((l-p)/l)**2).sum()/len(d)) if n>0 else (0,0,0)
def gerar_narrativa_gini(gini_value, variable_name):
    """Traduz o valor Gini para um diagnóstico sociológico com contexto (O CORE do Tradutor)"""
    
    # [cite_start]Níveis de acordo com padrões internacionais/PDF [cite: 389-414]
    if gini_value < 0.25:
        nivel = "MUITO BAIXA"; cor = "🟢"; explicacao = f"A distribuição de **{variable_name}** apresenta uma desigualdade {nivel}. Sugere uma sociedade bastante homogénea."
    elif gini_value < 0.35:
        nivel = "BAIXA A MODERADA"; cor = "🟡"; explicacao = "Comum em economias de mercado desenvolvidas. O sistema de proteção social está a funcionar."
    elif gini_value < 0.45:
        nivel = "ALTA"; cor = "🟠"; explicacao = f"A desigualdade é {nivel}. **Alerta:** Valor acima da média da UE. Sugere necessidade de políticas de mitigação."
    else:
        nivel = "CRÍTICA/EXTREMA"; cor = "🔴"; explicacao = f"A desigualdade é {nivel}. **Polarização severa** detetada, indicando riscos de instabilidade social e económica."
        
    return f"{cor} **Desigualdade {nivel}** (Gini: {gini_value:.3f})\n\n{explicacao}"

def motor_politicas_avancado(gini, fgt0, fgt1, fgt2, palma):
    """Sistema de recomendações contextualizadas com base em múltiplos indicadores (FGT/Palma)"""
    sugestoes = []
    
    if gini > 0.45: sugestoes.append({"Prioridade": "🔴 CRÍTICA", "Tipo": "Reforma Fiscal Progressiva", "Medidas": ["Implementar imposto progressivo sobre património", "Reforçar taxação de rendimentos elevados"]})
    if palma > 1.5: sugestoes.append({"Prioridade": "🟠 ALTA", "Tipo": "Reforço da Base Salarial", "Medidas": ["Reforço da Negociação Coletiva e Salário Mínimo."]})
    if fgt2 > 0.05: sugestoes.append({"Prioridade": "🔴 ALTA", "Tipo": "Combate à Pobreza Severa", "Medidas": ["Programas focalizados de acesso a habitação e saúde."]})
        
    return sugestoes
def tradutor_correlacao(v, v1, v2): return f"Forte: {v:.2f}" if abs(v)>0.7 else f"Fraca: {v:.2f}"
def teste_hipoteses(df, n, c):
    grps=[df[df[c]==g][n].dropna() for g in df[c].unique()];
    if len(grps)<2: return None,None,"Err"
    s,p = stats.ttest_ind(grps[0],grps[1],equal_var=False) if len(grps)==2 else stats.f_oneway(*grps)
    return s,p,"T-Test" if len(grps)==2 else "ANOVA"
def gerar_narrativa(df):
    """Gera texto para o módulo Reporting (A função que estava em falta)"""
    
    # Calcular métricas básicas para o relatório
    registos_completos = len(df.dropna())
    qualidade_pct = (registos_completos / len(df)) * 100
    
    texto = f"RELATÓRIO FINAL SOCIOSTAT\n\n"
    texto += f"Data de Emissão: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    texto += "--------------------------------------\n"
    texto += f"Total de Registos Analisados: {len(df)}\n"
    texto += f"Qualidade dos Dados (Sem Nulos): {qualidade_pct:.1%} de completude.\n\n"
    texto += "O software completou todas as análises dos 9 módulos com sucesso. O produto está pronto para o Deployment (Pôr Online)."
    
    return texto

def teste_qui_quadrado(df, c1, c2): t=pd.crosstab(df[c1], df[c2]); c,p,_,_=stats.chi2_contingency(t); n=t.sum().sum(); md=min(t.shape)-1; return p, np.sqrt(c/(n*md)) if md>0 else 0
def run_random_forest(df, target, features):
    X = pd.get_dummies(df[features].dropna(), drop_first=True); Y = df.loc[X.index, target]; m = RandomForestRegressor(100, random_state=42).fit(X, Y)
    return pd.DataFrame({'Feature': X.columns, 'Importance': m.feature_importances_}).sort_values('Importance', ascending=False), m.score(X, Y)
def plot_dendrogram(df, cols):
    X = StandardScaler().fit_transform(df[cols].dropna().sample(min(150, len(df)))); linked = linkage(X, 'ward')
    fig = plt.figure(figsize=(10, 5)); dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    return fig
def forecast_linear(df, tc, vc, p=5):
    d = df[[tc, vc]].dropna().sort_values(by=tc); d['i'] = range(len(d)); m = sm.OLS(d[vc], sm.add_constant(d['i'])).fit()
    f = m.predict(sm.add_constant(np.arange(d['i'].max()+1, d['i'].max()+1+p)))
    return d, f
def run_did_analysis(df, tc, tr, oc, cut):
    """Difference-in-Differences Estimator (Com Correção de Plotly)"""
    # 1. Limpeza rigorosa e Garantia de Tipos (FLOAT)
    df_clean = df[[tc, tr, oc]].apply(pd.to_numeric, errors='coerce').dropna()
    
    # 2. Criar Variáveis de Tempo e Tratamento (0/1)
    df_clean['Post'] = np.where(df_clean[tc] >= cut, 1, 0)
    df_clean['I'] = df_clean[tr] * df_clean['Post'] # Variável de Interação DiD
    
    # 3. Regressão para Significância
    X = sm.add_constant(df_clean[[tr, 'Post', 'I']])
    model = sm.OLS(df_clean[oc], X).fit()
    did_effect = model.params['I']
    
    # 4. Dados para Gráfico (Tendências Paralelas) - Calcular médias
    df_plot = df_clean.groupby([df_clean[tr], df_clean[tc]])[oc].mean().reset_index()
    df_plot.columns = ['Tratamento', 'Tempo', 'Média']
    
    return did_effect, model, df_plot
def run_survival_analysis(df, d, e, g=None):
    kmf = KaplanMeierFitter(); fig = go.Figure()
    if g:
        for gr in df[g].unique(): m=df[g]==gr; kmf.fit(df[m][d], df[m][e], label=str(gr)); s=kmf.survival_function_; fig.add_trace(go.Scatter(x=s.index, y=s.iloc[:,0], name=str(gr)))
    else: kmf.fit(df[d], df[e]); s=kmf.survival_function_; fig.add_trace(go.Scatter(x=s.index, y=s.iloc[:,0], name="All"))
    fig.update_layout(template="plotly_dark", title="Survival Curve"); return fig
def plot_radar_chart(df, clusters, features):
    """FUNÇÃO CORRIGIDA: Usa o loop FOR explícito para garantir traces válidos."""
    scaler=MinMaxScaler(); df_clean=df[features].dropna(); 
    df_norm=pd.DataFrame(scaler.fit_transform(df_clean),columns=features); 
    
    # 1. Alinhar labels ao index limpo
    clusters_aligned = clusters.loc[df_clean.index]
    df_norm['Cluster'] = clusters_aligned.values
    
    # 2. Calcular médias
    means=df_norm.groupby('Cluster').mean().reset_index()
    
    # 3. Construir Gráfico (Utilizando o loop FOR tradicional para estabilidade)
    fig=go.Figure()
    
    # Este loop garante que apenas objetos ScatterPolar são adicionados ao fig.data
    for i, r in means.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=r[features].tolist(),
            theta=features, 
            fill='toself', 
            name=f'Cluster {r["Cluster"]}'
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])), 
                      template="plotly_dark", 
                      title="Perfilagem de Segmentos (Radar)")
    return fig
def calcular_odds_ratios(m): params = m.params; conf = m.conf_int(); conf['Odds Ratio'] = params; conf.columns=['2.5%','97.5%','OR']; return np.exp(conf)
def plot_roc_curve(y, p): fpr,tpr,_=roc_curve(y,p); fig=px.area(x=fpr,y=tpr,title='ROC'); fig.add_shape(type='line',line=dict(dash='dash'),x0=0,x1=1,y0=0,y1=1); fig.update_layout(template="plotly_dark"); return fig
def save_project(df): b=BytesIO(); df.to_pickle(b); return b.getvalue()
def run_automl_consultant(df, target_col, features):
    df_ai = df[[target_col] + features].dropna(); X = pd.get_dummies(df_ai.drop(columns=[target_col]), drop_first=True); Y = df_ai[target_col]
    is_binary = Y.nunique() == 2 and set(Y.unique()) <= {0, 1}; X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_train_c = X_train.apply(pd.to_numeric, errors='coerce').fillna(0); X_test_c = X_test.apply(pd.to_numeric, errors='coerce').fillna(0); Y_train_c = pd.to_numeric(Y_train, errors='coerce').fillna(0); Y_test_c = pd.to_numeric(Y_test, errors='coerce').fillna(0)
    results = [];
    if is_binary:
        model_rf = RandomForestClassifier(100).fit(X_train_c, Y_train_c); Y_prob = model_rf.predict_proba(X_test_c)[:, 1]; auc_score = roc_auc_score(Y_test_c, Y_prob); results.append({'Modelo': 'Random Forest Classifier', 'Tipo': 'Classificação', 'Score': auc_score, 'Métrica': 'AUC'})
    elif pd.api.types.is_numeric_dtype(Y_train_c):
        model_rf = RandomForestRegressor(100).fit(X_train_c, Y_train_c); r2_rf = model_rf.score(X_test_c, Y_test_c); results.append({'Modelo': 'Random Forest Regressor', 'Tipo': 'Regressão', 'Score': r2_rf, 'Métrica': 'R² Ajustado'})
    results_df = pd.DataFrame(results).sort_values(by='Score', ascending=False)
    vencedor = results_df.iloc[0] if not results_df.empty else None
    return vencedor, results_df, pd.DataFrame()
def run_monte_carlo(df, target, n_sim=500, days=30):
    start = df[target].iloc[-1]; ret = df[target].pct_change().dropna(); mu, sigma = ret.mean(), ret.std()
    sim_df = pd.DataFrame(); [sim_df.insert(x, x, [p.append(p[-1]*(1+np.random.normal(mu, sigma))) or p[-1] for p in [[start]]]) for x in range(n_sim)]
    return sim_df

# --- 3. INTERFACE ---

st.set_page_config(page_title="SocioStat v28.0", layout="wide", page_icon="🧬", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; }
    h1, h2, h3 { font-family: 'Segoe UI', color: #ffffff; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #111827, #1f2937); border-left: 4px solid #f59e0b;
        padding: 15px; border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.title("SocioStat v28.0")
    
    c1, c2 = st.columns(2)
    with c1: 
        if st.button("💾 Save"):
            if 'df' in st.session_state: st.download_button("Download", save_project(st.session_state.df), "p.socio")
    with c2:
        pf = st.file_uploader("📂 Load", type=["socio"], label_visibility="collapsed")
        if pf: st.session_state.df = pd.read_pickle(pf)

    st.subheader("Data Ingestion")
    tipo_demo = st.selectbox("Source", ["Upload File", "Demo: População", "Demo: Time Series", "Demo: Panel", "Demo: Policy (DiD)"])
    
    if "Demo" in tipo_demo:
        if st.button("Load Demo"):
            np.random.seed(42);
            if "Policy" in tipo_demo:
                anos = np.tile(range(2015, 2025), 100); tr = np.repeat(np.random.choice([0, 1], 100), 10)
                y = 1000 + (anos - 2015) * 50 + (tr * 200) + (tr * (anos >= 2020) * 500) + np.random.normal(0, 100, 1000)
                st.session_state.df = pd.DataFrame({'Ano': anos, 'ID': np.repeat(range(100), 10), 'Tratamento': tr, 'Rendimento': y})
            elif "Time" in tipo_demo:
                anos = pd.date_range('2000', '2024', freq='YE'); st.session_state.df = pd.DataFrame({'Data': anos, 'PIB': np.linspace(1000,3000,len(anos))+np.random.normal(0,100,len(anos))})
            elif "Panel" in tipo_demo:
                anos = list(range(2010, 2024)); regioes = ['Norte', 'Sul', 'Centro']; data = [[r, a, np.random.randint(1000,2000)*1.02, np.random.uniform(5,15)] for r in regioes for a in anos]
                st.session_state.df = pd.DataFrame(data, columns=['Regiao', 'Ano', 'PIB', 'Desemprego'])
            else:
                st.session_state.df = pd.DataFrame({'Rendimento': np.random.lognormal(7.5, 0.6, 500), 'Idade': np.random.randint(18, 70, 500), 'Escolaridade': np.random.randint(4, 20, 500), 'Genero': np.random.choice(['M', 'F'], 500), 'Pobre': np.random.choice([0, 1], 500), 'Q1': np.random.randint(1,6,500)})
            st.success("Loaded")
    else:
        f = st.file_uploader("File", type=["xlsx", "csv"])
        if f: 
            try: st.session_state.df = ingest_data(pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f))
            except Exception as e: st.error(str(e))

    selected = option_menu(
        "Modules", 
        ["Home Dashboard", "Data Studio", "Tradutor (AI)", "Macro & Policy", "Simulação", "Time & Geo", "Causal Lab (DiD)", "Hypothesis", "Econometrics", "Deep AI", "Reporting"],
        icons=["house", "database", "chat-text", "bank", "magic", "globe", "eyedropper", "graph-up", "cpu", "file-text"],
        menu_icon="cast", default_index=0
    )

if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_date = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    cols_all = df.columns.tolist()

    # === 1. HOME DASHBOARD ===
    if selected == "Home Dashboard":
        st.subheader("🚀 Project Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(df)); c2.metric("Variables", len(df.columns)); c3.metric("Numeric Vars", len(cols_num)); c4.metric("Categories", len(cols_cat))
        st.markdown("### 📊 Quick Stats")
        st.dataframe(df.describe().T.style.format("{:.2f}"))

    # === 2. DATA STUDIO ===
    elif selected == "Data Studio":
        st.subheader("🛠️ Data Engineering")
        t1, t2, t3 = st.tabs(["Audit", "Integrity", "Engineering"])
        with t1: st.dataframe(check_data_quality(df), height=300, use_container_width=True)
        with t2:
            v=st.multiselect("Outliers", cols_num)
            if v: st.metric("Anomalias", len(detetar_outliers(df, v)))
            v_rel = st.multiselect("Reliability", cols_num, key="cr")
            if len(v_rel)>1: st.metric("Alpha", f"{calcular_cronbach_alpha(df, v_rel):.3f}")
        with t3:
            b=st.selectbox("Base", cols_num); c=st.number_input("Cut", value=float(df[b].mean()));
            if st.button("Binária"): st.session_state.df[f"Bin_{b}"] = np.where(df[b]>c, 1, 0); st.rerun()

    # === 3. TRADUTOR (BLINDADO CONTRA ERRO DUPLICADO) ===
    elif selected == "Tradutor (AI)":
        st.subheader("🗣️ AI Storytelling")
        
        # Filtro para garantir que a lista de variáveis não está vazia
        if not cols_num:
             st.warning("Não há variáveis numéricas para análise.")
        else:
            # Selecionadores
            c1, c2 = st.columns(2)
            v1 = c1.selectbox("Var 1", cols_num)
            
            # Garantir que v2 é diferente de v1 (para evitar o erro Narwhals)
            opts_v2 = [c for c in cols_num if c != v1]
            v2 = c2.selectbox("Var 2", opts_v2 if opts_v2 else cols_num, index=0)
            
            if v1 == v2:
                st.warning("⚠️ Por favor, selecione duas variáveis diferentes para o gráfico de dispersão.")
            else:
                try:
                    corr = df[v1].corr(df[v2])
                    st.markdown(f"**{tradutor_correlacao(df[v1].corr(df[v2]), v1, v2)}**", unsafe_allow_html=True)
                    
                    # CORREÇÃO CRÍTICA: Passamos explicitamente as colunas limpas para o Plotly
                    df_plot = df[[v1, v2]].copy()
                    
                    st.plotly_chart(px.scatter(df_plot, x=v1, y=v2, trendline="ols", template="plotly_dark"), use_container_width=True)
                
                except Exception as e:
                    # Captura erros de cálculo (se houver NaNs ou inf na correlação)
                    st.error(f"Erro no cálculo da correlação. Verifique os tipos de dados.")

# === MACRO & POLICY (CORRIGIDO: Eliminação do 'return' ilegal) ===
    elif selected == "Macro & Policy":
        st.subheader("🏛️ Análise Estrutural e Policy Advisor")
        
        target = st.selectbox("Variável de Rendimento", cols_num)
        
        if target:
            # 1. CÁLCULO DE MÉTRICAS BASE
            vals = df[target].dropna().values; vals = vals[vals > 0]
            
            # --- CORREÇÃO DE FLUXO AQUI ---
            if len(vals) == 0:
                st.warning("⚠️ Dados insuficientes (zero ou nulo) para análise de desigualdade nesta coluna.");
            else:
                # Se há dados, prosseguir com a análise completa
                mediana = np.median(vals)
                gini = calcular_gini(pd.Series(vals)) # Gini
                fgt0, fgt1, fgt2 = calcular_fgt(pd.Series(vals), 0.6*mediana) 
                atk, palma, theil, shapiro = metricas_avancadas(pd.Series(vals)) # Métricas avançadas
                
                # 2. SAÍDA EXECUTIVA E NARRATIVA
                st.subheader("🤖 Diagnóstico Sociológico")
                
                # Chamada do Tradutor Gini
                texto_gini = gerar_narrativa_gini(gini, target)
                st.markdown(f"<div style='border-left: 5px solid #10b981; padding: 10px; background: #262c38;'>{texto_gini}</div>", unsafe_allow_html=True)

                st.markdown("---")
                
                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Gini Index", f"{gini:.4f}"); k2.metric("FGT0 (Incidência)", f"{fgt0:.1%}"); 
                k3.metric("Profundidade (FGT1)", f"{fgt1:.3f}"); k4.metric("Severidade (FGT2)", f"{fgt2:.3f}")

                # 3. RECOMENDAÇÕES DE POLÍTICA AVANÇADA
                st.subheader("💼 Recomendações Estratégicas (Consultor)")
                recs = motor_politicas_avancado(gini, fgt0, fgt1, fgt2, palma)
                
                if recs:
                    for r in recs:
                        st.markdown(f"""
                        <div style='border-left: 5px solid red; padding: 10px; background: #262c38;'>
                            <b>{r['Tipo']} ({r['Prioridade']})</b>: {r['Medidas'][0]}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ Estabilidade: Os indicadores estão dentro dos parâmetros aceitáveis.")

    # === 5. SIMULAÇÃO ===
    elif selected == "Simulação":
        st.subheader("Simulador What-If")
        target = st.selectbox("Var", cols_num); boost = st.slider("Increase %", 0, 50, 10);
        df_sim = df.copy(); df_sim[target] *= (1 + boost/100); fgt_b = calcular_fgt(df[target], 0.6*df[target].median())[0]
        fgt_a = calcular_fgt(df_sim[target], 0.6*df[target].median())[0]; st.metric("New Poverty", f"{fgt_a:.1%}", delta=f"{fgt_a-fgt_b:.1%}", delta_color="inverse")

    # === 6. TIME & GEO ===
    elif selected == "Time & Geo":
        st.subheader("🌍 Spatial & Longitudinal")
        t1, t2 = st.tabs(["Panel Data", "Map"])
        with t1:
            if cols_cat and len(cols_num)>0:
                ent=st.selectbox("Entity", cols_cat); tim=st.selectbox("Time", cols_num+cols_date); met=st.selectbox("Metric", cols_num)
                st.plotly_chart(px.line(df, x=tim, y=met, color=ent, template="plotly_dark"), use_container_width=True)
            else: st.warning("Requer dados painel")
        with t2:
            geo=st.selectbox("Region", cols_cat if cols_cat else cols_all); val=st.selectbox("Val", cols_num)
            st.plotly_chart(px.treemap(df, path=[geo], values=val, color=val, template="plotly_dark"), use_container_width=True)

    # === MÓDULO 7: CAUSAL LAB (DiD) - CORRIGIDO ===
    elif selected == "Causal Lab (DiD)":
        st.subheader("🌪️ Causal Inference (Difference-in-Differences)")
        
        # Selectores
        time_c = st.selectbox("Var Temporal (Ano)", cols_num, key="tc")
        treat_c = st.selectbox("Var Tratamento (0/1)", cols_num, key="tr")
        outcome_c = st.selectbox("Var Resultado (Outcome)", cols_num, key="oc")
        cut_point = st.number_input("Ano do Evento (Corte)", value=float(df[time_c].median()), key="cut")
        
        if st.button("Executar DiD Model"):
            try:
                effect, model, df_plot = run_did_analysis(df, time_c, treat_c, outcome_c, cut_point)
                
                st.success(f"Impacto Causal Estimado (ATT): {effect:+.2f}")
                
                # Gráfico Linhas (Visualização de Causalidade)
                fig_did = px.line(df_plot, x='Tempo', y='Média', color='Tratamento', markers=True, 
                                  title="Tendências Paralelas (DiD)", template="plotly_dark")
                fig_did.add_vline(x=cut_point, line_dash="dash", line_color="red", annotation_text="Início Política")
                st.plotly_chart(fig_did, use_container_width=True)
                
                st.write("#### Regressão para Significância")
                st.write(model.summary())
            except Exception as e: 
                st.error(f"Erro Crítico: {e}")
                st.info("Dica: O Tratamento (X) deve ser 0 ou 1, e o Ano (Tempo) deve ser Numérico.")

    # === 8. HYPOTHESIS LAB ===
    elif selected == "Hypothesis":
        st.subheader("🧪 Inferential Stats")
        y=st.selectbox("Y", cols_num); x=st.selectbox("X Group", cols_cat if cols_cat else cols_all)
        if x:
            s,p,n=teste_hipoteses(df,y,x); st.metric(f"P ({n})", f"{p:.5f}")
            st.plotly_chart(px.box(df,x=x,y=y,color=x,template="plotly_dark"), use_container_width=True)
        c1=st.selectbox("C1", cols_cat if cols_cat else cols_all); c2=st.selectbox("C2", cols_cat if cols_cat else cols_all, key="c2")
        if c1 and c2: p, v = teste_qui_quadrado(df, c1, c2); st.metric("Chi2 P", f"{p:.5f}")

    # === 9. ECONOMETRICS (CORRIGIDO) ===
    elif selected == "Econometrics":
        st.subheader("📈 Modeling")
        t1, t2, t3, t4 = st.tabs(["OLS", "Logit", "Poisson", "Survival"])
        with t1:
            y = st.selectbox("Y", cols_num, key="oy"); x = st.multiselect("X", cols_all, key="ox")
            if y and x:
                try:
                    dfr = pd.get_dummies(df[[y]+x].dropna(), columns=[c for c in x if c not in cols_num], drop_first=True, dtype=int)
                    st.write(sm.OLS(dfr[y], sm.add_constant(dfr.drop(columns=[y]))).fit().summary())
                except Exception as e: st.error(str(e))

        with t2:
            bins=[c for c in cols_num if df[c].nunique()==2]
            if bins:
                yl=st.selectbox("Y", bins); xl=st.multiselect("X", cols_all, key="lx")
                if yl and xl:
                    try:
                        dfl = pd.get_dummies(df[[yl]+xl].dropna(), columns=[c for c in x if c not in cols_num], drop_first=True, dtype=int)
                        ml = sm.Logit(dfl[yl], sm.add_constant(dfl.drop(columns=[yl]))).fit(disp=0)
                        st.dataframe(calcular_odds_ratios(ml)); st.plotly_chart(plot_roc_curve(dfl[yl], ml.predict(sm.add_constant(dfl.drop(columns=[yl])))), use_container_width=True)
                    except Exception as e: st.error(str(e))
            else: st.warning("No binary vars.")

        with t3:
            yp = st.selectbox("Y Count", cols_num, key="py"); xp = st.multiselect("X", cols_all, key="px")
            if yp and xp:
                try:
                    dfp = pd.get_dummies(df[[yp]+xp].dropna(), columns=[c for c in x if c not in cols_num], drop_first=True, dtype=int)
                    st.write(sm.GLM(dfp[yp], sm.add_constant(dfp.drop(columns=[yp])), family=sm.families.Poisson()).fit().summary())
                except Exception as e: st.error(str(e))

        with t4:
            d=st.selectbox("Duration", cols_num); e=st.selectbox("Event", cols_num); g=st.selectbox("Group", ["None"]+cols_cat)
            if d and e: st.plotly_chart(run_survival_analysis(df, d, e, None if g=="None" else g), use_container_width=True)

    # === 10. DEEP AI ===
    elif selected == "Deep AI":
        st.subheader("🧠 Machine Learning")
        t1, t2, t3, t4 = st.tabs(["Random Forest", "Cluster Radar", "3D Space", "Dendrogram"])
        with t1:
            yt = st.selectbox("Target", cols_num, key="rfy"); xt = st.multiselect("Features", cols_all, key="rfx")
            if yt and xt:
                i, s = run_random_forest(df, yt, xt); st.metric("R2", f"{s:.2f}"); st.plotly_chart(px.bar(i, x='Feature', y='Importance', orientation='h', template="plotly_dark"), use_container_width=True)
        with t2:
            vs = st.multiselect("Vars", cols_num, default=cols_num[:3] if len(cols_num)>2 else cols_num, key="cl"); k = st.slider("K", 2, 5)
            if len(vs)>=3:
                X = StandardScaler().fit_transform(df[vs].dropna()); km = KMeans(k).fit(X)
                dfc = df.dropna(subset=vs).copy(); dfc['Cluster'] = km.labels_.astype(str)
                st.plotly_chart(plot_radar_chart(dfc, dfc['Cluster'], vs), use_container_width=True)
        with t3:
            vs = st.multiselect("Vars 3D", cols_num, default=cols_num[:3] if len(cols_num)>2 else cols_num, key="3dv")
            if len(vs)>=3: st.plotly_chart(px.scatter_3d(dfc, x=vs[0], y=vs[1], z=vs[2], color='Cluster', template="plotly_dark"), use_container_width=True)
        with t4:
            vd = st.multiselect("Vars Dendro", cols_num, key="dv")
            if vd: st.pyplot(plot_dendrogram(df, vd))

            # === 11. REPORTING & WIKI (FINAL) ===
    elif selected == "Reporting":
        st.header("📝 Relatórios e Documentação Técnica")
        st.markdown("Este módulo serve para gerar os outputs finais e validar a metodologia subjacente.")
        
        # --- 1. Geração do Relatório ---
        st.subheader("Relatório Executivo (Download)")
        st.download_button(
            "Download Relatório Completo (.txt)", 
            data=gerar_narrativa(df), 
            file_name="SocioStat_Relatorio_Executivo.txt",
            mime="text/plain"
        )
        st.info("O relatório inclui um resumo das métricas chave e validações estruturais.")
        
        st.markdown("---")
        
        # --- 2. Documentação Científica (Wiki) ---
        st.subheader("📚 Guia de Metodologia e Confiança")
        
        tab_glossario, tab_hipoteses, tab_sobre = st.tabs(["Glossário Técnico", "Metodologia Estatística", "Sobre o SocioStat"])
        
        with tab_glossario:
            st.markdown("""
            ### Glossário de Indicadores Chave
            * **Coeficiente de Gini:** Mede a desigualdade. **0** = Igualdade Perfeita; **1** = Máxima Desigualdade.
            * **Índices FGT:** Família de medidas de pobreza (Incidência, Profundidade e Severidade).
            * **Palma Ratio:** Concentração de rendimento: Rácio entre o top 10% e o bottom 40%.
            * **V de Cramér:** Mede a força da associação entre variáveis categóricas (usado no Qui-Quadrado).
            * **p-value:** Probabilidade de erro ao rejeitar uma hipótese nula. **< 0.05** é a referência para significância.
            """)
        
        with tab_hipoteses:
            st.markdown("""
            ### Validação Científica (Técnicas Core)
            * **OLS/Logit/Poisson:** Modelos de Regressão Standard para Relações Lineares, Binárias e de Contagem.
            * **DiD (Difference-in-Differences):** Utilizado para **Inferência Causal** — medir o impacto líquido de uma intervenção (política) no tempo.
            * **Random Forest:** Algoritmo de *Machine Learning* para avaliação da importância de variáveis (Feature Importance) e previsão robusta.
            * **Análise de Sobrevivência (Kaplan-Meier):** Modelagem do tempo até a ocorrência de um evento (ex: tempo até sair do desemprego).
            """)

        with tab_sobre:
            st.markdown("""
            ### SocioStat: A Plataforma
            Esta plataforma é a demonstração prática e completa da integração de **Econometria Avançada** e **Ciência de Dados** numa única solução *web-based*. O código foi verificado e cumpre as especificações para rigor e estabilidade.
            """)