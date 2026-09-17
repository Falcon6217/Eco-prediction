from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title='Ecosystem Health Predictor',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='auto',
    menu_items={'About': 'Ecosystem Health Predictor — responsive ML dashboard.'}
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / 'ecosystem_data.csv'
MODEL_PATH = BASE / 'ecosystem.pkl'
FEATURES = ['water_quality','air_quality_index','biodiversity_index','vegetation_cover','soil_ph']
LABELS = {0:'Healthy',1:'At Risk',2:'Degraded'}
COLORS = {'Healthy':'#22c55e','At Risk':'#f59e0b','Degraded':'#ef4444'}

st.markdown('''
<style>
:root{--bg:#071a12;--card:rgba(255,255,255,.075);--line:rgba(255,255,255,.13);--text:#edfdf3;--muted:#b9d8c6;}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 10% 0%,#124d31 0,#071a12 38%,#04110b 100%);}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1450px;padding:1.2rem clamp(.8rem,2vw,2.5rem) 3rem;}
.hero{padding:clamp(20px,4vw,42px);border:1px solid var(--line);border-radius:26px;background:linear-gradient(135deg,rgba(34,197,94,.22),rgba(14,116,144,.18));box-shadow:0 18px 50px rgba(0,0,0,.25);animation:rise .65s ease-out;}
.hero h1{font-size:clamp(2rem,5vw,3.5rem);margin:0;color:#f0fff5!important;line-height:1.05;}
.hero p{color:var(--muted)!important;font-size:clamp(.95rem,2vw,1.12rem);margin:.8rem 0 0;max-width:850px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:18px;backdrop-filter:blur(10px);}
.metric{background:linear-gradient(135deg,rgba(255,255,255,.09),rgba(255,255,255,.035));border:1px solid var(--line);border-radius:18px;padding:16px;text-align:center;min-height:105px;}
.metric .value{font-size:clamp(1.35rem,3vw,2rem);font-weight:800;color:#f0fff5;}
.metric .label{color:var(--muted);font-size:.9rem;}
.result{border-radius:20px;padding:24px;text-align:center;border:1px solid rgba(255,255,255,.18);animation:pop .45s ease-out;}
.result h2{margin:.2rem 0;color:#fff!important;font-size:clamp(1.4rem,3vw,2.1rem);}
.result p{color:#eafff0!important;margin:.2rem 0 0;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#06170e,#0a2518);border-right:1px solid var(--line);}
.stButton>button{width:100%;min-height:46px;border-radius:13px;font-weight:800;border:0;background:linear-gradient(90deg,#22c55e,#14b8a6);color:#03130b;transition:transform .18s,box-shadow .18s;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 10px 25px rgba(34,197,94,.25);}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);padding:14px;border-radius:16px;}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes pop{from{opacity:0;transform:scale(.98)}to{opacity:1;transform:scale(1)}}
@media(max-width:700px){.block-container{padding-top:.7rem}.hero{border-radius:18px}.card{padding:14px}}
</style>
''', unsafe_allow_html=True)

st.markdown('''<div class="hero"><h1>🌿 Ecosystem Health Predictor</h1><p>Enter environmental measurements and use a machine-learning model to classify ecosystem condition as <b>Healthy</b>, <b>At Risk</b>, or <b>Degraded</b>.</p></div>''', unsafe_allow_html=True)
st.write('')

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'{DATA_PATH.name} not found.')
    df=pd.read_csv(DATA_PATH)
    df.columns=[str(c).strip().lower() for c in df.columns]
    required=FEATURES+['ecosystem_health']
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Dataset is missing: {missing}')
    for c in FEATURES: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['ecosystem_health']=df['ecosystem_health'].astype(str).str.strip().str.lower()
    df=df.dropna(subset=required)
    df=df[df['ecosystem_health'].isin(['healthy','at risk','degraded'])].copy()
    df['label']=df['ecosystem_health'].map({'healthy':'Healthy','at risk':'At Risk','degraded':'Degraded'})
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'{MODEL_PATH.name} not found. Run: python train_model.py')
    return joblib.load(MODEL_PATH)

try:
    df=load_data(); model=load_model()
except Exception as e:
    st.error(f'Project setup error: {e}')
    st.stop()

healthy=(df.label=='Healthy').mean()*100
risk=(df.label=='At Risk').mean()*100
degraded=(df.label=='Degraded').mean()*100
m1,m2,m3,m4=st.columns(4, gap='medium')
for col,title,value in [(m1,'Dataset Records',f'{len(df):,}'),(m2,'Healthy',f'{healthy:.1f}%'),(m3,'At Risk',f'{risk:.1f}%'),(m4,'Degraded',f'{degraded:.1f}%')]:
    col.markdown(f'<div class="metric"><div class="label">{title}</div><div class="value">{value}</div></div>',unsafe_allow_html=True)

st.write('')
with st.sidebar:
    st.markdown('## 🎛️ Environmental Inputs')
    st.caption('Adjust the readings and click Predict.')
    values={}
    for c,icon,label in [
        ('water_quality','💧','Water Quality'),
        ('air_quality_index','🌫️','Air Quality Index'),
        ('biodiversity_index','🦋','Biodiversity Index'),
        ('vegetation_cover','🌳','Vegetation Cover (%)'),
        ('soil_ph','🧪','Soil pH')]:
        lo=float(df[c].min()); hi=float(df[c].max()); default=float(df[c].median())
        step=0.01 if c=='biodiversity_index' else 0.1
        values[c]=st.slider(f'{icon} {label}',lo,hi,default,step=step)
    predict=st.button('🔮 Predict Ecosystem Health',type='primary')
    st.markdown('---')
    st.caption('Model: Gaussian Naive Bayes pipeline')

input_df=pd.DataFrame([values],columns=FEATURES)
tab1,tab2,tab3=st.tabs(['🔮 Prediction','📊 Analytics','📋 Dataset'])

with tab1:
    left,right=st.columns([1,1],gap='large')
    with left:
        st.markdown('### Current ecosystem profile')
        st.dataframe(input_df.T.rename(columns={0:'Value'}),use_container_width=True,height=260)
        # Normalize indicators to a common 0–100 visual scale.
        radar=[]
        for c in FEATURES:
            mn=float(df[c].min()); mx=float(df[c].max()); x=values[c]
            radar.append(50 if mx==mn else (x-mn)/(mx-mn)*100)
        fig=go.Figure(go.Scatterpolar(r=radar+[radar[0]],theta=['Water','Air','Biodiversity','Vegetation','Soil pH','Water'],fill='toself',line=dict(color='#22c55e',width=3),fillcolor='rgba(34,197,94,.18)'))
        fig.update_layout(height=380,margin=dict(l=25,r=25,t=25,b=25),paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#dff7e8'),polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(range=[0,100],showticklabels=False,gridcolor='rgba(255,255,255,.12)'),angularaxis=dict(gridcolor='rgba(255,255,255,.12)')))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False,'responsive':True})
    with right:
        st.markdown('### Prediction result')
        if predict:
            pred=int(model.predict(input_df)[0]); label=LABELS.get(pred,str(pred))
            probs=model.predict_proba(input_df)[0] if hasattr(model,'predict_proba') else np.array([0,0,0])
            classes=getattr(model,'classes_',None)
            if classes is None and hasattr(model,'named_steps'): classes=model.named_steps.get('classifier').classes_
            prob_map={LABELS.get(int(cls),str(cls)):float(p) for cls,p in zip(classes,probs)}
            confidence=prob_map.get(label,0)*100
            bg={'Healthy':'linear-gradient(135deg,#15803d,#22c55e)','At Risk':'linear-gradient(135deg,#b45309,#f59e0b)','Degraded':'linear-gradient(135deg,#991b1b,#ef4444)'}[label]
            icon={'Healthy':'🌿','At Risk':'⚠️','Degraded':'🚨'}[label]
            st.markdown(f'<div class="result" style="background:{bg}"><div style="font-size:2.5rem">{icon}</div><h2>{label} Ecosystem</h2><p>Model confidence: <b>{confidence:.1f}%</b></p></div>',unsafe_allow_html=True)
            if label=='Healthy': st.success('The supplied readings fall into the Healthy class according to this model.')
            elif label=='At Risk': st.warning('The supplied readings fall into the At Risk class according to this model.')
            else: st.error('The supplied readings fall into the Degraded class according to this model.')
            prob_df=pd.DataFrame({'Class':list(prob_map.keys()),'Probability':[v*100 for v in prob_map.values()]})
            pf=px.bar(prob_df,x='Class',y='Probability',color='Class',text=prob_df.Probability.map(lambda x:f'{x:.1f}%'),color_discrete_map=COLORS)
            pf.update_layout(height=330,margin=dict(l=10,r=10,t=25,b=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#dff7e8',yaxis=dict(range=[0,100],title='Probability (%)'),xaxis_title=None,showlegend=False)
            st.plotly_chart(pf,use_container_width=True,config={'displayModeBar':False,'responsive':True})
        else:
            st.info('Set the environmental readings and click **Predict Ecosystem Health** to see the result.')
            st.markdown('<div class="card"><b>How it works</b><br><br>1. Read environmental indicators<br>2. Send them to the trained ML pipeline<br>3. Get the predicted ecosystem class<br>4. Review class probabilities</div>',unsafe_allow_html=True)

with tab2:
    a,b=st.columns(2,gap='large')
    with a:
        counts=df['label'].value_counts().reindex(['Healthy','At Risk','Degraded']).fillna(0).reset_index()
        counts.columns=['Class','Records']
        fig=px.pie(counts,names='Class',values='Records',hole=.58,color='Class',color_discrete_map=COLORS,title='Ecosystem class distribution')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',font_color='#dff7e8',legend_title_text='')
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False,'responsive':True})
    with b:
        feature=st.selectbox('Explore feature',FEATURES,format_func=lambda x:x.replace('_',' ').title())
        fig=px.histogram(df,x=feature,color='label',nbins=35,color_discrete_map=COLORS,title=f'{feature.replace("_"," ").title()} distribution')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#dff7e8',legend_title_text='')
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False,'responsive':True})

with tab3:
    st.caption(f'Showing {len(df):,} validated records. Use the controls below to inspect the dataset.')
    search=st.text_input('🔎 Filter ecosystem class',placeholder='healthy, at risk, or degraded')
    view=df if not search else df[df['label'].str.contains(search,case=False,na=False)]
    st.dataframe(view[FEATURES+['label']].head(1000),use_container_width=True,height=430)

st.markdown('---')
st.caption('🌱 Educational ML project. Predictions are model outputs and should not be treated as environmental certification or professional ecological assessment.')
