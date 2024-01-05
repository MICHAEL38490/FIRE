import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler,RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import f1_score, confusion_matrix,classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from PIL import Image
import base64

# os.chdir(r"C:\Users\mbobe\Desktop\Dossier APEC-DATA analyst\FORMATION DATA\FIRE")

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# @st.cache_data
# def load_csv(file_path):
#     return pd.read_csv(file_path)

wildfire_head=pd.read_csv("wildfire_head.csv")
wildfire_describe=pd.read_csv("wildfire_describe.csv")
wildfire_isna= pd.read_csv("wildfire_isna.csv")
df_fire_isna = pd.read_csv("df_fire_isna.csv")
df_fire_head = pd.read_csv("df_fire_head.csv")
df_duration_head = pd.read_csv("df_duration_head.csv")
df_duration_isna = pd.read_csv("df_duration_isna.csv")



# df_wild=load_csv(r"C:\Users\mbobe\Documents\DATAANALYSE\FIRE\wildfires.csv")
# df_duration=load_csv(r"C:\Users\mbobe\Documents\DATAANALYSE\FIRE\duration_dataviz.csv")
# df_fire=load_csv(r"C:\Users\mbobe\Documents\DATAANALYSE\FIRE\fire_dataviz.csv")

# Ouest=df_fire.loc[df_fire['cluster']==1]
# Sud_Est=df_fire.loc[df_fire['cluster']==4]
# Sud=df_fire.loc[df_fire['cluster']==3]
# Alaska=df_fire.loc[df_fire['cluster']==0]
# Nouvelle_Angleterre=df_fire.loc[df_fire['cluster']==5]
# MidWest=df_fire.loc[df_fire['cluster']==2]
  


# Insérer votre chemin d'image
# image_path = r"C:\Users\
# mbobe\Desktop\Dossier APEC-DATA analyst\FORMATION DATA\FIRE\FIRE_SIZE\feu.jpg"
image_path = "feu.jpg"
# # Insérer le code HTML avec la balise img pour l'image
banner_html = f"""
<div style="text-align: center; padding: 10px;">
    <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" alt="Image" style="max-width: 100%;">
</div>
"""
# Afficher la bannière HTML
st.markdown(banner_html, unsafe_allow_html=True)


st.title("Projet de classification de feux de forêts")
st.sidebar.title("Sommaire")
pages=["Introduction et Exploration jeu de données", "DataVizualisation", "Modélisation", "Optimisation", "Test de prédiction", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.title("Auteurs")
st.sidebar.markdown("**FONKOU Gabriella** - [LinkedIn](www.linkedin.com/in/gabriella-fonkou-40bb61120)")
st.sidebar.markdown("**RANVOISY Yann** - [LinkedIn](www.linkedin.com/in/yannranvoisy)")
st.sidebar.markdown("**BOBEUF Michaël** - [LinkedIn](www.linkedin.com/in/michaël-bobeuf)")


#---------------------------- PAGE INTRO OK --------------------------------------------------------------------------------------------------------------------------
if page == pages[0]:
   
    # df_duration=load_csv(r"C:\Users\mbobe\Desktop\Dossier APEC-DATA analyst\FORMATION DATA\FIRE\Duration\duration_dataviz.csv")
    # df_fire=load_csv(r"C:\Users\mbobe\Desktop\Dossier APEC-DATA analyst\FORMATION DATA\FIRE\FIRE_SIZE\fire_dataviz.csv")

    st.write("### Introduction")
    
    st.markdown("""
    <div style="text-align: justify;">
    Notre projet contient des données spatiales sur les incendies de forêt survenus aux États-Unis de 1992 à 2015.
    Les enregistrements ont été acquis à partir des systèmes de reporting des organisations fédérales, étatiques et locales de
    lutte contre les incendies. Les données ont été transformées pour se conformer, 
    lorsque cela était possible, aux normes de données du National Wildfire Coordinating Group (NWCG). La base de données comprend
    1,88 millions d'enregistrements géoréférencés, représentant un total de 140 millions d'acres brûlés entre 1992 et 2015.


    Les incendies de forêt aux États-Unis représentent une menace croissante et récurrente, posant des défis significatifs tant sur le plan environnemental
    que socio-économique.L’enjeu central de notre projet vise à présenter une analyse approfondie de feux d’incendies majeurs survenus aux États-Unis de 1992 à 2015.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: justify;">
    A travers cette analyse, plusieurs interrogations seront abordées :

    <ul>
    <li><strong>Quels sont les paramètres pouvant faire varier la taille et la durée d'un feu?</strong></li>
    <li><strong>Quelles sont les régions les plus sensibles aux incendies ?</strong></li>
    <li><strong>Pourrons-nous prédire la taille et la durée d’un incendie en fonction de divers paramètres?</strong></li>
    </ul>

    L’objectif de notre projet est donc d’utiliser au mieux les données spécifiques disponibles pour répondre à  ces interrogations.
    Nous serons en mesure de mettre en lumière les facteurs contributifs, les conséquences et les leçons apprises pour renforcer les mesures
    préventives et la gestion future de ces catastrophes.
    </div>
        """, unsafe_allow_html=True)

    st.write("### Dataset")
    st.markdown("""
    <div style="text-align: justify;">
        <ul><strong>SOURCE: </strong><a href="https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires">1.88 Million US Wildfires</a></ul>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(wildfire_head)
   
    st.markdown("""
    <div style="text-align: justify;">
    <ul>Le dataset principal contient 1 880 465 entrées et 39 variables.</ul>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(wildfire_describe)

    if st.checkbox("Afficher les NA"):
        st.dataframe(wildfire_isna)
    st.markdown("""
    <div style="text-align: justify;">
    <ul>Certaines variables contiennent plus de 50% de valeurs manquantes. Nous en supprimerons quelques unes, et traiterons les NaN le cas échéant.</ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("### Datasets secondaires")
    st.markdown("""
    <div style="text-align: justify;">
    <ul><strong>DF_FIRE</strong></ul>
    Ce dataset nous permettra d'analyser la superficie (FIRE_SIZE_CLASS) en fonction de différents paramètres.
    Il est obtenu par:
    <li>Conversion et création de variables temporelles (Année, heures, minutes)</li>
    <li>Création de six clusters géographiques qui nous permettront d'analyser les incendies selon leur géolocalisation</li>
    <li>Suppression de variables identifiantes redondantes et de variables temporelles dont NaN>50%</li>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_fire_head)
    
    st.write("DF_FIRE contient 1 876 954 entrées (nous avons supprimé 3511 doublons (0.18%) générés par la suppression\
            de variables identifiantes, seules variables distinctrices de feux identiques). DF_FIRE contient 19 variables.")
    if st.checkbox("Afficher les  NA"):
        st.dataframe(df_fire_isna)
        st.write("Nous conservons les NaNs de quelques variables, qui seront traités lors de la préparation au Machine Learning.")

    st.markdown("""
    <div style="text-align: justify;">
    <ul><strong>DF_DURATION</strong></ul>
    Ce dataset a pour objectif d'analyser la durée des feux en fonction de différents paramètres.
    Il est obtenu par:
    <li>Suppression des NaNs pour chaque ligne en contenant parmi les variables temporelles</li>
    <li>Création de la variable DURATION_IN_HOURS nous renseignant sur la durée des feux</li>
    <li>Conversion logarithmique de DURATION_IN_HOUR pour ensuite la classifier selon des plages de valeurs plus restreintes</li>
    <li>Restriction des valeurs à la limite max (q3+1.5*IQ) pour s'affranchir des outliers</li>
    <li>Classification de DURATION_LOG pour enfin créer trois classes majeures de DURATION_CLASS: 0-1h, 1h-3h15, 3h15-6j</li>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_duration_head)
    
    st.write("DF_DURATION contient 751 697 entrées et 30 variables.")
    if st.checkbox("Afficher  les NA"):
        st.dataframe(df_duration_isna)  
        


# DEFINITION DES FONCTIONS MISES EN CACHE POUR GENERER LES GRAPHIQUES PLUS RAPIDEMENT     

# @st.cache_data
# def generate_etat_chart(df_fire,chart_choice):
#     if chart_choice == 'US clusters':
#         fig=px.scatter(data_frame=df_fire,x='LONGITUDE',y='LATITUDE', color='cluster',color_continuous_scale='ORRD')   #code pour afficher en plotly
#         # plt.figure(figsize=(15,7))
#         # sns.scatterplot(x=df_fire['LONGITUDE'],y=df_fire['LATITUDE'], hue=df_fire['cluster'])
#         # plt.legend(title='Clusters')
       



if page == pages[1]:
    # df_duration=load_csv(r"C:\Users\mbobe\Desktop\Dossier APEC-DATA analyst\FORMATION DATA\FIRE\Duration\duration_dataviz.csv")
    # df_fire=load_csv(r"C:\Users\mbobe\Desktop\Dossier APEC-DATA analyst\FORMATION DATA\FIRE\FIRE_SIZE\fire_dataviz.csv")
        
# AFFICHAGE DES TARGETS
    st.write("### DataVizualisation")
    if st.checkbox("Afficher les graphiques Targets"):
        image = Image.open("piecharts.png")     #chemin de votre image à définir
        st.image(image, use_column_width=True)
        
        # colors = ['lightblue', 'beige', 'lightseagreen', 'antiquewhite', 'cadetblue', 'darkorange', 'goldenrod']
        # fig = make_subplots(rows=1, cols=2, subplot_titles=["Répartition de FIRE_SIZE_CLASS", "Répartition de DURATION_CLASS"],
        #                 specs=[[{'type': 'pie'}, {'type': 'pie'}]])

        # # Ajouter le premier Pie chart
        # fig.add_trace(
        #     go.Pie(labels=df_fire['FIRE_SIZE_CLASS'].value_counts().index,
        #         values=df_fire['FIRE_SIZE_CLASS'].value_counts().values,
        #         marker_colors=colors,
        #         marker_line=dict(color='black', width=2.5),
        #         pull=[0.2, 0.2, 0, 0.2, 0.4, 0.6, 0.8]),
        #     row=1, col=1
        # )

        # # Ajouter le deuxième Pie chart
        # fig.add_trace(
        #     go.Pie(labels=df_duration['DURATION_CLASS'].value_counts().index,
        #         values=df_duration['DURATION_CLASS'].value_counts().values,
        #         marker_colors=colors,
        #         marker_line=dict(color='black', width=2.5),
        #         pull=[0, 0.2, 0.3]),
        #     row=1, col=2
        # )
        # button_fire_size_class = dict(label="Superficie", method="restyle", args=[{"visible": [True, False]}])
        # button_duration_class = dict(label="Durée", method="restyle", args=[{"visible": [False, True]}])
        # button_both = dict(label="Les deux", method="restyle", args=[{"visible": [True, True]}])
        # fig.update_layout(updatemenus=[dict(type='buttons',
        #                                     buttons=[button_fire_size_class, button_duration_class, button_both])])
                                               
       
        # fig.update_layout(title_text="Répartition de FIRE_SIZE_CLASS et DURATION_CLASS")
        # st.plotly_chart(fig)
    st.markdown("""
    <div style="text-align: justify;">
    La cible FIRE_SIZE_CLASS est composée de 7 classes. Les 3 premières, A, B et C représentent 97% des informations, dont 85% pour A et B.
    Les incendies sont majoritairement de petite superficie.
    La cible DURATION_CLASS est très bien équilibrée, nous n'aurons pas de biais à ce niveau lors de la modélisation.
    </div><br><br>
    """, unsafe_allow_html=True)

        
    # AFFICHAGE DISTRIBUTION DES TAILLES ET DUREES DE FEUX SELON LES VARIABLES EXPLICATIVES
    if st.checkbox("Afficher les tailles et durées de feux"):
        choix = ['STAT_CAUSE_DESCR', 'CLUSTER', 'DISCOVERY_DOY', 'DISCOVERY_MONTH', 'OWNER_DESCR', 'DISCOVERY_YEAR']
        option = st.selectbox('Choix de la variable explicative', choix)
        st.write('La variable explicative choisie est :', option)

        if option == 'STAT_CAUSE_DESCR':
            image = Image.open("STAT_CAUSE_DESCR.png")     #chemin de votre image à définir
            st.image(image, use_column_width=True)
        if option == 'CLUSTER':
            image = Image.open("CLUSTER.png")     #chemin de votre image à définir
            st.image(image, use_column_width=True)
        if option == 'DISCOVERY_DOY':
            image = Image.open("DISCOVERY_DOY.png")     #chemin de votre image à définir
            st.image(image, use_column_width=True)
        if option == 'DISCOVERY_MONTH':
            image = Image.open("DISCOVERY_MONTH.png")     #chemin de votre image à définir
            st.image(image, use_column_width=True)
        if option == 'OWNER_DESCR':
            image = Image.open("OWNER_DESCR.png")     #chemin de votre image à définir
            st.image(image, use_column_width=True)
        if option == 'DISCOVERY_YEAR':
            image = Image.open("DISCOVERY_YEAR.png")     #chemin de votre image à définir
            st.image(image, use_column_width=True)

        # fig, axes = plt.subplots(2, 1, figsize=(20,15))
        # fig.subplots_adjust(hspace=0.3)
        # # Premier graphique
        # data=pd.DataFrame(df_duration[['DURATION_CLASS',option]].value_counts().reset_index())
        # pourcentage = data.groupby(option)['count'].transform(lambda x: x / x.sum() * 100)
        # data['Pourcentage'] = pourcentage.values
        # sns.barplot(x='Pourcentage', y=option, hue='DURATION_CLASS', data=data, fill=True, orient='h', color='#8BD3E6',ax=axes[0])
        # axes[0].set_title("Durée")
        # if option != 'DURATION_CLASS':
        #     # Ajustez les coordonnées pour positionner l'axe intérieur sur la droite
        #     inner_ax = fig.add_axes([axes[0].get_position().x1 + 0.01, axes[0].get_position().y0, 0.3, axes[0].get_position().height])
        #     sns.barplot(x=option, y='DURATION_IN_HOURS', data=df_duration, dodge=False, color='#8BD3E6', ax=inner_ax)
        #     inner_ax.set_xticklabels(inner_ax.get_xticklabels(), rotation=45)
        #     inner_ax.set_title(f'Distribution de DURATION_IN_HOURS pour {option}')
        #     inner_ax.set(xlabel=None, ylabel=None)
        #     inner_ax.yaxis.tick_right()
        #     inner_ax.tick_params(axis='y', which='both', left=False, right=True)

        # #  Deuxième graphique
        
        # datas=pd.DataFrame(df_fire[['FIRE_SIZE_CLASS',option]].value_counts().reset_index())
        # pourcentages = datas.groupby(option)['count'].transform(lambda x: x / x.sum() * 100)
        # datas['Pourcentage'] = pourcentages.values
        # sns.barplot(x='Pourcentage', y=option, hue='FIRE_SIZE_CLASS', data=datas[(datas['FIRE_SIZE_CLASS']=='A')|(datas['FIRE_SIZE_CLASS']=='B')|(datas['FIRE_SIZE_CLASS']=='C')], fill=True, orient='h', color='#F5B7B1',ax=axes[1])
        # axes[1].set_title("Superficie")
        # if option != 'FIRE_SIZE_CLASS':
        #     # Ajustez les coordonnées pour positionner l'axe intérieur sur la droite
        #     inner_ax = fig.add_axes([axes[1].get_position().x1 + 0.01, axes[1].get_position().y0, 0.3, axes[1].get_position().height])
        #     sns.barplot(x=option, y='FIRE_SIZE', data=df_fire, dodge=False, color='#F5B7B1', ax=inner_ax)
        #     inner_ax.set_xticklabels(inner_ax.get_xticklabels(), rotation=45)
        #     inner_ax.set_title(f'Distribution de FIRE_SIZE pour {option}')
        #     inner_ax.set(xlabel=None, ylabel=None)
        #     inner_ax.yaxis.tick_right()
        #     inner_ax.tick_params(axis='y', which='both', left=False, right=True)

        # st.pyplot(fig)

    st.markdown("""
    <div style="text-align: justify;">
    <li><strong>Causes:</strong> La foudre provoque le plus de dégâts aux Etats-Unis, 
                que ce soit par l'étendue ou la durée des feux. On note également que 
                les incendies causés par la foudre sont ceux caractérisés par les durées les plus longues 
                (dans 70% des cas au-delà de 3h). A l'inverse, les feux causés par des enfants causent peu de dégâts par leur étendue, 
                et dans 58% des cas durent moins d'une heure.
    <li><strong>Zones géographiques:</strong> L'Alaska est la région ayant connu un cumul de surfaces et de durées de feux le plus élevé. 
                A contrario, les régions de l'est des Etats-Unis ont beaucoup plus été épargnées. 58% des feux en Nouvelle Angleterre durent moins d'une heure.</li>
    <li><strong>Weekday:</strong> On n'observe pas de différence significative entre les jours de la semaine.</li>
    <li><strong>Mois:</strong> Sans surprise, les mois d'été voient les surfaces et durées de feu les plus intenses. 
                55% des feux en août ont ainsi une durée de plus de 3h. En janvier et en avril, la majorité des feux sont de durée courte.</li>
    <li><strong>Propriétaire terrain:</strong> Les terrains FWS ont subi les plus grandes étendues cumulées de feu. Les terrains connaissant en règle générale des 
                feux de durées plus longues sont : USFS (Forest Service) et BLM (Bureau of Land Management). 
                A l'inverse les terrains BIA (Bureau of Indian Affairs) et Tribal connaissent en générale des feux de durée courte.</li>
    <li><strong>Année:</strong> Les surfaces cumulées de feux augmentent année après année, alors que la durée cumulée elle est plutôt en décroissance depuis 2008. 
                En effet, on observe un pic de feux de longue durée entre 2000 et 2008.</li>
    </div><br><br>
    """, unsafe_allow_html=True)


    # @st.cache
    # def plot_charts(option, df_duration, df_fire):
    #     fig, axes = plt.subplots(2, 1, figsize=(20,15))
    #     fig.subplots_adjust(hspace=0.3)

    #     # Premier graphique
    #     sns.countplot(y=option, hue='DURATION_CLASS', data=df_duration, dodge=False, color='#8BD3E6', ax=axes[0])
    #     axes[0].set_title("Durée")
    #     if option != 'DURATION_CLASS':
    #         # Ajustez les coordonnées pour positionner l'axe intérieur sur la droite
    #         inner_ax = fig.add_axes([axes[0].get_position().x1 + 0.01, axes[0].get_position().y0, 0.3, axes[0].get_position().height])
    #         sns.barplot(x=option, y='DURATION_IN_HOURS', data=df_duration, dodge=False, color='#8BD3E6', ax=inner_ax)
    #         inner_ax.set_xticklabels(inner_ax.get_xticklabels(), rotation=45)
    #         inner_ax.set_title(f'Distribution de DURATION_IN_HOURS pour {option}')
    #         inner_ax.set(xlabel=None, ylabel=None)
    #         inner_ax.yaxis.tick_right()
    #         inner_ax.tick_params(axis='y', which='both', left=False, right=True)

    #     # Deuxième graphique
    #     sns.countplot(y=option, hue='FIRE_SIZE_CLASS', data=df_fire, dodge=False, color='#F5B7B1', ax=axes[1])
    #     axes[1].set_title("Superficie")
    #     if option != 'FIRE_SIZE_CLASS':
    #         # Ajustez les coordonnées pour positionner l'axe intérieur sur la droite
    #         inner_ax = fig.add_axes([axes[1].get_position().x1 + 0.01, axes[1].get_position().y0, 0.3, axes[1].get_position().height])
    #         sns.barplot(x=option, y='FIRE_SIZE', data=df_fire, dodge=False, color='#F5B7B1', ax=inner_ax)
    #         inner_ax.set_xticklabels(inner_ax.get_xticklabels(), rotation=45)
    #         inner_ax.set_title(f'Distribution de FIRE_SIZE pour {option}')
    #         inner_ax.set(xlabel=None, ylabel=None)
    #         inner_ax.yaxis.tick_right()
    #         inner_ax.tick_params(axis='y', which='both', left=False, right=True)

    #     return fig

    # Utilisation de la fonction mise en cache
    # if st.checkbox("Afficher les tailles et durées de feux"):
    #     choix = ['STAT_CAUSE_DESCR', 'cluster', 'DISCOVERY_DOY', 'DISCOVERY_MONTH', 'OWNER_DESCR', 'DISCOVERY_YEAR']
    #     option = st.selectbox('Choix de la variable explicative', choix)
    #     st.write('La variable explicative choisie est :', option)

    #     # Appel de la fonction mise en cache
    #     fig = plot_charts(option, df_duration, df_fire)

    #     st.pyplot(fig)
        

# AFFICHAGE GEOGRAPHIE
    if st.checkbox("Afficher les graphiques géographiques"):
        chart_choice = st.selectbox('Choix du graphique', ['Détail des clusters', 'US Clusters'])
        if chart_choice == "US Clusters":
          image = Image.open("US clusters.png")     #chemin de votre image à définir
          st.image(image, use_column_width=True)
        if chart_choice =="Détail des clusters":
          image = Image.open("Each cluster.png")  #chemin de votre image à définir
          st.image(image,  use_column_width=True)
          
    
    st.markdown("""
    <div style="text-align: justify;">
    Les US ont été compartimentés selon les latitudes et longitudes. Une carte est ainsi obtenue et nous permet d'affiner notre analyse.            
    Ces graphiques représentent la répartition des incendies selon leur taille, pour chaque cluster.
    Les régions Sud et Ouest sont les plus touchées par les incendies de grande surface (ces scatterplot nous indiquent où se situent les incendies de grande superficie, pas leur fréquence).
    </div><br><br>
    """, unsafe_allow_html=True)

    # if st.checkbox("Carte US interactive"):
    #     fig=px.scatter(data_frame=df_fire,x='LONGITUDE',y='LATITUDE', color='FIRE_SIZE_CRITICITY',color_continuous_scale='ORRD', animation_frame='DISCOVERY_YEAR')   #code pour afficher en plotly
    #     fig['layout'].pop("updatemenus")
    #     st.plotly_chart(fig) 
      

#         chart_choice_geo = st.selectbox('Choix du graphique', ['US clusters', 'Per cluster', 'Cluster by time'])
#         # Affichage carte globale
#         fig_etat = generate_etat_chart(df_fire,chart_choice_geo)
#         # st.plotly_chart(fig_etat) # pour afficher le graph version plotly
#         st.pyplot(fig_etat)
#         # Affichage des superficies par cluster 
#         fig_cluster = generate_cluster_chart(Ouest,Sud_Est,Sud,Alaska,Nouvelle_Angleterre,MidWest,chart_choice_geo)
#         st.pyplot(fig_cluster)       
#         # Affichage interactif de chaque cluster       
#         if chart_choice_geo == 'Cluster by time':
#             fig_1=px.scatter(data_frame=Ouest, x='LONGITUDE', y='LATITUDE',color='FIRE_SIZE_CRITICITY', color_continuous_scale='ORRD',animation_frame='DISCOVERY_YEAR',
#                             hover_data={'STAT_CAUSE_DESCR': True})
#             fig_1['layout'].pop("updatemenus")
#             fig_1.update_layout(title="Ouest")
#             st.plotly_chart(fig_1) 
            
#             fig_2=px.scatter(data_frame=Sud_Est, x='LONGITUDE', y='LATITUDE',color='FIRE_SIZE_CRITICITY', color_continuous_scale='ORRD',animation_frame='DISCOVERY_YEAR',
#                             hover_data={'STAT_CAUSE_DESCR': True})
#             fig_2['layout'].pop("updatemenus")
#             fig_2.update_layout(title="Sud-Est")
#             st.plotly_chart(fig_2) 
            
#             fig_3=px.scatter(data_frame=Sud, x='LONGITUDE', y='LATITUDE',color='FIRE_SIZE_CRITICITY', color_continuous_scale='ORRD',animation_frame='DISCOVERY_YEAR',
#                             hover_data={'STAT_CAUSE_DESCR': True})
#             fig_3['layout'].pop("updatemenus")
#             fig_3.update_layout(title="Sud")
#             st.plotly_chart(fig_3) 
            
#             fig_4=px.scatter(data_frame=Alaska, x='LONGITUDE', y='LATITUDE',color='FIRE_SIZE_CRITICITY', color_continuous_scale='ORRD',animation_frame='DISCOVERY_YEAR',
#                             hover_data={'STAT_CAUSE_DESCR': True})
#             fig_4['layout'].pop("updatemenus")
#             fig_4.update_layout(title="Alaska")
#             st.plotly_chart(fig_4) 
            
#             fig_5=px.scatter(data_frame=Nouvelle_Angleterre, x='LONGITUDE', y='LATITUDE',color='FIRE_SIZE_CRITICITY', color_continuous_scale='ORRD',animation_frame='DISCOVERY_YEAR',
#                             hover_data={'STAT_CAUSE_DESCR': True})
#             fig_5['layout'].pop("updatemenus")
#             fig_5.update_layout(title="Nouvelle-Angleterre")
#             st.plotly_chart(fig_5) 
            
#             fig_6=px.scatter(data_frame=MidWest, x='LONGITUDE', y='LATITUDE',color='FIRE_SIZE_CRITICITY', color_continuous_scale='ORRD',animation_frame='DISCOVERY_YEAR',
#                             hover_data={'STAT_CAUSE_DESCR': True})
#             fig_6['layout'].pop("updatemenus")
#             fig_6.update_layout(title="MidWest")       
#             st.plotly_chart(fig_6)    


#--------------------------------------------------------------MODELISATION ----------------------------------------------------------
if page == pages[2]:
    st.write("### Modélisation")
    st.markdown("""
    <div style="text-align: justify;">
    <ul><strong>PREPROCESSING DF_FIRE et DF_DURATION</strong></ul>
    Pour l'un comme pour l'autre, nous sommes partis des datasets préparés préalablement pour la datavisualisation.
    Nous les avons préprocessés de la manière suivante:
    <li>Encodage "manuel" de variables explicatives (ex: mois et jour selon système trigonométrique)</li>     
    <li>Suppression de variables redondantes </li>
    <li>Réduction des occurences de la variable cible aux trois principales (97% de l'information) pour FIRE_SIZE_CLASS</li> 
    <li>Pour des raisons de performance, nous avons divisé pour DF_FIRE le nombre d'entrées par deux, tout en conservant la répartition de FIRE_SIZE_CLASS</li>          
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: justify;">
    <ul><strong>SEPARATION DES JEUX ET PROCESSING</strong></ul>
    Les jeux d'entrainement et de test ont été séparés selon la proportion 80%/20%, tout en respectant les proportions de chaque classe.
    Nous les avons ensuite processés de la manière suivante:
    <li>Remplacement des NaNs de DF_FIRE['DISCOVERY_TIME'] par la médiane (SimpleImputer)</li>     
    <li>Redéfinition des variables HOURS et MiN à partir de cette dernière puis conversion selon le système trigonométrique</li>
    <li>Encodage des variables cible (LabelEncoder)</li> 
    <li>Standardisation des valeurs par StandardScaler</li>          
    </div><br><br><br>
    """, unsafe_allow_html=True)

    
#     # CHOIX DU MODELE
    st.markdown("""
    <div style="text-align: justify;">
    <ul><strong>RESULTATS POUR FIRE_SIZE_CLASS</strong></ul>
    </div>
    """, unsafe_allow_html=True)

    choix = ['Random Forest','Logistic Regression', 'XG Boost']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)
    display = st.radio('Que souhaitez-vous montrer?', ('Accuracy','Classification_report', 'Confusion matrix'))
    if option == 'Random Forest':
        if display == 'Accuracy':
            rf_score= joblib.load("Rf_score")
            st.write(rf_score)
            # st.write('0.6467658143477044')
        elif display == 'Classification_report':
            rf_classification= joblib.load("Rf_class")
            st.text(rf_classification)
        else:
            rf_matrix= joblib.load("Rf_matrix")
            st.dataframe(rf_matrix)
    elif option == 'Logistic Regression':
        if display == 'Accuracy':
            logreg_score= joblib.load("logreg_score")
            st.write(logreg_score)
            # st.write('0.5803869721922024')
        elif display == 'Classification_report':
            logreg_classification= joblib.load("logreg_classification")
            st.text(logreg_classification)
        else:
            logreg_matrix= joblib.load("logreg_matrix")
            st.dataframe(logreg_matrix)
    else:
        if display == 'Accuracy':
            xgboost_score= joblib.load("xgboost_score")
            st.write(xgboost_score)
            # st.write('0.6649953095942114')
        elif display == 'Classification_report':
            xgboost_classification= joblib.load("Fxgboost_classification")
            st.text(xgboost_classification)
        else:
            xgboost_matrix= joblib.load("xgboost_matrix")
            st.dataframe(xgboost_matrix)  


    st.markdown("""
    <div style="text-align: justify;">
    <ul><strong>RESULTATS POUR DURATION_CLASS</strong></ul>
    </div>
    """, unsafe_allow_html=True)
    choix_duration = ['Random Forest','Logistic Regression', 'XG Boost']
    option_duration = st.selectbox('Choix  du modèle', choix_duration)
    st.write('Le modèle choisi est :', option_duration)
    displayd = st.radio('Que souhaitez-vous  montrer?', ('Accuracy','Classification_report', 'Confusion matrix'))
    if option_duration == 'Random Forest':
        if displayd == 'Accuracy':
            rf_scored= joblib.load("rf_scored")
            st.write(rf_scored)
        elif displayd == 'Classification_report':
            rf_classificationd= joblib.load("rf_classificationd")
            st.text(rf_classificationd)
        else:
            rf_matrixd= joblib.load("rf_matrixd")
            st.dataframe(rf_matrixd)
    elif option_duration == 'Logistic Regression':
        if displayd == 'Accuracy':
            logreg_scored= joblib.load("logreg_scored")
            st.write(logreg_scored)
        elif displayd == 'Classification_report':
            logreg_classificationd= joblib.load("logreg_classificationd")
            st.text(logreg_classificationd)
        else:
            logreg_matrixd= joblib.load("logreg_matrixd")
            st.dataframe(logreg_matrixd)
    else:
        if displayd == 'Accuracy':
            xgboost_scored= joblib.load("xgboost_scored")
            st.write(xgboost_scored)
        elif displayd == 'Classification_report':
            xgboost_classificationd= joblib.load("xgboost_classificationd")
            st.text(xgboost_classificationd)
        else:
            xgboost_matrixd= joblib.load("xgboost_matrixd")
            st.dataframe(xgboost_matrixd)
    
    
    st.markdown("""
    <div style="text-align: justify;"><br><br>
    <ul><strong>INTERPRETATION</strong></ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: justify;">
    Nous avons choisi des modèles de classification pour notre analyse, notamment le Random Forest et XG Boost qui évitent l'overfitting et sont adaptés aux datasets de grande dimension.  
    Pour interpréter nos modèles, nous regardons:
    <li>le score: nous éliminons ainsi la régression logistique.</li>    
    <li>le rappel: Cette métrique est un indicateur de performance quant au fait de prédire correctement les classes.
    L'objectif est de prédire de manière qualitative puisqu'il serait désastreux d'allouer des moyens insuffisants pour maîtriser
    un feu de superficie sous-estimée par notre modèle.</li><br>
    
    Nous choisissons de continuer avec le XG Boost et le Random Forest lors de l'étape d'optimisation, car ils présentent les meilleurs scores.
    Les classes 0 et 1 (les plus petites superficies regroupant 85% des données) sont mieux prédites. 
    La classe 2 l'est moins bien voire pas du tout dans le cas de la régression logistique.
    Cela est normal car la classe 2 est minoritaire (11% des données), les modèles ont donc des difficultés à la prédire.
                
    
    </div>
    """, unsafe_allow_html=True)
    rf_score= joblib.load("Rf_score")
    logreg_score= joblib.load("logreg_score")
    xgboost_score= joblib.load("XGBoost_opti_score")
    rf_scored= joblib.load("rf_scored")
    logreg_scored= joblib.load("logreg_scored")
    xgboost_scored= joblib.load("xgboost_scored")

    scores=[[rf_score, rf_scored],[xgboost_score,xgboost_scored],[logreg_score,logreg_scored]]
    modeles=['Random Forest','XG Boost','Logistic Regression']
    results=pd.DataFrame(data =scores,index=modeles,columns=['FIRE_SIZE_CLASS score','DURATION_CLASS score'])
    recalls=[['0.66,   0.76,   0.14','0.69,   0.49,   0.75'],['0.65,   0.82,   0.05','0.68,   0.45,   0.76'],['0.47,   0.80,   0','0.48,   0.36,   0.64']]
    results['FIRE_SIZE_CLASS recall'] = [recall[0] for recall in recalls]
    results['DURATION_CLASS recall'] = [recall[1] for recall in recalls]
    st.dataframe(results)

    
  

if page == pages[3]:
    st.write("### Optimisation")
    st.markdown("""
    <div style="text-align: justify;">
    L'optimisation  de Random Forest a été effectuée à travers la recherche des meilleurs hyperparamètres du modèle, via la méthode GridCVSearch. Comme les données étaient conséquentes, nous avons
    défini un choix restreint d'hyperparamètres dont voici la liste (pour les deux dataframes confondus):
    <ul><li>n_estimators: [10,20,40,60,50,100,250,500,1000]:  best=  250/1000 (resp. fire_size_class/duration_class)</li></ul>
    <ul><li>min_samples_leaf: [1,3,5]:  best= 5</li></ul>
    <ul><li>max_features: [sqrt,log2]:  best= sqrt</li></ul>
    
    Il en va de même pour le XG Boost dont l'optimisation a été plus rapide:
    <ul><li>n_estimators: [60,220,40]:  best= 180</li></ul>
    <ul><li>learning_rate: [0.01,0.1,0.05]:  best= 0.1/0.01 (resp. fire_size_class/duration_class) </li></ul>
    <ul><li>max_depth: [2,10,1]:  best= 9/2 (resp. fire_size_class/duration_class) </li></ul>          
    </div><br>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: justify;">
    <ul><strong>RESULTATS POUR FIRE_SIZE_CLASS</strong></ul>
    </div>
    """, unsafe_allow_html=True)

    #-------------------------------------------------------------FIRE OPTIMISATION---------------------------------------------------
    choix = ['Random Forest Updated', "XG boost Updated"]
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)
    display = st.radio('Que souhaitez-vous montrer?', ('Accuracy','Classification_report', 'Confusion matrix'))
    if option == 'Random Forest Updated':
        if display == 'Accuracy':
            rfopti_score=joblib.load("rfopti_score")
            st.write(rfopti_score)
        elif display == 'Classification_report':
            rfopti_classification=joblib.load("rfopti_classification")
            st.text(rfopti_classification)
        else:
            rfopti_matrix=joblib.load("rfopti_matrix")
            st.dataframe(rfopti_matrix)
    elif option == 'XG boost Updated':
        if display == 'Accuracy':
            xgopti_score=joblib.load("XGBoost_opti_score")
            st.write(xgopti_score)
        elif display == 'Classification_report':
            xgopti_classification=joblib.load("XGBoost_opti_class")
            st.text(xgopti_classification)
        else:
            xgopti_matrix=joblib.load("XGBoost_opti_matrix")
            st.dataframe(xgopti_matrix)

#-----------------------------------------------------------------DURATION OPTIMISATION---------------------------------------------
    st.markdown("""
    <div style="text-align: justify;">
    <ul><strong>RESULTATS POUR DURATION_CLASS</strong></ul>
    </div>
    """, unsafe_allow_html=True)
    choixd = ['Random Forest Updated', "XG boost Updated"]
    optiond = st.selectbox('Choix du  modèle', choixd)
    st.write('Le modèle choisi est :', optiond)
    displayd = st.radio('Que souhaitez-vous  montrer?', ('Accuracy','Classification_report', 'Confusion matrix'))
    if optiond == 'Random Forest Updated':
        if displayd == 'Accuracy':
            rfopti_scored=joblib.load("rfopti_scored")
            st.write(rfopti_scored)
        elif displayd == 'Classification_report':
            rfopti_classificationd=joblib.load("rfopti_classificationd")
            st.text(rfopti_classificationd)
        else:
            rfopti_matrixd=joblib.load("rfopti_matrixd")
            st.dataframe(rfopti_matrixd)
    elif optiond == 'XG boost Updated':
        if displayd == 'Accuracy':
            xgopti_scored=joblib.load("xgopti_scored")
            st.write(xgopti_scored)
        elif displayd == 'Classification_report':
            xgopti_classificationd=joblib.load("xgopti_classificationd")
            st.text(xgopti_classificationd)
        else:
            xgopti_matrixd=joblib.load("xgopti_matrixd")
            st.dataframe(xgopti_matrixd)
    
    rfopti_score=joblib.load("rfopti_score")
    rfopti_scored=joblib.load("rfopti_scored")
    xgopti_score=joblib.load("XGBoost_opti_score")
    xgopti_scored=joblib.load("xgopti_scored")

    st.markdown("""
    <div style="text-align: justify;"><br><br>
    <ul><strong>INTERPRETATION</strong></ul>
    </div>""", unsafe_allow_html=True)

    score=[[rfopti_score, rfopti_scored],[xgopti_score,xgopti_scored]]
    modele=['Random Forest','XG Boost']
    resultats=pd.DataFrame(data =score,index=modele,columns=['FIRE_SIZE_CLASS score','DURATION_CLASS score'])
    recalls=[['0.68,   0.80,   0.11','0.69,   0.49,   0.77'],['0.65,   0.80,   0.10','0.68,   0.40,   0.53']]
    resultats['FIRE_SIZE_CLASS recall'] = [recall[0] for recall in recalls]
    resultats['DURATION_CLASS recall'] = [recall[1] for recall in recalls]
    st.dataframe(resultats)           

    
    st.markdown("""
    <div style="text-align: justify;">
    L'optimisation de nos modèles nous a permis de gagner en performance sur nos Random Forest. XG Boost conserve sa performance initiale pour ce qui concerne les prédictions de superficie.
    En revanche, nous n'avons pas obtenu le résultat escompté pour la durée, dont la prédiction de la classe 2 s'est amoindrie.
    Les rappels sont restés égaux, nous n'avons pas réussi à améliorer les prédictions de la classe 2.
    
    En définitif, nous préférons travailler avec:
    <ul><li><strong>Random Forest pour prédire la durée</strong></li></ul>
    <ul><li><strong>XG Boost dont le temps de calcul est inférieur, pour prédire la superficie</strong></li></ul>
    </div>
    """, unsafe_allow_html=True)



  




if page==pages[4]:
    st.write("### Test de prédiction")
    
    cause=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    option = st.selectbox('Choix du paramètre cause: 1-Lightning, 2-Equipment Use, 3-Smoking, 4-Campfire, 5-Debris Burning, 6-Railroad, 7-Arson, 8-Children, 9-Miscellaneous, 10-Fireworks,11-Powerline, 12-Structure, 13-Missing/Undefined', cause)
    
    from datetime import date
    # Obtenez la date actuelle
    today = date.today()
    selected_date = st.date_input("Sélectionnez une date", today)
        
    option_3= selected_date.strftime('%B')
    option_4=selected_date.strftime('%A')
    option_7=selected_date.year
    option_5 = st.text_input("Entrez l'heure et minutes au format XXxx")
     
    owner=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    option_6=st.selectbox('Choix du paramètre Owner: 0-FOREIGN, 1-BLM, 2-BIA, 3-NPS, 4-FWS, 5-USFS, 6-OTHER FEDERAL, 7-STATE, 8-PRIVATE, 9-TRIBAL, 10-BOR, 11-COUNTY, 12-MUNICIPAL-LOCAL, 13-STATE OR PRIVATE, 14-NOT SPECIFIED, 15-UNDEFINED FEDERAL', owner)
    
    from sklearn.cluster import KMeans
    kmeans = joblib.load("kmeans")
    image = Image.open("US clusters.png")     #chemin de votre image à définir
    st.image(image, caption='Image PNG', use_column_width=True)
    long_input=st.text_input("Saisissez la longitude comprise entre -180 et -65")
    lat_input=st.text_input("Saissiez la latitude comprise entre 20 et 70")
    
    # Convertissez les entrées en tableau bidimensionnel
    if long_input and lat_input:
            input_data = np.array([[float(long_input), float(lat_input)]])
    clust = kmeans.predict(input_data)
    # On renomme les labels des clusters
    if clust==0:
        option_2=1
    elif clust==1:
        option_2=5
    elif clust==2:
        option_2=0
    elif clust==3:
        option_2=3
    elif clust==4:
        option_2=2
    else:
        option_2=4

  
    liste={'année':option_7,'mois':option_3,'jour':option_4, 'time':option_5,'cause':option,'cluster': option_2, 'terrain':option_6,'longitude':long_input,'latitude':lat_input}
    observation=pd.DataFrame(liste, index=[0])
    
# PREPROCESSING DE L'OBSERVATION----------------------------------------------------------------------------------------------------------------   
  
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score, confusion_matrix,classification_report
    from sklearn.impute import SimpleImputer
    import xgboost as xgb
    observation['jour']=observation['jour'].replace(['Wednesday', 'Monday', 'Thursday', 'Tuesday', 'Friday', 'Sunday','Saturday'], [2,0,3,1,4,6,5]).apply(lambda x: np.cos((2*np.pi*x)/7))
    observation['mois'] = observation['mois'].replace(['February', 'May', 'June', 'July', 'March', 'September', 'October','November', 'August', 'January', 'April', 'December'],
                                                              [2,5,6,7,3,9,10,11,8,1,4,12]).astype('float').apply(lambda x: np.cos((2*np.pi*x)/12))
    imputer=SimpleImputer()
    observation['time'] = imputer.fit_transform(observation['time'].values.reshape(-1, 1))
    observation['time'] = imputer.transform(observation['time'].values.reshape(-1, 1))
    observation['heure']= observation['time'].apply(lambda x: x//100).apply(lambda x : np.cos((2*np.pi*x)/24))
    observation['minute']=observation['time'].apply(lambda x: x%100).apply(lambda x : np.cos((2*np.pi*x)/60))
    observation['jour']= observation['jour'].apply(lambda x : np.cos((2*np.pi*x)/7))
    
    observationd=observation.drop('année',axis=1)
   
    # # # Standardisation des données
    scaler = StandardScaler()
    observation= scaler.fit_transform(observation)
    observationd= scaler.fit_transform(observationd)
    
    if st.checkbox("Afficher la prédiction"):
        clf = joblib.load("XGBoost")                         # pour les deux cibles, nous avons utilisé le xgboost dans le streamlit car cela consomme moins de ressource
        clfd=joblib.load("XGBoostd")
        predict = clf.predict(observation) # prédiction superficie
        predictd=clfd.predict(observationd) # prédiction durée
        st.write("La classe de superficie sera: ", predict)
        st.write("La classe de durée sera: ", predictd)
   
    

if page==pages[5]:

    st.write("### Conclusion")

    st.markdown("""
    <div style="text-align: justify;">
    Nous avons travaillé sur un jeu de données contenant plus de 1,88 millions d'entrées. Nous avons défini deux variables cible, <strong>la superficie et la durée</strong>.
    Nos variables explicatives, sur lesquelles nous avons axé notre étude étaient:
    <li>Les variables temporelles: année, mois, jour, heures, minutes</li>
    <li>Les causes d'incendies</li>
    <li>Les zones géographiques</li>
    
    En résumé, les incendies ont pour la grande majorité (85%) de 1992 à 2015 une petite superficie (entre 0 et 9,9 acres soit 4 hectares) et ne durent pas longtemps (entre 0 et 3h15min, classe 0 et 1).
    Les superficies sont influencées par divers facteurs:
               <li>la cause: ce sont surtout les éclairs qui favorisent des incendies de grande surface</li>
                <li>la zone géographique: les régions de l'Ouest et de l'Alaska sont propices au développement des feux</li>
                <li>la saison: les feux se développent plus facilement durant la période estivale bien que le plus grand nombre d'incendies se déclare de Mars à Juin</li>

    Par le biais de notre étude sur la durée, nous mettons en évidence plusieurs points:
    <li>la cause: les activités humaines sont à l'origine de feux de courte durée (feu de camp, débris incandescent etc), tandis que les feux les plus longs sont dus aux éclairs</li>
    <li>la zone géographique: les feux les plus longs sont sans suprise associés aux feux les plus étendues, donc en régions Ouest et Alaska</li>
    <li>la saison: même constat, les feux sont plus longs durant la période estivale</li><br>

    Nous avons ensuite modélisé nos données afin de prédire d'une part la superficie d'un feu, d'autre part sa durée. Nos modèles Random Forest et XG Boost ont obtenu des scores
    avoisinant les 66%, avec des rappels satisfaisants sur les classes 0 et 1, que ce soit pour la superficie ou la durée.

    <ul><strong>LES LIMITES</strong></ul>    
    <li>Notre jeu de données contient très peu d'informations sur les incendies de grande taille et de longue durée.</li> 
    <li>CLasse 2 mal prédite</li>
    <li>Incorporation de données météorologiques</li><br><br>
                
    Pour conclure, notre étude souligne l'urgence et l'importance d'une compréhension approfondie de ces phénomènes naturels et de la mise en place de mesures 
    d'atténuation efficaces ou de mesures préventives. Face à l'évolution des conditions climatiques, il est impératif d'adopter des pratiques durables et de développer
    des stratégies innovantes afin de mieux prévenir et gérer les incendies de manière à garantir un avenir plus résilient et durable.
    </div>
    """, unsafe_allow_html=True)


      
    
    image = Image.open("planet.jpg")     #chemin de votre image à définir
    st.image(image, use_column_width=True)
    
   