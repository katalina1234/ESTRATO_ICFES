import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px 
from PIL import Image
import yfinance as yf
df = pd.read_excel('BASE_FINAL.xlsx')

df.loc[df['ESTU_HORASSEMANATRABAJA'] == 0, 'ESTU_HORASSEMANATRABAJA'] = 'cero'

st.title('ICFES|2019-1|ESTRATOS 1 Y 2')
st.write('Esta app calcula el puntaje del icfes para los estratos 1 y 2 a partir de caracteristicas especificas.')
st.write(df.head())

st.subheader('ANÁLISIS DESCRIPTIVO DEL PUNTAJE GLOBAL GENERAL')
st.write(df.describe())
st.write('El mínimo puntaje alcanzado para los estratos 1 y 2 es de 123 puntos. el mayor puntaje alcanzado fue de 422 puntos. El promedio de los puntajes obtenidos en el Icfes para los estudiantes quienes se encuentran en el estrato 1 y 2 es de 241 puntos.')

#DEPARTAMENTO
st.subheader('PUNTAJE GLOBAL Y DEPARTAMENTO')
st.write(df.loc[:,['ESTU_DEPTO_RESIDE', 'PUNT_GLOBAL']].groupby('ESTU_DEPTO_RESIDE').describe())
st.write('El departamento que obtuvo el mayor puntaje en promedio fue el valle con una puntuación global igual a 271. [Córdoba obtuvo un mayor puntaje equivalente a 288, sin embargo, solamente hay 3 estudiantes evaluados para ese departamento]')
st.write('El departamento que obtuvo el menor puntaje en promedio fue magdalena con una puntuación global igual a 198.')
st.write('Para los departamentos donde se encuentra gran parte de la población se puede decir que: la capital de Colombia, Bogotá, cuenta con 747 estudiantes quienes se encuentran en los estratos 1 y 2 para este estudio. el puntaje promedio fue igual a 233 puntos. para el departamento de Antioquia con 564 estudiantes obtuvo una puntuación promedio igual a 212 puntos. finalmente se puede decir que el valle con una población igual a 1218 estudiantes de estratos 1 y 2 tienen la mejor puntuación en promedio comparando con los demás departamentos.')

#SEXO
st.subheader('PUNTAJE GLOBAL Y SEXO')
st.write(df.loc[:,['ESTU_GENERO', 'PUNT_GLOBAL']].groupby('ESTU_GENERO').describe())
st.write('El 51.47 porciento de la poblacion es femenina y el 48.53 porciento es masculina')
st.write('En promedio la puntuacion de las mujeres es menor que la de los hombres. Las mujeres obtuvieron un puntaje de 237 puntos mientras que los hombres obtuvieron en promedio 246 puntos.')

Courses = list({'FEMENINO':237.148713, 'MASCULINO':246.119192}.keys())
values = list({'FEMENINO':237.148713, 'MASCULINO':246.119192}.values())

fig2 = plt.figure(figsize = (5, 2))

plt.bar(Courses, values)
plt.xlabel("Genero")
plt.ylabel("Promedio")
plt.title("GRÁFICO PUNTAJE GLOBAL SEGUN LA VARIABLE SEXO")
st.pyplot(fig2)



#EDAD
st.subheader('PUNTAJE GLOBAL Y EDAD')
st.write('Se puede observar que en promedio los hombres quienes están en el estrato 1 y 2 obtuvieron un mayor puntaje global.') 
fig = plt.figure(figsize=(5, 2))
ax = fig.add_subplot(1,1,1)

ax.scatter(
        df['EDAD'],
        df['PUNT_GLOBAL'],
    )

ax.set_xlabel("EDAD")
ax.set_ylabel("PUNT_GLOBAL")
st.write(fig)

st.write('Se puede observar que a partir de los 25 años de edad empieza a decrementar el puntaje global de la prueba Icfes. Asi mismo, los puntajes mas altos estan en los invidivuos que tienente entre los 16 y 20 años.')

#EDUCACION PADRE Y MADRE
st.subheader('PUNTAJE GLOBAL SEGUN LA EDUCACIÓN DEL PADRE Y DE LA MADRE')
st.write('PUNTAJE GLOBAL SEGUN EDUCACIÓN DEL PADRE')
st.write(df.loc[:,['FAMI_EDUCACIONPADRE', 'PUNT_GLOBAL']].groupby('FAMI_EDUCACIONPADRE').describe())
st.write('PUNTAJE GLOBAL SEGUN EDUCACIÓN DE LA MADRE')
st.write(df.loc[:,['FAMI_EDUCACIONMADRE', 'PUNT_GLOBAL']].groupby('FAMI_EDUCACIONMADRE').describe())

st.write('GRÁFICA')
image = Image.open(r"./IMAGENPADRE.PNG")
st.image(image)

st.write('Se puede decir que los estudiantes quienes su padre o madre son profesionales y tienen un postgrado mantienen la mayor puntuacion en la prueba icfes. Mientras que, aquellos quienes no tienen ningun nivel educativo más alto alcanzado por sus padres corresponden a la puntuacion mas baja en el examen.')


# INTERNET
st.subheader('PUNTAJE GLOBAL SEGUN EL ACCESO A INTERNET, COMPUTADOR Y BILINGUISMO DEL ESTABLECIMIENTO')
labels = 'SI', 'NO'
sizes = [249.989521, 220.734060]
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.pie(sizes, labels=labels,
        shadow=False, startangle=90)
ax1.set_xlabel("¿TIENE INTERNET?")    
ax1.set_title('GRÁFICO PUNTAJE GLOBAL SEGUN LA VARIABLE INTERNET')      
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# COMPUTADOR
labels = 'SI', 'NO'
sizes = [250.546163,225.593414]
fig5, ax2 = plt.subplots(figsize=(10, 5))
ax2.pie(sizes, labels=labels,
        shadow=False, startangle=90)
ax2.set_xlabel("¿TIENE COMPUTADOR?")  
ax2.set_title('GRÁFICO PUNTAJE GLOBAL SEGUN LA VARIABLE COMPUTADOR')      
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# BILINGUE
labels = 'SI', 'NO'
sizes = [281.185185,240.578510]
fig7, ax3 = plt.subplots(figsize=(10, 5))
ax3.pie(sizes, labels=labels,
        shadow=False, startangle=90)
ax3.set_xlabel("¿EL ESTABLECIMIENTO ES BILINGUE?")  
ax3.set_title('GRÁFICO PUNTAJE GLOBAL SEGUN LA VARIABLE BILINGUE')      
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

col5, col6, col7 = st.columns((1,1,1))
with col5:
 st.pyplot(fig1)
with col6:
 st.pyplot(fig5)
with col7:
 st.pyplot(fig7)


st.write('Se evidencia que, aquellos individuos quienes tienen acceso a internet, tienen una puntuación más alta en la prueba icfes. Aproximadamente 29 puntos más que los individuos quienes no cuentan con este servicio. También se puede decir que, aquellos estudiantes quiénes tienen acceso a un computador, tienen una puntuación mas alta en la prueba Icfes. Aproximadamente 25 puntos mayor.De acuerdo al tipo de establecimiento se puede evidenciar que, aquellos estudiantes quienes provienen de un establecimiento bilingue tienen un puntaje mas elevado. Aproximadamente 42 puntos mas. Así mismo,  el 98 porciento de los estudiantes pertenecen a colegios que no son bilingues mientras que el restante 2 porciento si.')


#LECTURA
st.subheader('PUNTAJE GLOBAL SEGUN LA CANTIDAD DE TIEMPO QUE DEDICA A LEER')
st.write(df.loc[:,['ESTU_DEDICACIONLECTURADIARIA', 'PUNT_GLOBAL']].groupby('ESTU_DEDICACIONLECTURADIARIA').describe())
st.write('Solo el 3.51 porciento de los estudiantes lee mas de dos horas. El 62 porciento de los estudiantes no leen o leen menos de 30 minutos.')
Courses = list({'30 minutos o menos': 234.176471, 'Entre 1 y 2 horas':255.424731, 'Entre 30 y 60 minutos':257.345857,
                   'No leo por entretenimiento':225.829218,'mas de 2 horas':272.000000}.keys())
values = list({'30 minutos o menos': 234.176471, 'Entre 1 y 2 horas':255.424731, 'Entre 30 y 60 minutos':257.345857,
                   'No leo por entretenimiento':225.829218,'mas de 2 horas':272.000000}.values())

fig6 = plt.figure(figsize = (5, 2))

plt.barh(Courses, values)
plt.xlabel("PUNTAJE GLOBAL")
plt.ylabel("TIEMPO DE LECTURA")
plt.title("GRÁFICO PUNTAJE GLOBAL SEGUN LA CANTIDAD DE HORAS DEDICADAS A LA LECTURA")
st.pyplot(fig6)
st.write('De acuerdo al gráfico anterior se puede evidenciar que aquellos quienes leen mas de 2 horas tienen la mejor puntuación en el examen. Los estudiantes que leen entre 30 minutos a 2 horas también obtienen resultados mas arriba que el promedio. Finalmente, aquellos individuos que no leen por entretenimiento o que leen menos de 30 minutos tienen las puntuaciones mas bajas.')

#OFICIAL
st.subheader('PUNTAJE GLOBAL Y NATURALEZA DEL ESTABLECIMIENTO')

Courses = list({'NO OFICIAL': 245.289198, 'OFICIAL':215.340909}.keys())
values = list({'NO OFICIAL': 245.289198, 'OFICIAL':215.340909}.values())

fig8 = plt.figure(figsize = (5, 2))

plt.bar(Courses, values)
plt.xlabel("Naturaleza")
plt.ylabel("Puntaje Global")
plt.title("GRÁFICO PUNTAJE GLOBAL SEGUN LA NATURALEZA DEL COLEGIO")
st.pyplot(fig8)
st.write('Se puede apreciar que, aquellos estudiantes que provienen de colegios no oficiales tienen una puntuación más alta en el examen. Aproximadamente el 87 por ciento de los individuos provienen de colegios no oficiales.')

#ALIMENTACION
st.subheader('PUNTAJE GLOBAL Y ALIMENTACIÓN')
st.write(df.loc[:,['FAMI_COMECARNEPESCADOHUEVO', 'PUNT_GLOBAL']].groupby('FAMI_COMECARNEPESCADOHUEVO').describe())
st.write('La proteina como la carne, el pescado, el huevo y pollo son importantes para el desarrollo del cerebro.Aquellos estudiantes quienes no comen aquellas proteinas o se alimentan de ellas 1 o 2 veces a la semana tienen rendimientos mas bajos.Aproximadamente el 27,1 por ciento pertenece a la población anteriormente nombrada. Por otro lado, los individuos quienes comen 3 o 5 veces por semana, incluso aquellos que se alimentan de estas proteinas todos los dias tienen los mejores puntajes. Aproximadamente el 72,9 por ciento pertenece a esta población.')

#ESTU_HORASSEMANATRABAJA
st.subheader('PUNTAJE GLOBAL Y HORAS DE TRABAJO')
import plotly.express as px
fig9 = px.box(df, x="ESTU_HORASSEMANATRABAJA", y="PUNT_GLOBAL")
st.write(df.loc[:,['ESTU_HORASSEMANATRABAJA', 'PUNT_GLOBAL']].groupby('ESTU_HORASSEMANATRABAJA').describe())
st.write('GRÁFICA')
st.write(fig9)
st.write('De acuerdo a la tabla anterior se puede decir que aquellos quienes no trabajan tienen en promedio el puntaje mas alto a comparación de quiénes si trabajan. El puntaje más bajo corresponde a quiénes trabajan más de 30 horas a la semana. Hay algunos datos atipicos dentro de la categoria "menos de 10 horas". Asi mismo puede verse que no existe gran diferencia en el puntaje de aquellos estudiantes que trabajan entre 11 y 20  a quienes trabajan menos de 10 horas y mas de 30 horas. ')

st.subheader('MODELO')
edad = st.number_input('Edad', min_value=8)
pd.get_dummies(df,columns={'FAMI_EDUCACIONMADRE'},drop_first=True)

famimadre=st.selectbox('FAMI_EDUCACIONMADRE',options={'Ninguno','Primaria incompleta','Primaria completa','Secundaria (Bachillerato) incompleta','Secundaria (Bachillerato) completa','Tecnica o tecnologica incompleta','Tecnica o tecnologica completa','Educacion profesional incompleta','Postgrado','No Aplica','No sabe'})

if famimadre == 'Ninguno':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=1
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'Primaria incompleta':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=1
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'Primaria completa':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=1
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'Secundaria (Bachillerato) incompleta':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=1
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'Secundaria (Bachillerato) completa':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=1
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'Tecnica o tecnologica incompleta':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=1
elif famimadre == 'Tecnica o tecnologica completa':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=1
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'Educacion profesional incompleta':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=1
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'Posgrado':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=1
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'No Aplica':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=1
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
elif famimadre == 'No sabe':
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=1
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0
else:
    FAMI_EDUCACIONMADRE_Educacion_profesional_incompleta=0
    FAMI_EDUCACIONMADRE_Ninguno=0
    FAMI_EDUCACIONMADRE_No_Aplica=0
    FAMI_EDUCACIONMADRE_No_sabe=0
    FAMI_EDUCACIONMADRE_Postgrado=0
    FAMI_EDUCACIONMADRE_Primaria_completa=0
    FAMI_EDUCACIONMADRE_Primaria_incompleta=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_completa=0
    FAMI_EDUCACIONMADRE_Secundaria_Bachillerato_incompleta=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_completa=0
    FAMI_EDUCACIONMADRE_Tecnica_o_tecnologica_incompleta=0

sexo=st.selectbox('ESTU_GENERO_M',options={'Femenino','Masculino'})

if sexo == 'Masculino':
    ESTU_GENERO_M = 1
    ESTU_GENERO_F = 0
elif sexo == 'Femenino':
    ESTU_GENERO_M = 0
    ESTU_GENERO_F = 1
else:
    ESTU_GENERO_M = 0
    ESTU_GENERO_F = 0



internet=st.selectbox('FAMI_TIENEINTERNET_Si',options={'Si','No'})

if internet == 'Si':
    FAMI_TIENEINTERNET_Si = 1
    FAMI_TIENEINTERNET_No = 0
elif internet == 'No':
    FAMI_TIENEINTERNET_Si = 0
    FAMI_TIENEINTERNET_No = 1
else:
    FAMI_TIENEINTERNET_Si = 0
    FAMI_TIENEINTERNET_No = 0


computador=st.selectbox('FAMI_TIENECOMPUTADOR_Si',options={'Si','No'})

if computador == 'Si':
    FAMI_TIENECOMPUTADOR_Si = 1
    FAMI_TIENECOMPUTADOR_No = 0
elif computador == 'No':
    FAMI_TIENECOMPUTADOR_Si = 0
    FAMI_TIENECOMPUTADOR_No = 1
else:
    FAMI_TIENECOMPUTADOR_Si = 0
    FAMI_TIENECOMPUTADOR_No = 0


bilingue=st.selectbox('COLE_BILINGUE_S',options={'S','N'})

if bilingue == 'S':
    COLE_BILINGUE_S = 1
    COLE_BILINGUE_N = 0
elif bilingue == 'N':
    COLE_BILINGUE_S = 0
    COLE_BILINGUE_N = 1
else:
    COLE_BILINGUE_S = 0
    COLE_BILINGUE_N = 0


naturaleza=st.selectbox('COLE_NATURALEZA_OFICIAL',options={'OFICIAL','NO OFICIAL'})

if naturaleza == 'OFICIAL':
    COLE_NATURALEZA_OFICIAL = 1
    COLE_NATURALEZA_NOOFICIAL = 0
elif naturaleza == 'NO OFICIAL':
    COLE_NATURALEZA_OFICIAL = 0
    COLE_NATURALEZA_NOOFICIAL = 1
else:
    COLE_NATURALEZA_OFICIAL = 0
    COLE_NATURALEZA_NOOFICIAL = 0