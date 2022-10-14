# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:25:54 2022

@author: rokefeler@gmail.com
Explore and analize data with python
https://learn.microsoft.com/en-us/training/modules/explore-analyze-data-with-python/3-exercise-explore-data
"""
#%%
data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
print(data)

#%%
import numpy as np
grades = np.array(data)
print(grades)

#%% diferencias de array y vecto numpy
print(type(data),'x 2:',data*2) #Aqui solo duplica la cant de elementos
print(type(grades),'x 2:',grades*2) #aqui se realiza operacion x cada elemento (x2)
grades.shape #devuelve longitud de matriz grades
grades.mean() #dev. el promedio de los valores
#%%

study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]
# Create a 2D array (an array of arrays)
student_data = np.array([study_hours, grades])
#%%
student_data
print(student_data)
student_data.shape  #ahora muestra que es una matriz dimensional de 2x22
student_data[1][2]  #tener en cuenta que index empieza en 0, aqui muestra 47
#%%
# Get the mean value of each sub-array
avg_study = student_data[0].mean()
avg_grade = student_data[1].mean()

print('Average study hours: {:.2f}\nAverage grade: {:.2f}'.format(avg_study, avg_grade))
#%% trabajando con dataframes de pandas
import pandas as pd
df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie',
                                     'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
                                     'Rhonda', 'Giovanni', 'Francesca',
                                     'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem','Helena','Ismat','Anila',
                                     'Skye','Daniel','Aisha'],
                            'StudyHours':student_data[0],
                            'Grade':student_data[1]})
df_students
#%%
# Get the data for index value 5
df_students.loc[5]

# Get the rows with index values from 0 to 5, muestra registros del index 0 a 5 (6 elementos)
df_students.loc[0:5]

# Get data in the first five rows, muestra registros de solo los primeros 5 elementos
df_students.iloc[0:5]
'''iloc identifica los valores de datos en un DataFrame por posición, 
  que se extiende más allá de las filas a las columnas. 
  Entonces, por ejemplo, puede usarlo para encontrar los valores de las 
  columnas en las posiciones 1 y 2 en la fila 0, así: '''
df_students.iloc[0,[1,2]]
'''Volvamos al métodolocy veamos cómo funciona con las columnas. 
Recuerde que loc se utiliza para localizar elementos de datos en función 
de los valores del índice en lugar de las posiciones.
En ausencia de una columna de índice explícita, las filas de nuestro marco
de datos se indexan como valores enteros, pero las columnas se identifican 
por su nombre:'''
df_students.loc[0,'Grade']
#%%
#Aquí hay otro truco útil. Puede utilizar el métodolocpara buscar 
#filas indizadas en función de una expresión de filtrado que haga referencia
# a columnas con nombre distintas del índice, Ej.:
df_students.loc[df_students['Name']=='Aisha']

#En realidad, no es necesario usar explícitamente el método loc para hacer esto
#, simplemente puede aplicar una expresión de filtrado DataFrame, como esta:
df_students[df_students['Name']=='Aisha']

#Y para una buena medida, puede lograr los mismos resultados utilizando 
#el método deconsultade DataFrame, como este:
df_students.query('Name=="Aisha"')

'''Otro ejemplo de esto es la forma en que se hace referencia a un nombre de 
columna DataFrame. Puede especificar el nombre de la columna como un valor de
índice con nombre (como en los ejemplos que hemos visto hasta ahora), o puede 
usar la columna como una propiedad del DataFrame, de la siguiente manera:
    df_students['Name'] '''
df_students[df_students.Name == 'Aisha']

#%%
#***** Carga de un DataFrame desde un archivo ***
'''Construimos el DataFrame a partir de algunas matrices existentes. 
Sin embargo, en muchos escenarios del mundo real, los datos se cargan desde 
fuentes como archivos. 
Reemplacemos las calificaciones del estudiante DataFrame con el contenido 
de un archivo de texto.'''
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv
df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')
#%%
df_students.head()
'''El método read_csv de DataFrame se utiliza para cargar datos de archivos
 de texto. Como puede ver en el código de ejemplo, puede especificar 
 opciones como el delimitador de columnas y qué fila (si la hay) 
 contiene encabezados de columna (en este caso, el delimitador es una coma 
                                  y la primera fila contiene los nombres de 
                                  columna; estos son los ajustes 
                                  predeterminados, por lo que los parámetros 
                                  podrían haberse omitido).'''
#%% ****Manejo de valores faltantes ***
'''Uno de los problemas más comunes con los que los científicos de datos 
deben lidiar es con datos incompletos o faltantes. Entonces, ¿cómo sabríamos 
que el DataFrame contiene valores faltantes? Puede utilizar el método 
isnull para identificar qué valores individuales son nulos, de la siguiente
manera:'''
df_students.isnull()
df_students.isnull().sum() # Para un DataFrame más grande, 
                           #sería ineficiente revisar todas las filas 
                           #y columnas individualmente
#%%
'''Así que ahora sabemos que falta un valor de StudyHours y faltan dos 
valores de Grade. Para verlos en contexto, podemos filtrar el dataframe 
para incluir solo filas donde cualquiera de las columnas 
(eje 1 del DataFrame) sea nula.'''
df_students[df_students.isnull().any(axis=1)]
#%%
'''Cuando se recupera el DataFrame, los valores numéricos que faltan 
aparecen comoNaN (no un número).
Entonces, ahora que hemos encontrado los valores nulos, ¿qué podemos hacer 
al respecto?

Un enfoque común es imputar valores de reemplazo. Por ejemplo, si falta el
número de horas de estudio, podríamos asumir que el estudiante estudió 
durante un período promedio de tiempo y reemplazar el valor faltante con 
la media de horas de estudio. Para ello, podemos utilizar el método fillna,
 así:'''
df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())
df_students
# %% Explrar datos en DataFrame
# Get the mean study hours using to column name as an index
#obtener la media de estudio usando nombre de columna como un indice
mean_study = df_students['StudyHours'].mean()

# Get the mean grade using the column name as a property (just to make the point!)
# obtener la media usando la columna como propiedad
mean_grade = df_students.Grade.mean()

# Print the mean study hours and mean grade
print('Average weekly study hours: {:.2f}\nAverage grade: {:.2f}'.format(mean_study, mean_grade))
# %% De acuerdo, filtremos el DataFrame para encontrar solo a los estudiantes que estudiaron durante más de la cantidad promedio de tiempo.
df_students[df_students.StudyHours > mean_study]


# %%
'''Tenga en cuenta que el resultado filtrado es en sí mismo un 
DataFrame, por lo que puede trabajar con sus columnas como cualquier
otro DataFrame.
Por ejemplo, encontremos la calificación promedio de los estudiantes
que realizaron más que la cantidad promedio de tiempo de estudio.'''
df_students[df_students.StudyHours > mean_study].Grade.mean()

# %%
'''Supongamos que la calificación aprobatoria para el curso es de 60.
Podemos usar esa información para agregar una nueva columna al 
DataFrame, indicando si cada estudiante aprobó o no.
Primero, crearemos unaseriede Pandas que contenga el indicador de 
aprobación/falla (True o False), y luego concatenaremos esa serie como 
una nueva columna (eje 1) en el DataFrame.'''
passes = pd.Series(df_students['Grade']>=60)
df_students_aprobed = pd.concat([df_students, passes.rename('Pass')], axis=1)
df_students_aprobed
# %%
'''Los DataFrames están diseñados para datos tabulares, y puede usarlos
para realizar muchos de los tipos de operaciones de análisis de 
datos que puede realizar en una base de datos relacional; como 
agrupar y agregar tablas de datos.
Por ejemplo, puede usar el método groupby para agrupar los datos de
los estudiantes en grupos basados en la columna Aprobado que agregó 
anteriormente y contar el número de nombres en cada grupo, en otras 
palabras, puede determinar cuántos estudiantes aprobaron y reprobaron.
'''
print(df_students_aprobed.groupby(df_students_aprobed.Pass)['StudyHours','Grade'].mean())
# %%
'''Los DataFrames son increíblemente versátiles y facilitan la manipulación 
de datos. Muchas operaciones de DataFrame devuelven una nueva copia del
DataFrame; por lo tanto, si desea modificar un DataFrame pero mantener
la variable existente, debe asignar el resultado de la operación a
la variable existente. Por ejemplo, el código siguiente ordena los 
datos del alumno en orden descendente de Grade y asigna el DataFrame 
ordenado resultante a la variable df_students_aprobed.'''
df_students_aprobed = df_students_aprobed.sort_values('Grade', ascending=False)
df_students_aprobed
# %%
