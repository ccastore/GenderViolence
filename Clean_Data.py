import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from collections import Counter

#Directory clean
dir_data = '/home/carlos/Escritorio/ArticuloVI/n_Data.txt'


#Import the data 
data = pd.read_csv(dir_data, header=None)
n_data=open("/home/carlos/Escritorio/ArticuloVI/PostData.txt",'w')


for frase in range (len(data)):
	print(data[0][frase])
	longitud_frase=len(data[0][frase])

	apuntador=0
	letra=0
	j=0
	try:
		while letra < (longitud_frase):
			
			if data [0][frase][letra] == "@" or data[0][frase][letra] =="#" or (data[0][frase][letra]=="h" and data[0][frase][letra+1] =="t" and data[0][frase][letra+2] =="t"):
				i=0
				while data[0][frase][letra+i] != " " and (apuntador+i+1) < longitud_frase:
					i=i+1
				apuntador=letra+i+1
				letra=letra+i
			else:
				if apuntador < longitud_frase:
					n_data.write(data[0][frase][apuntador])
					apuntador=apuntador+1
					j=j+1
				else:
					n_data.write("")
			letra=letra+1
		n_data.write(os.linesep)
	except:
		print("NO OK")
n_data.close()
