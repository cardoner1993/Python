#!/usr/bin/env python
import json
import numpy as np
import urllib3
import sqlite3
from pandas.io.json import json_normalize
import time
import datetime
import pandas as pd
###monedes
###order book
import os.path
os.chdir("/opt/bitfinex")

##sa de tenir la data base de sqlite al mateix lloc que el working directory del python
##sino hem de ficar el path al open directory perque sapiga on es la db.
##Ara hem ficat al mateix dir tot la db i el script sino cambiar wd o posar aixo de sota del path

#BASE_DIR = os.path.dirname("C:\\Users\\David\\TutorialPython\\scripts")
#db_path = os.path.join(BASE_DIR, "crypto.db")
#conn = sqlite3.connect(db_path)
    
##Open a conection in the current wd to crypto db database created with the CREATE table and add the 3 tables.
conn = sqlite3.connect('crypto.db')
c = conn.cursor()
#save changes and then close. Debemos hacer el commit para guardar los cambios realizados.
##solo quedarse con los unique del orderbook porque es precio min de compra y venta del order book. Es decir toda la gente que quere
## comprar o vender al minimo precio de mercado dado el actual.
url = ["https://api.bitfinex.com/v1/book/"]
pref_cand = '/hist?limit=100'
url2 = ['https://api.bitfinex.com/v1/trades/','https://api.bitfinex.com/v1/stats/','https://api.bitfinex.com/v1/pubticker/','https://api.bitfinex.com/v2/candles/trade:5m:t']
opcions = ["_orderb","_trades","_stats","_pubticker"]
echange = "_bitfinex"
dic = ['ethbtc','gntbtc','ltcbtc','sanbtc','bchbtc']
dicpubt = ['ethusd','gntusd','iotusd']

tablen = ['trades','stats','ticker','candles']
###falta pasar timestamp a format correcte.
## aixo ho farem servir per manipular el format de les crides.
http = urllib3.PoolManager()
for k in range(0,len(url2)):
    str1 = str(url2[k])
    str2 = 'candles'
    print(str1.find(str2))
            
    for i in range(len(dicpubt)):
        
        if str1.find(str2) != -1:
            response = http.request('GET', url2[k]+dicpubt[i].upper()+'/hist?limit=100')   
            a=json.loads(response.data)
        else:
            response = http.request('GET', url2[k]+dicpubt[i].upper())   
            a=json.loads(response.data)
        str1 = str(url2[k])
        str2 = 'candles'
        print(str1.find(str2))
        if str1.find(str2) != -1:
               b=pd.DataFrame(a)
               b.columns = ["TIME","OPEN","CLOSE","HIGH","LOW","VOLUME"]
               coin = np.repeat(dicpubt[i],len(b))
               b.insert(loc=len(b.columns), column='COIN', value=coin)
               print(b)
        else:
               b=json_normalize(a)
               coin = np.repeat(dicpubt[i],len(b))
               b.insert(loc=len(b.columns), column='coin', value=coin)
               print(b)
                    
#"index amount exchange price tid timestamp type"
    #Trades
        tabla = tablen[k]
        if tabla == 'trades':
               for i in range(0,len(b)): 
                   b.iloc[i,4] = datetime.datetime.fromtimestamp(float(b.iloc[i,4])).strftime('%Y-%m-%d %H:%M:%S')
                   try:
                       c.execute('INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)',(str(b.index[i]),str(b.iloc[i,0]),str(b.iloc[i,1]),str(b.iloc[i,2]),str(b.iloc[i,3]),str(b.iloc[i,4]),str(b.iloc[i,5]),str(b.iloc[i,6])))
                   except sqlite3.IntegrityError:
                       pass 
# stats.
        elif tabla == 'stats':
               ts = time.time()
               st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
               timestamp = np.repeat(st,len(b))
               b.insert(loc=len(b.columns),column="timestamp",value=timestamp)
               for i in range(0,len(b)):  
                   try:
                       c.execute("INSERT INTO stats values (?, ?,?,?,?)", (str(b.index[i]),str(b.iloc[i,0]),str(b.iloc[i,1]),str(b.iloc[i,2]),str(b.iloc[i,3])))
                   except sqlite3.IntegrityError:
                       pass
##provar que pasa con json que es solo una linea. 
##ticker.
        elif tabla == 'ticker':
               for i in range(0,len(b)):
                   b.iloc[i,6] = datetime.datetime.fromtimestamp(float(b.iloc[i,6])).strftime('%Y-%m-%d %H:%M:%S')
                   try:
                       c.execute('INSERT INTO ticker VALUES (?,?,?,?,?,?,?,?,?,?)',(str(b.index[i]),str(b.iloc[i,0]),str(b.iloc[i,1]),str(b.iloc[i,2]),str(b.iloc[i,3]),str(b.iloc[i,4]),str(b.iloc[i,5]),str(b.iloc[i,6]),str(b.iloc[i,7]),str(b.iloc[i,8])))
                   except sqlite3.IntegrityError:
                       pass
           
        else:
               for i in range(0,len(b)):
                   try:
                       c.execute('INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)',(str(b.index[i]),str(b.iloc[i,0]),str(b.iloc[i,1]),str(b.iloc[i,2]),str(b.iloc[i,3]),str(b.iloc[i,4]),str(b.iloc[i,5]),str(b.iloc[i,6])))
                   except sqlite3.IntegrityError:
                       pass


conn.commit()
conn.close()
######
#pensar orderbook i ja haurem acabat.
############################
#Proves funcionament insert.
#candles
response = http.request('GET', 'https://api.bitfinex.com/v2/candles/trade:5m:tBTCUSD/hist?limit=10')   
a=json.loads(response.data) 
b=pd.DataFrame(a)


#ticker
response = http.request('GET', 'https://api.bitfinex.com/v1/pubticker/ethbtc')   
a=json.loads(response.data) 
b=json_normalize(a)

#stats
response = http.request('GET', 'https://api.bitfinex.com/v1/stats/ethbtc')   
a=json.loads(response.data) 
b=json_normalize(a)

#trades
response = http.request('GET', 'https://api.bitfinex.com/v1/trades/ethbtc')   
a=json.loads(response.data) 
b=json_normalize(a)

##Add order book or the current order book. remember that have asks and bids.

#soup = BeautifulSoup(response.data) 

##Si se quiere mirar alguna cartera respecto a usd
#if "extension de la url ej book o trades" in url[k]:
 #           response=requests.request("GET", url[k]+dicpubt[i]) 
#        else:
 
 