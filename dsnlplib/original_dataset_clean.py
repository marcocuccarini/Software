#dataset clean

df = pd.read_csv('DSNLP_Dataset_2.1.csv')

df = df[df['Rep'] != 'ridimensionamento'] # dovrebbero essere 2 in meno
df = df[df['Rep'] != 'conseguenza'] # 1 in meno
df = df.dropna(subset=['QRep']) # 0 in meno
df = df.dropna(subset=['Rep']) # 6 in meno
try:
	df.drop(columns=['index'])

#df.dropna(subset=['Question']) # 3062 in meno


df.loc[df['QRep'] == 'dichiarazione di intenti', 'QRep'] = 'dichiarazione_intenti'

df.loc[df['QRep'] == 'riferimento obiettivo', 'QRep'] = 'riferimento_obiettivo'

df.loc[df['QRep'] == 'non risposta', 'QRep'] = 'non_risposta'


df.loc[df['Rep'] == 'dichiarazione di intenti', 'Rep'] = 'dichiarazione_intenti'

df.loc[df['Rep'] == 'riferimento obiettivo', 'Rep'] = 'riferimento_obiettivo'

df.loc[df['Rep'] == 'non risposta', 'Rep'] = 'non_risposta'

len(df)

df['p'] = df['Rep'].apply(augmentRep)

df['MacroCat'] = df['p'].apply(lambda x: x[0])

df['text'] = df['Answer']

df['label'] = df['MacroCat']

df['label2'] = df['Rep']