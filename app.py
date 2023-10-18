import pandas as pd
data=pd.read_csv('dataset.csv')
data=data.drop('trouble.concentrating',axis=1)
for i in data.columns:
    data[i]=data[i].replace({'yes':True,'no':False})
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.13, random_state=0)
from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators=64, max_depth=6, oob_score=True)
RF_classifier.fit(x_train,y_train)
y_pred = RF_classifier.predict(x_test)

import gradio as gr
def condition(nervous:bool,panic:bool,breathing:bool,sweating:bool,concentration:bool,sleeping:bool,work:bool,hopelessness:bool,anger:bool,react:bool,eating:bool,suicidal:bool,tired:bool,friend:bool,social:bool,weight:bool,material:bool,introvert:bool,memory:bool,nightmares:bool,avoids:bool,negative:bool,blamming:bool):
    disorder =  RF_classifier.predict([[nervous,panic,breathing,sweating,concentration,sleeping,work,hopelessness,anger,react,eating,suicidal,tired,friend,social,weight,material,introvert,memory,nightmares,avoids,negative,blamming]])
    disorder=list(disorder)
    return disorder[0]
desc="Check the options below and submit to know your Mental health condition"

iface=gr.Interface(fn=condition,
                  inputs=[gr.components.Checkbox(label='Feeling Nervous?'),
                          gr.components.Checkbox(label='Feeling Panic?'),
                          gr.components.Checkbox(label='Breathing rapidly?'),
                          gr.components.Checkbox(label='Getting Sweat frequently?'),
                          gr.components.Checkbox(label='Facing trouble while concentrating?'),
                          gr.components.Checkbox(label='Having trouble in sleeping?'),
                          gr.components.Checkbox(label='Having trouble with work?'),
                          gr.components.Checkbox(label='Feeling hopeless?'),
                          gr.components.Checkbox(label='Anger issues?'),
                          gr.components.Checkbox(label='Over reacting to things?'),
                          gr.components.Checkbox(label='Change in Eating?'),
                          gr.components.Checkbox(label='Getting Suicidal thoughts?'),
                          gr.components.Checkbox(label='Feeling Tired?'),
                          gr.components.Checkbox(label='Have close friends?'),
                          gr.components.Checkbox(label='Addicted to Social Media?'),
                          gr.components.Checkbox(label='Gaining Weight?'),
                          gr.components.Checkbox(label='Material Possessive?'),
                          gr.components.Checkbox(label='Introvert?'),
                          gr.components.Checkbox(label='Popping up Stressful Memories?'),
                          gr.components.Checkbox(label='Getting Nightmares?'),
                          gr.components.Checkbox(label='Avoids people or activities?'),
                          gr.components.Checkbox(label='Feeling Negative?'),
                          gr.components.Checkbox(label='Blamming yourself?')],
                   outputs=gr.components.Textbox(label='Your Mental Health Condition is'),
                   title="Mental Health Predictor",
                   description=desc
                  )
iface.launch(share = True)