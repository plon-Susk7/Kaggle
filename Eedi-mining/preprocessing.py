import pandas as pd
import numpy as np

def cleanTrainData(df):

    options = ['A','B','C','D']
    originalCols = list(df.columns)

    newCols = originalCols
    newCols.append('AnswerText')
    newCols.append('MisconceptionId')
    
    
    newDf = pd.DataFrame(columns=newCols)

    for i in range(df.shape[0]):
    
        for option in options:
            new_row = df.iloc[i,:].copy()
            new_row['QuestionId'] = f"{new_row['QuestionId']}_{option}"
            new_row['AnswerText'] = new_row[f"Answer{option}Text"]
            new_row['MisconceptionId'] = new_row[f"Misconception{option}Id"]
            newDf = pd.concat([newDf, new_row.to_frame().T], ignore_index=True)

    newDf = newDf.drop(['ConstructId','SubjectId','AnswerAText','AnswerBText','AnswerCText','AnswerDText','MisconceptionAId','MisconceptionBId','MisconceptionCId','MisconceptionDId'],axis=1)

    return newDf.dropna()
            
            
def cleanTestData(df):
    options = ['A','B','C','D']
    originalCols = list(df.columns)

    newCols = originalCols
    newCols.append('AnswerText')
    
    
    newDf = pd.DataFrame(columns=newCols)

    for i in range(df.shape[0]):
    
        for option in options:
            if df.iloc[i,:]['CorrectAnswer'] == option:
                continue
            new_row = df.iloc[i,:].copy()
            new_row['QuestionId'] = f"{new_row['QuestionId']}_{option}"
            new_row['AnswerText'] = new_row[f"Answer{option}Text"]
            newDf = pd.concat([newDf, new_row.to_frame().T], ignore_index=True)

    newDf = newDf.drop(['ConstructId','SubjectId','AnswerAText','AnswerBText','AnswerCText','AnswerDText'],axis=1)

    return newDf.dropna()