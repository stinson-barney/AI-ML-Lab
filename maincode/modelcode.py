# %%
import pickle
import numpy as np
import pandas as pd
import math 
import os
import glob

def predict(file):
    classifier = open("KNN_model.pkl","rb")
    classifier = pickle.load(classifier)

    # Load the scaler and encoder from disk
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    # %%
    folders = ['\Test']
    colnum = 476
    merged_df = pd.DataFrame(columns = range(0,51))

    # %%
    # for folder in folders:
    #     path = os.getcwd() + folder
    #     csv_files = glob.glob(os.path.join(path, "*.csv"))
        
    #     for file in csv_files:
    df = pd.read_csv(file, header=None)
    prev_row=[]
    dist=[]
    
    for i in range(0, colnum+1):
        j=0
        if(df.shape[0]-1 <= i): #if records are less then append 0
            x=0
            y=0
        else:
            for row in df.iloc[i,:]: #get x and y from row
                if(j==0):
                    x = row
                if(j==1):
                    y= row
                j+=1

        if prev_row: 
            if(x==0 and y==0): # if there is no record left then distance =0
                distance=0;
            else:
                distance = round(math.sqrt(((x - prev_row[0])**2) + ((y-prev_row[1])**2)),2) #Euclidean distance
                distance= round(distance*distance,0)
            dist.append(distance)
            prev_row.clear() #update previous row
            prev_row.append(x)
            prev_row.append(y) 
        else:
            prev_row.append(x)
            prev_row.append(y)
    
    temp = pd.Series(x for x in dist) #convert it to series to append to dataframe
    temp = temp.sort_values()
    temp_1=temp[-50:].copy()
    temp_1 = temp_1.reset_index(drop=True)
    
    filecolname = str(temp_1.shape[0])
        
    
    temp_1[filecolname] = file[-20:]
    
    merged_df = pd.concat([merged_df,temp_1.to_frame().T], axis=0, ignore_index=True)

    # %%
    merged_df.drop([50],axis=1, inplace=True)

    # %%
    merged_df

    # %%
    data_df = merged_df.iloc[:,:-1]

    # %% [markdown]
    # ## Scaling

    # %%
    X_test = scaler.transform(data_df)

    # %% [markdown]
    # ## Predicting

    # %%
    y_pred = classifier.predict(X_test)

    # %% [markdown]
    # ## Prediction

    # %%
    result = encoder.inverse_transform(y_pred)
    return(result[0])


