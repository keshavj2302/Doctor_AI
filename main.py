import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


### Helper funcn from modify input feature to change the symptoms into it's numeric value
def change_to_numeric(status):
    if(status == 'Yes'):
        return 1
    
    return 0


### modify the input feature into it's encoded form

def modify_imput_feature(params):
    x = np.zeros((1, len(params)))
    ind = 0

    x[0][0] = change_to_numeric(params['fever'])
    x[0][1] = change_to_numeric(params['headache'])
    x[0][2] = change_to_numeric(params['nausea'])
    x[0][3] = change_to_numeric(params['vomiting'])
    x[0][4] = change_to_numeric(params['fatigue'])
    x[0][5] = change_to_numeric(params['joint_pain'])
    x[0][6] = change_to_numeric(params['skin_rash'])
    x[0][7] = change_to_numeric(params['cough'])
    x[0][8] = change_to_numeric(params['weight_loss'])
    x[0][9] = change_to_numeric(params['yellow_eyes'])
    
    return x

### Predict funcn to take inputs from the dropdowns

def predict(x_test):
    return clf.predict(x_test)

#### Get the disease name from it's encoded numeric value 

def decode(encoded_disease):
    return encoder.classes_[encoded_disease]


###### Funcn to get prescription from disease 

def get_prescription(disease):

    top_drugs = []

    print(type(prescription))

    drugs = prescription[disease]

    ind = 0

    for each_drugs in drugs:
        top_drugs.append(each_drugs)
        ind += 1
        if(ind == 4):
            break
    
    return top_drugs




##################################################
###### Disease Prediction from Symptoms   ########
##################################################

dataset = pd.read_csv('improved_disease_dataset.csv')

# dataset.head()

encoder = LabelEncoder()

dataset['disease'] = encoder.fit_transform(dataset['disease'])

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 42, test_size = 0.25)

# x_train.shape, x_test.shape, y_train.shape, y_test.shape
clf = RandomForestClassifier(random_state = 42)
clf.fit(X, Y)


##################################################
###### Drugs Prediction from disease   ###########
##################################################

disease_drug = pd.read_csv('Drug_Data.csv')

# disease_drug = disease_drug.drop(columns = ['Drug_Review', 'User_Rating', 'Date', 'Count_of_Reviews'])
disease_drug = disease_drug.iloc[:, :2]
disease_drug = disease_drug.rename(columns = {'drug_name': 'drugName', 'medical_condition': 'Prescribed_for'})
# disease_drug.head(20)
prescription = {}

for index, each_row in disease_drug.iterrows():
    if each_row['Prescribed_for'] in prescription:
        presc_array = prescription[each_row['Prescribed_for']]
        presc_array.add(each_row['drugName'])
        prescription[each_row['Prescribed_for']] = presc_array
    else:
        presc_array = set()
        presc_array.add(each_row['drugName'])
        prescription[each_row['Prescribed_for']] = presc_array
        
    






