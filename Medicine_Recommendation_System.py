import pandas as pd
dataset=pd.read_csv('medicine recommendation system dataset/Training.csv')
dataset.head()
dataset.shape
len(dataset['prognosis'].unique())
dataset['prognosis'].unique()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = dataset.drop("prognosis",axis=1)
y = dataset['prognosis']
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)
Y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 20)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
#create a dictionary to store models
models = {
    'SVC' : SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100,random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "MultinomialNB": MultinomialNB()
}
# Convert DataFrame to NumPy arrays
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)

# Train and test models
for model_name, model in models.items():
    # Train model
    model.fit(X_train_np, y_train)
    
    # Test model
    predictions = model.predict(X_test_np)
        
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
        
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)
        
    print(f"{model_name} accuracy : {accuracy}")
    print(f"{model_name} Confusion Matrix:")
    print(np.array2string(cm, separator=', '))
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
ypred = svc.predict(X_test)
print(f"SVC accuracy: {accuracy_score(y_test, ypred)}")
# saving model
import pickle
pickle.dump(svc, open("models/svc.pkl", "wb"))
# load model
svc = pickle.load(open("models/svc.pkl", 'rb'))
# Convert the first test sample to a NumPy array and reshape it
sample = X_test.iloc[0].to_numpy().reshape(1, -1)
prediction = svc.predict(sample)
print(prediction)
y_test[10]
# Assuming X_test is a DataFrame and y_test is an array or Series
print("Predicted Label: ", svc.predict(X_test.iloc[10].to_numpy().reshape(1, -1)))
print("Actual Label:", y_test[10])
sym_des = pd.read_csv('medicine recommendation system dataset/symtoms_df.csv')
sym_des
precautions = pd.read_csv('medicine recommendation system dataset/precautions_df.csv')
precautions
workout = pd.read_csv('medicine recommendation system dataset/workout_df.csv')
workout
description = pd.read_csv('medicine recommendation system dataset/description.csv')
description
medications = pd.read_csv('medicine recommendation system dataset/medications.csv')
medications
diets = pd.read_csv('medicine recommendation system dataset/diets.csv')
diets

# custome and helping functions
#helper funtions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values[0]
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]
    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]
import numpy as np

# Ensure the get_predicted_value and helper functions are defined
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values[0]
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values.flatten()]
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]
    wrkout = workout[workout['disease'] == dis]['workout']
    
    return desc, pre, med, die, wrkout

# Define the input handling and prediction logic
symptoms = input("Enter your symptoms.......")
user_symptoms = [s.strip() for s in symptoms.split(',')]
user_symptoms = [symptom.strip() for symptom in user_symptoms]

# Ensure user symptoms are valid
user_symptoms = [symptom for symptom in user_symptoms if symptom in symptoms_dict]

if not user_symptoms:
    print("Invalid symptoms entered. Please try again with valid symptoms.")
else:
    try:
        predicted_disease = get_predicted_value(user_symptoms)
        desc, pre, med, die, wrkout = helper(predicted_disease)

        print("=================predicted disease============")
        print(predicted_disease)
        print("=================description==================")
        print(desc)
        print("=================precautions==================")
        for i, p_i in enumerate(pre, start=1):
            print(f"{i} : {p_i}")

        print("=================medications==================")
        for i, m_i in enumerate(med, start=len(pre)+1):
            print(f"{i} : {m_i}")

        print("=================workout==================")
        for i, w_i in enumerate(wrkout, start=len(pre) + len(med) + 1):
            print(f"{i} : {w_i}")

        print("=================diets==================")
        for i, d_i in enumerate(die, start=len(pre) + len(med) + len(wrkout) + 1):
            print(f"{i} : {d_i}")

        print("=================medications==================")
        i = 1
        for m_i in med:
            print(f"{i} : {m_i}")
            i += 1

    except NameError as e:
        print("An error occurred: ", e)
