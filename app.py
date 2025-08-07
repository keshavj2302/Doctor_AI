import streamlit as st
import main

st.title('Doctor AI')

st.subheader('Please enter yes / no for all the symptoms that you are facing')

options = ['No', 'Yes']

fever = st.selectbox('Do you have fever ?', options)

headache = st.selectbox('Do you have headaches ?', options)

nausea = st.selectbox(' nausea ?', options)

vomiting = st.selectbox('Do you have vomiting ?', options)

fatigue = st.selectbox('Do you feel fatigue also to your body ?', options)

joint_pain = st.selectbox('Do you have any join pain also ?', options)

skin_rash = st.selectbox('Do you have any skin rash also ?', options)

cough = st.selectbox('Do you have cough ?', options)

weight_loss = st.selectbox('Did you lose your weight suddenly ?', options)

yellow_eyes = st.selectbox('Do your eyes turns yellow ?', options)


params = {
    'fever': fever,
    'headache': headache,
    'nausea': nausea,
    'vomiting': vomiting,
    'fatigue': fatigue,
    'joint_pain': joint_pain,
    'skin_rash': skin_rash,
    'cough': cough,
    'weight_loss': weight_loss,
    'yellow_eyes': yellow_eyes
}

submit = st.button('Submit')

if(submit):
    X_INP = main.modify_imput_feature(params)

    Y_OUT = main.predict(X_INP)

    disease = main.decode(Y_OUT)

    st.write(' Possible disease :- ')

    st.success(disease)

    prescription = main.get_prescription(disease[0])

    st.write(' Please refer to these drugs for your symptoms :- ')

    st.success(prescription)







