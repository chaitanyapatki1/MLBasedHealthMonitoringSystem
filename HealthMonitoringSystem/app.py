import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'assessment_history.sqlite3'
st.set_page_config(page_title='Respiratory Health Monitoring', page_icon='🫀', layout='wide')

@st.cache_resource
def assets():
    model = joblib.load(ROOT / 'respiratory_model.pkl')
    order = joblib.load(ROOT / 'feature_order.pkl')
    enc = joblib.load(ROOT / 'feature_encodings.pkl')
    if len(order) != 16 or model.n_features_in_ != 16:
        raise ValueError('Model and feature order must contain 16 features.')
    return model, order, enc

def database():
    with sqlite3.connect(DB) as db:
        db.execute('CREATE TABLE IF NOT EXISTS assessments (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, result TEXT NOT NULL, payload TEXT NOT NULL)')

def save_result(result, payload):
    with sqlite3.connect(DB) as db:
        db.execute('INSERT INTO assessments(created_at,result,payload) VALUES (?,?,?)', (datetime.now().isoformat(timespec='seconds'), result, json.dumps(payload)))

def latest_result():
    with sqlite3.connect(DB) as db:
        return db.execute('SELECT id,created_at,result,payload FROM assessments ORDER BY id DESC LIMIT 1').fetchone()

def binary(value, classes):
    if value not in ('Yes', 'No'):
        raise ValueError('An unanswered or uncertain symptom cannot be encoded.')
    label = 'True' if value == 'Yes' else 'False'
    classes = [str(x) for x in classes]
    if label not in classes:
        raise ValueError(f'Expected category {label} not present in original encoding.')
    return classes.index(label)

def warnings_for(survey, manual, sensor):
    w = []
    if survey.get('Breathing Difficulty') == 'Severe':
        w.append('Severe breathing difficulty: seek urgent medical attention.')
    if survey.get('Medicine Overdose') in ('Yes', 'Not Sure'):
        w.append('Possible medication overdose: seek immediate medical or poison-centre advice.')
    if manual.get('Respiratory Rate', 0) >= 30:
        w.append('Markedly elevated reported breathing rate: urgent assessment is recommended.')
    if 0 < sensor.get('spo2', 100) < 90:
        w.append('Low reported oxygen saturation: seek immediate medical attention.')
    return w

def prepare(sensor, survey, manual, order, enc):
    names = ['Dehydration','Medicine Overdose','Acidious','Cold','Cough']
    b = {name: binary(survey[name], enc[name]) for name in names}
    values = dict(b)
    values.update({
        'Cold .1': b['Cold'], 'Cough.1': b['Cough'],
        'Type': sum(b.values()),
        'Temperature': manual['body_temperature_c'] * 9 / 5 + 32,
        'Heart Rate': sensor['heart_rate'],
        'Pulse': manual['Pulse'],
        'BPSYS': manual['BPSYS'], 'BPDIA': manual['BPDIA'],
        'Respiratory Rate': manual['Respiratory Rate'],
        'Oxygen Saturation': sensor['spo2'] / 100,
        'PH': manual['PH'],
    })
    return pd.DataFrame([[values[n] for n in order]], columns=order)

database()
for name in ('sensor_data','survey_data','manual_data'):
    st.session_state.setdefault(name,{})

st.sidebar.title('Health Monitoring')
page = st.sidebar.radio('Navigation', ['Dashboard','Symptom Questionnaire','Additional Measurements','Assessment'])

if page == 'Dashboard':
    st.title('Respiratory Health Monitoring')
    st.subheader('Measurements')
    old = st.session_state.sensor_data
    with st.form('sensors'):
        c1,c2 = st.columns(2)
        with c1:
            hr = st.number_input('Heart Rate (BPM)', min_value=0.0, max_value=250.0, value=float(old.get('heart_rate',78.0)))
            rt = st.number_input('Room Temperature (°C)', value=float(old.get('room_temperature',28.0)))
        with c2:
            spo2 = st.number_input('SpO₂ (%)', min_value=0.0, max_value=100.0, value=float(old.get('spo2',98.0)))
            humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=float(old.get('humidity',55.0)))
        if st.form_submit_button('Save Measurements', use_container_width=True):
            if hr <= 0 or spo2 <= 0:
                st.error('Enter valid readings before continuing.')
            else:
                st.session_state.sensor_data = dict(heart_rate=hr,spo2=spo2,room_temperature=rt,humidity=humidity)
                st.success('Measurements saved.')
    if st.session_state.sensor_data:
        d=st.session_state.sensor_data
        for col,label,value in zip(st.columns(4),['Heart Rate','SpO₂','Room Temperature','Humidity'],[f"{d['heart_rate']:g} BPM",f"{d['spo2']:g}%",f"{d['room_temperature']:g} °C",f"{d['humidity']:g}%"]):
            col.metric(label,value)

elif page == 'Symptom Questionnaire':
    st.title('Symptom Questionnaire')
    old=st.session_state.survey_data
    def radio(label,key,options):
        v=old.get(key)
        return st.radio(label,options,index=options.index(v) if v in options else None)
    with st.form('survey'):
        cold=radio('Cold symptoms?','Cold',['No','Yes'])
        cough=radio('Cough?','Cough',['No','Yes'])
        dehydration=radio('Dehydration symptoms?','Dehydration',['No','Yes','Not Sure'])
        overdose=radio('Taken more medication than prescribed?','Medicine Overdose',['No','Yes','Not Sure'])
        acid=radio('Are you experiencing acidity or acid reflux symptoms?','Acidious',['No','Yes','Unknown'])
        breathing=radio('Breathing difficulty?','Breathing Difficulty',['No','Mild','Moderate','Severe'])
        duration=radio('Symptom duration?','Duration',['Less than 24 hours','1–3 days','4–7 days','More than 7 days'])
        if st.form_submit_button('Save Questionnaire',use_container_width=True):
            if any(x is None for x in (cold,cough,dehydration,overdose,acid,breathing,duration)):
                st.error('Please answer every question.')
            else:
                st.session_state.survey_data={'Cold':cold,'Cough':cough,'Dehydration':dehydration,'Medicine Overdose':overdose,'Acidious':acid,'Breathing Difficulty':breathing,'Duration':duration}
                st.success('Questionnaire saved.')
                for warning in warnings_for(st.session_state.survey_data,st.session_state.manual_data,st.session_state.sensor_data):
                    st.error(warning)

elif page == 'Additional Measurements':
    st.title('Additional Measurements')
    old=st.session_state.manual_data
    with st.form('manual'):
        a,b=st.columns(2)
        with a:
            sys=st.number_input('Systolic BP (mmHg)',min_value=1,max_value=300,value=old.get('BPSYS'))
            rr=st.number_input('Respiratory Rate (breaths/min, counted over 60 seconds)',min_value=1,max_value=100,value=old.get('Respiratory Rate'))
            pulse=st.number_input('Pulse (BPM, independently measured)',min_value=1,max_value=250,value=old.get('Pulse'))
        with b:
            dia=st.number_input('Diastolic BP (mmHg)',min_value=1,max_value=200,value=old.get('BPDIA'))
            temp=st.number_input('Body Temperature (°C, separate thermometer)',min_value=25.0,max_value=45.0,value=old.get('body_temperature_c'),step=0.1)
            ph=st.number_input('PH (original dataset scale 1–14; measured value only)',min_value=1,max_value=14,value=old.get('PH'))
        ph_confirm=st.checkbox('I have an actual measurement for PH and understand the original dataset definition.',value=False)
        if st.form_submit_button('Save Additional Measurements',use_container_width=True):
            if any(v is None for v in (sys,dia,rr,pulse,temp,ph)):
                st.error('All measurements are required.')
            elif sys<=dia:
                st.error('Check the blood pressure values.')
            elif not ph_confirm:
                st.error('PH must be independently measured and correctly defined; do not guess it.')
            else:
                st.session_state.manual_data={'BPSYS':sys,'BPDIA':dia,'Respiratory Rate':rr,'Pulse':pulse,'body_temperature_c':temp,'PH':ph,'ph_confirmed':True}
                st.success('Additional measurements saved.')

elif page == 'Assessment':
    st.title('Health Assessment Report')
    sensor=st.session_state.sensor_data
    survey=st.session_state.survey_data
    manual=st.session_state.manual_data
    if sensor and survey and manual:
        st.subheader('Measurements')
        c=st.columns(4)
        c[0].metric('Heart Rate',f"{sensor['heart_rate']:g} BPM")
        c[1].metric('SpO₂',f"{sensor['spo2']:g}%")
        c[2].metric('Body Temperature',f"{manual['body_temperature_c']:g} °C")
        c[3].metric('Respiratory Rate',f"{manual['Respiratory Rate']}/min")
        c=st.columns(3)
        c[0].metric('Pulse',f"{manual['Pulse']} BPM")
        c[1].metric('Blood Pressure',f"{manual['BPSYS']}/{manual['BPDIA']} mmHg")
        c[2].metric('PH (dataset field)',str(manual['PH']))
        st.subheader('Symptoms')
        st.dataframe(pd.DataFrame([{'Symptom':k,'Response':v} for k,v in survey.items()]),hide_index=True,use_container_width=True)
        flags=warnings_for(survey,manual,sensor)
        for flag in flags:
            st.error(flag)
        st.divider()
        st.subheader('ML Assessment')
        st.write('Generate an assessment from the currently entered measurements and questionnaire. Previous results remain saved until a new assessment is generated.')
        if st.button('Generate Assessment',type='primary',use_container_width=True):
            if survey.get('Acidious') not in ('Yes','No') or survey.get('Dehydration') not in ('Yes','No') or survey.get('Medicine Overdose') not in ('Yes','No'):
                st.error('Uncertain/undefined categorical values cannot be passed to the original model.')
            elif not manual.get('ph_confirmed'):
                st.error('A verified PH input is required.')
            else:
                try:
                    model,order,enc=assets()
                    frame=prepare(sensor,survey,manual,order,enc)
                    code=int(model.predict(frame)[0])
                    category={0:'Normal',1:'Mild',2:'Severe',3:'Chronic'}.get(code,f'Class {code}')
                    payload={'measurements':{**sensor,**manual},'survey':survey,'model_features':frame.iloc[0].to_dict(),'safety_alerts':flags}
                    save_result(category,payload)
                    st.success('Assessment generated and saved.')
                except (OSError,ValueError,KeyError,IndexError,ImportError,AttributeError) as exc:
                    st.error(f'Unable to generate assessment: {exc}')
    else:
        st.info('Complete the Dashboard, Symptom Questionnaire and Additional Measurements first.')

    record=latest_result()
    if record:
        rid,when,result,payload_text=record
        payload=json.loads(payload_text)
        st.divider()
        st.subheader('Saved ML Result')
        st.metric('Experimental SVM classification',result)
        st.caption(f'Assessment #{rid} · {when.replace("T"," ")}')
        if payload.get('safety_alerts'):
            st.error('Safety alerts were present at the time of this assessment. Do not use the model category to dismiss them.')
            for flag in payload['safety_alerts']:
                st.warning(flag)
        st.download_button('Download Assessment Report',data=json.dumps({'assessment_id':rid,'created_at':when,'experimental_ml_classification':result,**payload},indent=2),file_name=f'assessment_{rid}.json',mime='application/json',use_container_width=True)
