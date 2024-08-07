import streamlit as st
import numpy as np
import joblib



scalar = joblib.load("Scalar.pkl")


st.title('Resturant Rating Prediction App')

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/58a77daf-3a13-4866-a645-41364f081700/width=450/360802.jpeg");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.divider()



averagecost = st.number_input("Please enter the estimated cost for two",min_value=50,max_value=999999,value=1000,step=200)

tablebooking = st.selectbox('Resturant has table booking ?',['Yes','No'])

onlinedelivery = st.selectbox('Resturant has online booking ?',['Yes','No'])

pricerange = st.selectbox('What is price range (1 Cheapest, 4 Most Expensive)',[1,2,3,4])

predictbutton = st.button('Predict the review!')

st.divider()

model = joblib.load("ML_MODEL.pkl")

bookingstatus = 1 if tablebooking=='Yes' else 0
deliverystatus = 1 if onlinedelivery=='Yes' else 0

X = [[averagecost,bookingstatus,deliverystatus,pricerange]]
X = np.array(X)

X = scalar.transform(X)

if predictbutton:
   
    prediction = model.predict(X)

    if prediction>=1.8 and prediction<2.5:
        st.write('Poor')
    elif prediction>=2.5 and prediction<=3.4:
        st.write("Average")
    elif prediction>=3.5 and prediction<=3.9:
        st.write("Good")
    elif prediction>=4.0 and prediction<=4.4:
        st.write("Very Good")
    elif prediction>=4.5 and prediction<=5:
        st.write("Excellent")
    else:
        st.write("Not Rated")
   


