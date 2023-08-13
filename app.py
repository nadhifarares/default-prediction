import streamlit as st
import pandas as pd
import numpy  as np
import joblib

st.image('webheader.jpg')
st.header('Bank Danamon Customer Database')
st.write("""
This information is of a highly confidential nature and should be handled with the utmost care to ensure its security and privacy. Access to this data is restricted to authorized personnel only, and any dissemination or sharing of this information is strictly prohibited without proper authorization.
""")
         
@st.cache_data
def fetch_data():
    df = pd.read_csv('UCI_Credit_Card.csv')
    return df

df = fetch_data()

#Renaming df_raw for convenience
df.rename(columns={'ID': 'cust_id', 'LIMIT_BAL': 'limit_balance','SEX' : 'sex', 'EDUCATION' : 'education', 'MARRIAGE':'marriage', 'AGE' : 'age', 'PAY_0' : 'pay_0', 'PAY_2' : 'pay_2', 'PAY_3' : 'pay_3', 'PAY_4' : 'pay_4', 'PAY_5' : 'pay_5', 'PAY_6' : 'pay_6', 'BILL_AMT1' : 'bill_amt1', 'BILL_AMT2' : 'bill_amt2', 'BILL_AMT3' : 'bill_amt3', 'BILL_AMT4' : 'bill_amt4', 'BILL_AMT5' : 'bill_amt5', 'BILL_AMT6' : 'bill_amt6', 'PAY_AMT1' : 'pay_amt1', 'PAY_AMT2' : 'pay_amt2', 'PAY_AMT3' : 'pay_amt3', 'PAY_AMT4' : 'pay_amt4', 'PAY_AMT5' : 'pay_amt5', 'PAY_AMT6' : 'pay_amt6', 'default.payment.next.month' : 'default'}, inplace=True)

cust_id = st.number_input('Cust ID', value=0)
limit_balance = st.number_input('Limit Balance', value=0)

st.subheader("Repayment Status according to Customers")
st.write('Description:')
st.write('-2: Customer did not utilize their credit card for any transactions or payments during that specific month.')
st.write('0: Pay duly')
st.write('1: Payment delay for one month')
st.write('2: Payment delay for two months')
st.write('3: Payment delay for three months')
st.write('Etc.')

pay_0 = st.selectbox('Repayment status in September', sorted(df['pay_0'].unique()))
pay_2 = st.selectbox('Repayment status in August', sorted(df['pay_2'].unique()))
pay_3 = st.selectbox('Repayment status in July', sorted(df['pay_3'].unique()))
pay_4 = st.selectbox('Repayment status in June', sorted(df['pay_4'].unique()))
pay_5 = st.selectbox('Repayment status in May', sorted(df['pay_5'].unique()))
pay_6 = st.selectbox('Repayment status in April', sorted(df['pay_6'].unique()))

data = {
    'cust_id' : cust_id,
    'limit_balance' : limit_balance,
    'pay_0' : pay_0,
    'pay_2' : pay_2,
    'pay_3' : pay_3,
    'pay_4' : pay_4,
    'pay_5' : pay_5,
    'pay_6' : pay_6,
}
input = pd.DataFrame(data, index=[0])

st.subheader('Summary')
st.write(input)


load_model = joblib.load("danamonclas.pkl")

if st.button('Predict'):
    prediction = load_model.predict(input)

    if prediction[0] == 1:
        prediction_text = 'Will be defaulted.'
    else:
        prediction_text = 'Will not be defaulted.'

    st.write('Based on the input, the placement model predicted:')
    st.write(prediction_text)
