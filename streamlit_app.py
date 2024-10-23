import streamlit as st
import pandas as pd
import requests
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import pickle
from openai import OpenAI

client = OpenAI(
  base_url = "https://api.groq.com/openai/v1",
  api_key = st.secrets["GROQ_API_KEY"]
)


def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


xgboost_model = load_model('xgb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
gb_model = load_model('gb_model.pkl')
svm_model = load_model('svm_model.pkl')


def prepare_input(credit_score, location, gender, age, tenure, balance, 
                  num_products, has_credit_card, is_active_member, estimated_salary):
  
    input_dict = {
          'CreditScore': credit_score,
          'Age': age,
          'Tenure': tenure,
          'Balance': balance,
          'NumOfProducts': num_products,
          'HasCrCard': int(has_credit_card),
          'IsActiveMember': int(is_active_member),
          'EstimatedSalary': estimated_salary,
          'Geography_France': 1 if location == "France" else 0,
          'Geography_Germany': 1 if location == 'Germany' else 0,
          'Geography_Spain': 1 if location == 'Spain' else 0,
          'Gender_Female': 1 if gender == "Female" else 0,
          'Gender_Male': 1 if gender == 'Male'else 0,
          'CLV' : balance * estimated_salary,
          'TenureAgeRatio' : tenure / age,
          'AgeGroup_MiddleAge' : 1 if   30 < age <= 45 else 0,
          'AgeGroup_Senior' : 1 if   45 < age <= 60 else 0,
          'AgeGroup_Elderly' : 1 if   60 < age <= 100 else 0
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def prepare_json_input(credit_score, location, gender, age, tenure, balance, 
  num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
          'CreditScore': credit_score,
          'Geography': location,
          'Gender': gender,
          'Age': age,
          'Tenure': tenure,
          'Balance': balance,
          'NumOfProducts': num_products,
          'HasCrCard': int(has_credit_card),
          'IsActiveMember': int(is_active_member),
          'EstimatedSalary': estimated_salary
    }

    return input_dict
    


def make_predictions(input_df, input_dict):
      probabilities = [
        xgboost_model.predict_proba(input_df)[0][1] * 100,
        random_forest_model.predict_proba(input_df)[0][1] * 100,
        decision_tree_model.predict_proba(input_df)[0][1] * 100,
        gb_model.predict_proba(input_df)[0][1] * 100,
        svm_model.predict_proba(input_df)[0][1] * 100
      ]
      return probabilities
         


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:

      selected_customer_id = int(selected_customer_option.split(" - ")[0])
    
      selected_surname = selected_customer_option.split(" - ")[1]
    
      selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
    
      col1, col2 = st.columns(2)
    
      with col1:
    
            credit_score = st.number_input(
              "Credit Score",
              min_value=300,
              max_value=850,
              value=int(selected_customer['CreditScore']))
        
            location = st.selectbox(
              "Location", ["Spain", "France", "Germany"],
              index=["Spain", "France", "Germany"].index(selected_customer["Geography"]))
        
            gender = st.radio("Gender", ["Male", "Female"],
                              index=0 if selected_customer["Gender"] == "Male" else 1)
        
            age = st.number_input(
              "Age",
              min_value=18,
              max_value=100,
              value=int(selected_customer["Age"]))
        
            tenure = st.number_input(
              "Tenure (years)",
              min_value=0,
              max_value=50,
              value=int(selected_customer["Tenure"]))
    
      with col2:
    
            balance = st.number_input(
              "Balance",
              min_value=0.0,
              value=float(selected_customer["Balance"])
            )
        
            num_products = st.number_input(
              "Number of Products",
              min_value = 1,
              max_value = 10,
              value=int(selected_customer["NumOfProducts"])
            )
        
        
            has_credit_card = st.checkbox(
              "Has Credit Card",
              value=bool(selected_customer["HasCrCard"])
            )
        
            is_active_member = st.checkbox(
              "Is Active Member",
              value=bool(selected_customer["IsActiveMember"])
            )
        
            estimated_salary = st.number_input(
              "Estimated Salary",
              min_value = 0.0,
              value = float(selected_customer["EstimatedSalary"])
            )

            st.markdown("<br><br><br>", unsafe_allow_html=True)
            


      st.divider()
      col3, col4 = st.columns(2)


      with col3:
            api_url = "http://ec2-18-218-234-68.us-east-2.compute.amazonaws.com:80/predict"
            payload = prepare_json_input(credit_score, 
                                         location, 
                                         gender, 
                                         age, 
                                         tenure, 
                                         balance, 
                                         num_products, 
                                         has_credit_card, 
                                         is_active_member, 
                                         estimated_salary) 


            result = None
        
            try:
              response = requests.post(api_url, json=payload)  

              response.raise_for_status()  # Raise an error for bad responses

              # # Display the response from the API
              st.markdown('#### Churn Probability')
              result = response.json()  # Display the JSON response

            except requests.exceptions.RequestException as e:
              st.error(f"API call failed: {e}")


            val = result['Probability'][0][1] * 100 if result else 0
            bar_color = "#0aa60f"
            if 33 <= val < 66:
              bar_color = "yellow"
            elif 66 <= val <= 100:
              bar_color = "#bd2020"

        
            fig = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = val,
                mode = "gauge+number",
                gauge = {'axis': {'range': [0, 100],
                                  'tickvals': [0, 50, 100],  
                                  'ticktext': ["0", "50", "100"]},  
                         'borderwidth': 2,
                         'bordercolor': "white",
                         'bar': {'color': bar_color},
                         'steps' : [
                             {'range': [0, 33], 'color': "darkgreen"},
                             {'range': [33, 66], 'color': "#8f8b22"},
                             {'range': [66, 100], 'color': "#753030"}],
                         'threshold' : {'line': {'color': "white", 'width': 4}, 
                                        'thickness': 0.90, 'value': 100}}))

            fig.update_layout(
                height=300,  # Set the height of the chart
                width=300,   # Set the width of the chart
            )
            st.plotly_chart(fig)
            st.write(f"The customer has a {val:.2f}% probability of churning.")


      with col4:
            st.markdown('#### Model Predictions')

            input_df, input_dict = prepare_input(credit_score, location, gender, age,
             tenure, balance, num_products, has_credit_card, is_active_member,
             estimated_salary)


            categories = ['Xgboost', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'SVM']
            values = make_predictions(input_df, input_dict)


            fig = px.bar(x=values, y=categories, orientation='h', labels={'x': '% Probability', 'y': 'Models'})

            fig.update_layout(height = 400, width = 300)

            fig.update_xaxes(range=[0, 100])

            st.plotly_chart(fig)


      st.divider()

      st.markdown('## Explanation of Prediction')

      def explain_prediction(probability, input_dict, surname):
              prompt = f"""You are an expert data scientist at a bank, where you specialize in interpeting and explaining predictions of machine learning models.
              Your machine learning model has predicted that a customer named {surname} has a {round(probability, 1)}% probability of churning, based on the information provided below.

              - IMPORTANT: If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
              - IMPORTANT: If the customer has less than 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
              - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importance provided.
              Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.
              - Do not provide context statement such as "Here is a 3 sentence...", just explain the prediction
              
              Here is the customers information:
              {payload}
              
              Here are the machine learning model's top 10 most important features of predicting churn:
                  Feature           | Importance
                  -------------------------------
                  NumOfProducts     | 0.323888
                  IsActiveMember    | 0.164146
                  Age               | 0.109550
                  Geography_Germany | 0.091373
                  Balance           | 0.052786
                  Geography_France  | 0.045463
                  Gender_Female     | 0.045283
                  Geography_Spain   | 0.036585
                  CreditScore       | 0.032655
                  EstimatedSalary   | 0.032555
                  HasCrCard         | 0.031940
                  Tenure            | 0.030504
                  Gender_Male       | 0.000000
      
                  
              {pd.set_option('display.max_columns', None)}
              Here are summary statistics for churned customers:
              {df[df['Exited'] == 1].describe()}
              Here are summary statistics for non-churned customers:
              {df[df['Exited'] == 0].describe()}
              """
              # print("EXPLANATION PROMPT", prompt)
              try:
                  raw_response = client.chat.completions.create(
                      model = "llama3-70b-8192",
                      messages=[{
                          "role" : "user",
                          "content" : prompt
                      }],
                  )
                  return raw_response.choices[0].message.content
              except Exception as e:
                  print(f"Error making OpenAI request: {e}")
              return "Error: Could not genereate explanation."


      explanation = explain_prediction(val, input_dict, selected_surname)
      st.write(explanation)




  

      st.divider()

      st.markdown('## Generated Email to Customer')


      def generate_email(probability, input_dict, explanation, surname):
            prompt = f"""You are a manager at HS Bank. You are responsible for
            ensuring customers stay with the bank and are incentivized with various offers.
            
            You noticed a customer named {surname} has a {round(probability, 1)}% probability of churning.
            
            Here is the customer's information:
            {input_dict}
            
            Here is some explanation as to why the customer might be at risk of churning:
            {explanation}
            
            Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

            
            Make sure to list out a set of incentives to stay based on their information,
            in bullet point format. Don't ever mention the
            probability of churning, or the machine learning model to the customer."""
        
            raw_response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[{
                "role": "user",
                "content": prompt
                }]
            )
        
            # print("\n\nEMAIL PROMPT", prompt)
                  
            return raw_response.choices[0].message.content

      st.write(generate_email(val, payload, explanation, selected_surname))
            