import joblib
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns


from streamlit_option_menu import option_menu
st.markdown("<h1 style='text-align: center; color: #AED6F1;'>Retention Radar– A Predictive Customer Churn Web Application</h1>", unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
nav_option = option_menu(
    menu_title=None,
    options=["About", "Data Analysis", "Prediction"],
    icons=["house", "bar-chart", "lightbulb"],
    orientation="horizontal",
    default_index=0,
)

if nav_option == "About":
    title = """
    <h1 style='color:  #D4E6F1 ; font-size: 36px;'>What is customer churn?</h1>"""
    st.markdown(title, unsafe_allow_html=True)
    text = """
    <p style='text-align: Left; color: #D6EAF8; font-size: 18px;'>
    Customer churn, often referred to simply as churn, is a critical metric for businesses across various industries, 
    indicating the rate at which customers discontinue their relationship with a company or service. 
    Understanding and managing churn is essential for maintaining customer loyalty, sustaining revenue streams, and fostering long-term growth.
    </p>
    """

    st.markdown(text, unsafe_allow_html=True)
    title = """
    <h1 style='color:  #D4E6F1 ; font-size: 36px;'>Model Overview?</h1>"""
    st.markdown(title, unsafe_allow_html=True)
    text = """

    <p style='text-align: Left; color: #D6EAF8; font-size: 18px;'>
    Retention Radar is a sophisticated web application designed to tackle customer churn head-on. Using advanced machine learning algorithms, it predicts churn probabilities by analyzing historical data and customer behavior. The platform segments customers based on demographics and behavior, allowing businesses to tailor retention strategies. Real-time monitoring keeps businesses informed of key churn indicators for prompt action. Its intuitive dashboard offers dynamic visualization of churn predictions and performance metrics. Integration with existing systems streamlines workflows, automating customer communication. Retention Radar empowers businesses to minimize churn, drive revenue growth, and cultivate lasting customer relationships with actionable insights and predictive analytics.
    </p>
    """

    st.markdown(text, unsafe_allow_html=True)
elif nav_option == "Data Analysis":
    title = """
        <h1 style='color:  #D4E6F1 ; font-size: 30px;'>Churn Model Dataset Exploration</h1>"""
    st.markdown(title, unsafe_allow_html=True)

    data = pd.read_csv("./Churn_Modelling.csv")
    st.dataframe(data.head(), width=700)

    st.markdown("## Data Analysis Options")
    analysis_options = ["Show Dataset", "Histograms",
                        "Correlation Heatmap", "Pairplot", "Boxplot"]
    analysis_choice = st.selectbox("Choose an analysis type", analysis_options)

    if analysis_choice == "Show Dataset":
        st.dataframe(data, width=700, height=300)

    elif analysis_choice == "Histograms":
        selected_columns = st.multiselect(
            "Select columns to plot", data.columns, default=list(data.columns))
        fig, ax = plt.subplots(figsize=(10, 6))
        data[selected_columns].hist(ax=ax, bins=20, color="#107ab0")
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_choice == "Correlation Heatmap":
        plt.figure(figsize=(12, 8))
        numeric_data = data.select_dtypes(include=[np.number])
        sns.heatmap(numeric_data.corr(), annot=True,
                    fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot()

    elif analysis_choice == "Pairplot":
        st.markdown("### Select Columns for Pairplot")
        selected_columns = st.multiselect(
            "Select columns", data.columns, default=list(data.columns[:4]))
        if len(selected_columns) > 1:
            sns.pairplot(data[selected_columns])
            st.pyplot()
        else:
            st.error("Please select at least two columns.")

    elif analysis_choice == "Boxplot":
        st.markdown("### Select Column for Boxplot")
        selected_column = st.selectbox(
            "Select one column", list(data.columns)[3:13])
        fig, ax = plt.subplots()
        sns.boxplot(x=data[selected_column], color="lightblue")
        st.pyplot(fig)

elif nav_option == "Prediction":
    selected_model = None
    col1, col2, = st.columns(2)

    button_width = 350
    button_height = 50
    hover_color = "#9EFFF7"

    button_style = f"""
    <style>
    div.stButton > button {{
        width: {button_width}px;
        height: {button_height}px;
        border: 2px solid transparent; /* Set initial border */
    }}
    div.stButton > button:hover {{
        border: 2px solid {hover_color}; /* Change border color on hover */
    }}
    </style>
    """

    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = None

    with col1:
        st.write(button_style, unsafe_allow_html=True)
        if st.button("Logistic Regression"):
            st.session_state['selected_model'] = './LogisticRegression_pipeline.pkl'
        if st.button("K-Nearest Neighbors"):
            st.session_state['selected_model'] = './KNeighborsClassifier_pipeline.pkl'

    with col2:
        st.write(button_style, unsafe_allow_html=True)
        if st.button("Random Forest"):
            st.session_state['selected_model'] = './RandomForestClassifier_pipeline.pkl'
        if st.button("Support Vector Classifier"):
            st.session_state['selected_model'] = './SVC_pipeline.pkl'

    title = """
    <h1 style='text-align: center; color: #D4E6F1; font-size: 36px;'>Provide Input</h1>
    """

    st.markdown(title, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score:")
        options = ['MALE', 'FEMALE']
        selected_option_gender = st.selectbox("Gender", options)
        options = ['Spain', 'France', 'Germany']
        selected_option_geography = st.selectbox("Geography", options)
        balance = st.number_input("Balance:")
        numofProducts = st.number_input("NumOfProducts:")

    with col2:
        age = st.number_input("Age:")
        estimatedSalary = st.number_input("Salary:")
        tenure = st.number_input("Tenure:")
        isActiveMember = st.selectbox("IsActiveMember", [0, 1])
        hasCrCard = st.selectbox("Do you have Credit Card?", [0, 1])
    with st.expander("User Inputs"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Credit Score: {credit_score}")
            st.write(f"Geography: {selected_option_geography}")
            st.write("Gender:", selected_option_gender)
            st.write(f"Balance: {balance}")
            st.write(f"NumOfProducts: {numofProducts}")
        with col2:
            st.write(f"Age: {age}")
            st.write(f"Tenure: {tenure}")
            st.write(f"IsActiveMember: {isActiveMember}")
            st.write(f"EstimatedSalary: {estimatedSalary}")
            st.write(f"Do you have a Credit Card: {hasCrCard}")
    if st.button("PREDICT"):
        selected_model_path = st.session_state.get('selected_model')
        if selected_model_path is not None:
            model = joblib.load(selected_model_path)
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Geography': [selected_option_geography],
                'Gender': [selected_option_gender],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [numofProducts],
                'HasCrCard': [hasCrCard],
                'IsActiveMember': [isActiveMember],
                'EstimatedSalary': [estimatedSalary]
            })
            prediction = model.predict(input_data)
            prediction_text = f"Prediction: <span style='font-size:24px; color:#2ecc71; font-weight:bold;'>{'Churn' if prediction[0] == 1 else 'No Churn'}</span>"
            st.markdown(prediction_text, unsafe_allow_html=True)
        else:
            st.error("Please select a model before predicting.")
