import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# Load model and preprocessor
model = joblib.load('random_forest_income_model.pkl')
preprocessor = joblib.load('income_preprocessor.pkl')

# Set professional page icon and title
st.set_page_config(page_title="Income Predictor", page_icon="📈", layout="centered")
st.title("💼 Employee Salary Prediction ")
st.markdown("Predict whether income is >50K or <=50K based on user input.")

# Input form
def user_input():
    age = st.slider(" Age", 18, 90, 30)
    workclass = st.selectbox("🏢 Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])

    job_type = st.selectbox("🕒 Job Type", ['Full-time', 'Part-time', 'Over-time'])

    education = st.selectbox("🎓 Education", [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate'])

    marital_status = st.selectbox("💍 Marital Status", [
        'Never-married', 'Married-civ-spouse', 'Divorced', 'Separated',
        'Widowed', 'Married-spouse-absent'])

    occupation = st.selectbox("💼 Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical'])

    relationship = st.selectbox(" Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family',
        'Other-relative', 'Unmarried'])

    race = st.selectbox(" Race", [
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])

    sex = st.radio(" Gender", ['Male', 'Female'])

    hours_per_week = st.slider("⏱️ Hours per Week", 1, 100, 40)
    experience = st.slider(" Experience (Years)", 0, 40, 5)

    native_country = st.selectbox(" Native Country", [
        'United-States', 'India', 'Philippines', 'Germany', 'Canada'])

    # Dictionary for input
    data = {
        'age': age,
        'workclass': workclass,
        'job-type': job_type,
        'education': education,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'hours-per-week': hours_per_week,
        'experience': experience,
        'native-country': native_country
    }

    return pd.DataFrame([data])

# Gather input
input_df = user_input()

# Predict on button click
if st.button("📤 Predict Income"):
    try:
        input_encoded = preprocessor.transform(input_df)
        prediction = model.predict(input_encoded)
        prob = model.predict_proba(input_encoded)[0][1]

        result = ">50K" if prediction[0] == 1 else "<=50K"
        st.success(f"✅ Predicted Income: **{result}**")
        st.info(f"📌 Model Confidence (Probability of >50K): **{prob:.2%}**")

                # 🔎 Natural language summary
        st.markdown("### 📝 Prediction Summary")
        summary = f"""
        Based on the provided details — a {input_df['age'][0]} year-old individual with {input_df['education'][0]} education,
        working in the {input_df['occupation'][0]} sector for approximately {input_df['experience'][0]} years — the model predicts
        their income to be **"{result}"**. The model is **{prob:.2%} confident** in this prediction.
        """
        st.markdown(summary)

        # 📊 Input Summary Table
        st.subheader("🔍 Input Summary Table")
        st.dataframe(input_df)

        

        # 📊 Feature Comparison Graph
        st.subheader("📊 Feature Comparison Graph")
        features = ['age', 'hours-per-week', 'experience']
        fig_bar = go.Figure(data=[ # type: ignore
            go.Bar(name='Input Features', x=features, y=[ # type: ignore
                input_df['age'][0],
                input_df['hours-per-week'][0],
                input_df['experience'][0]
            ])
        ])
        fig_bar.update_layout(title="User Feature Profile", yaxis_title="Value", xaxis_title="Feature")
        st.plotly_chart(fig_bar)


        # 🔍 Show input summary AFTER prediction
        st.subheader("🔍 Input Summary Table")
        st.dataframe(input_df)

        st.subheader("📊 Input Feature Graphs")
        numeric_vals = pd.DataFrame({
            'Age': [input_df['age'][0]],
            'Hours/Week': [input_df['hours-per-week'][0]],
            'Experience': [input_df['experience'][0]]
        }).T
        numeric_vals.columns = ['Value']
        st.bar_chart(numeric_vals)

    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
