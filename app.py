import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("fitness_model_final.pkl")
explainer = shap.TreeExplainer(model)

st.sidebar.title("Links:")
st.sidebar.markdown("[GitHub](https://github.com/Abhid234) | [LinkedIn](https://www.linkedin.com/in/abhiram-darbha)")

if "step" not in st.session_state:
    st.session_state.step = 1

if "info" not in st.session_state:
    st.session_state.info = {}


def go_to_step1():
    st.session_state.step = 1


if st.session_state.step == 1:
    st.title("Fitness Score Predictor")
    st.write("Enter your stats to get a predicted fitness score out of 100.")

    form_value = {
        "age": None,
        "gender": None,
        "weight_kg": None,
        "height_cm": None,
        "resting_heart_rate": None,
        "calorie_intake": None,
        "body_fat_pct": None,
        "lean_mass_pct": None,
        "sleep_hours": None,
        "workout_minutes": None,
        "daily_activity_minutes": None,
    }

    with st.form(key="input"):
        col1, col2 = st.columns(2)

        with col1:
            form_value["age"] = st.number_input("Age", min_value=18, max_value=80, value=None, placeholder="Age (18-80)")
            form_value["weight_kg"] = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=None, placeholder="Weight in kg (30-200)")
            form_value["resting_heart_rate"] = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=None, placeholder="Resting BPM (40-120)")
            form_value["body_fat_pct"] = st.number_input("Body Fat %", min_value=3.0, max_value=50.0, value=None, placeholder="Body fat % (3-50)")
            form_value["sleep_hours"] = st.number_input("Sleep Hours", min_value=3.0, max_value=11.0, value=None, placeholder="Hours of sleep (3-11)")
            form_value["daily_activity_minutes"] = st.number_input("Daily Activity Minutes", min_value=0, max_value=720, value=None, placeholder="Daily activity min (0-720)")

        with col2:
            form_value["gender"] = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select gender")
            form_value["height_cm"] = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=None, placeholder="Height in cm (120-220)")
            form_value["calorie_intake"] = st.number_input("Daily Calorie Intake", min_value=800, max_value=5000, value=None, placeholder="Calories/day (800-5000)")
            form_value["lean_mass_pct"] = st.number_input("Lean Mass %", min_value=30.0, max_value=95.0, value=None, placeholder="Lean mass % (30-95)")
            form_value["workout_minutes"] = st.number_input("Workout Minutes (weekly)", min_value=0, max_value=1500, value=None, placeholder="Weekly workout min (0-1500)")

        submit_button = st.form_submit_button("Predict Fitness Score")

    if submit_button:
        if any(v is None for v in form_value.values()):
            st.warning("Please fill in all the details")
        else:
            st.session_state.info = form_value
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    form_value = st.session_state.info

    input_df = pd.DataFrame([{
        "age": form_value["age"],
        "gender": 1 if form_value["gender"] == "Male" else 0,
        "weight_kg": form_value["weight_kg"],
        "height_cm": form_value["height_cm"],
        "resting_heart_rate": form_value["resting_heart_rate"],
        "calorie_intake": form_value["calorie_intake"],
        "body_fat_pct": form_value["body_fat_pct"],
        "lean_mass_pct": form_value["lean_mass_pct"],
        "sleep_hours": form_value["sleep_hours"],
        "workout_minutes": form_value["workout_minutes"],
        "daily_activity_minutes": form_value["daily_activity_minutes"],
    }])

    score = model.predict(input_df)[0]

    st.title("Your Result")

    if score >= 80:
        st.balloons()
        st.success(f"Excellent score!")
    elif score >= 60:
        st.info(f"Good score!")
    elif score >= 50:
        st.info(f"Average score")    
    else:
        st.warning(f"Must improve")

    st.metric("Predicted Fitness Score", f"{score:.1f} / 100")

    st.subheader("What's driving your score")
    shap_values = explainer(input_df)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.button("Back", on_click=go_to_step1)