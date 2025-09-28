import streamlit as st
import pandas as pd

# Simple predictor class
class SimpleInjuryPredictor:
    def predict_risk(self, player_data):
        risk_score = 0
        risk_score += max(0, (player_data['age'] - 25) * 2)
        risk_score += max(0, (player_data['training_hours_week'] - 20) * 1.5)
        risk_score += (player_data['fatigue_level'] - 5) * 3
        risk_score += player_data['previous_injuries_count'] * 8
        risk_score += max(0, (7 - player_data['sleep_hours_day']) * 4)
        risk_score += max(0, (70 - player_data['fitness_score']) * 0.5)

        if risk_score > 50:
            risk_level, confidence = 'High', 0.85
            probs = {'High': 0.7, 'Medium': 0.2, 'Low': 0.1}
        elif risk_score > 25:
            risk_level, confidence = 'Medium', 0.75
            probs = {'High': 0.2, 'Medium': 0.6, 'Low': 0.2}
        else:
            risk_level, confidence = 'Low', 0.80
            probs = {'High': 0.1, 'Medium': 0.2, 'Low': 0.7}

        return {
            'risk_level': risk_level,
            'risk_score': min(100, max(0, int(risk_score))),
            'confidence': confidence,
            'probabilities': probs,
            'recommendations': self.get_recommendations(risk_level)
        }
    
    def get_recommendations(self, risk_level):
        if risk_level == 'High':
            return [
                "Reduce training intensity by 40-50%",
                "Consult team physiotherapist immediately",
                "Take 2-3 days complete rest",
                "Focus on recovery and hydration"
            ]
        elif risk_level == 'Medium':
            return [
                "Reduce training by 20-30%",
                "Increase recovery time between sessions",
                "Focus on sleep quality (7-8 hours)",
                "Consider sports massage"
            ]
        else:
            return [
                "Maintain current training regimen",
                "Continue good recovery practices",
                "Monitor fatigue levels weekly",
                "Maintain balanced nutrition"
            ]

# Streamlit app
st.set_page_config(page_title="Cricket Injury Predictor", layout="wide")
st.title("Cricket Player Injury Risk Predictor")
st.write("Predict injury risk for cricket players using fitness analytics")

predictor = SimpleInjuryPredictor()

st.header("Player Assessment")
col1, col2 = st.columns(2)

with col1:
    player_type = st.selectbox("Player Type", ["Batsman", "Bowler", "All-Rounder"])
    age = st.slider("Age", 18, 40, 25)
    training_hours = st.slider("Training Hours/Week", 10, 40, 20)
    matches = st.slider("Matches/Month", 1, 10, 4)

with col2:
    fatigue = st.slider("Fatigue Level (1-10)", 1, 10, 5)
    sleep = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.5)
    injuries = st.slider("Previous Injuries", 0, 10, 1)
    fitness = st.slider("Fitness Score", 40, 100, 75)

if st.button("Predict Injury Risk"):  # no type="primary"
    player_data = {
        'player_type': player_type,
        'age': age,
        'training_hours_week': training_hours,
        'matches_last_month': matches,
        'fatigue_level': fatigue,
        'sleep_hours_day': sleep,
        'previous_injuries_count': injuries,
        'fitness_score': fitness
    }
    
    result = predictor.predict_risk(player_data)
    
    st.success("### Prediction Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level", result['risk_level'])
    col2.metric("Risk Score", f"{result['risk_score']}/100")
    col3.metric("Confidence", f"{result['confidence']:.1%}")
    
    st.write("### Probability Distribution")
    prob_df = pd.DataFrame(list(result['probabilities'].items()), 
                          columns=['Risk Level', 'Probability'])
    st.bar_chart(prob_df.set_index('Risk Level'))
    
    st.write("### Recommendations")
    for i, rec in enumerate(result['recommendations'], 1):
        st.write(f"{i}. {rec}")

st.info("💡 **Tip:** Adjust the sliders to see how different factors affect injury risk!")
