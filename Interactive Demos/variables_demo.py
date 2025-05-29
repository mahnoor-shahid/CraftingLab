import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="Interactive Variables Learning Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .concept-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .example-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎓 Interactive Variables Learning Hub</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="concept-box">
    <h3>🎯 Welcome to the Ultimate Variables Demo!</h3>
    <p>This interactive learning platform helps you master different types of variables in data science, 
    statistics, and machine learning through hands-on examples and real-world scenarios.</p>
    <p><strong>🚀 What makes this special:</strong></p>
    <ul>
        <li>🎮 Interactive controls to see variables in action</li>
        <li>📊 Multiple real-world examples for each concept</li>
        <li>🔬 Experiment with different scenarios</li>
        <li>💡 Learn through visualization and exploration</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("Choose a concept to explore:")

page = st.sidebar.selectbox(
    "Select Variable Type:",
    [
        "🎯 Independent & Dependent Variables",
        "🔄 Interaction Variables", 
        "👻 Latent Variables",
        "🌪️ Confounding Variables",
        "🎛️ Control Variables",
        "🔗 Correlated Variables",
        "💧 Leaky Variables",
        "📈 Stationary vs Non-stationary",
        "⏰ Lagged Variables",
        "🎲 Variable Relationships Quiz"
    ]
)

# Helper functions
def create_sample_data(n=200, seed=42):
    """Create realistic sample datasets"""
    np.random.seed(seed)
    
    # Base variables
    temperature = np.random.normal(75, 15, n)
    temperature = np.clip(temperature, 40, 110)  # Realistic temperature range
    
    income = np.random.lognormal(10.5, 0.5, n)  # Log-normal for realistic income distribution
    age = np.random.gamma(2, 20, n)  # Gamma distribution for age
    age = np.clip(age, 18, 80)
    
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, 
                                     p=[0.4, 0.35, 0.2, 0.05])
    
    city_size = np.random.choice(['Small', 'Medium', 'Large'], n, p=[0.3, 0.4, 0.3])
    
    # Create realistic relationships
    ice_cream_sales = (temperature - 40) * 3 + np.random.normal(0, 20, n)
    ice_cream_sales = np.clip(ice_cream_sales, 0, None)
    
    ac_sales = (temperature - 60) * 2 + np.random.normal(0, 15, n)
    ac_sales = np.clip(ac_sales, 0, None)
    
    return pd.DataFrame({
        'temperature': temperature,
        'income': income,
        'age': age,
        'education': education_level,
        'city_size': city_size,
        'ice_cream_sales': ice_cream_sales,
        'ac_sales': ac_sales
    })

def create_business_scenario():
    """Create business-related data scenarios"""
    np.random.seed(42)
    n = 500
    
    # Marketing data
    ad_spend = np.random.exponential(1000, n)
    social_media_followers = np.random.lognormal(8, 1, n)
    website_traffic = ad_spend * 2 + social_media_followers * 0.1 + np.random.normal(0, 500, n)
    
    # Sales influenced by multiple factors
    base_sales = 1000
    sales = (base_sales + 
             ad_spend * 0.5 + 
             website_traffic * 0.2 + 
             np.random.normal(0, 200, n))
    sales = np.clip(sales, 0, None)
    
    return pd.DataFrame({
        'ad_spend': ad_spend,
        'social_followers': social_media_followers,
        'website_traffic': website_traffic,
        'sales': sales,
        'month': np.tile(range(1, 13), n//12 + 1)[:n]
    })

# Main content based on page selection
if page == "🎯 Independent & Dependent Variables":
    st.markdown('<h2 class="section-header">🎯 Independent & Dependent Variables</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Independent Variable (X):</strong> The variable YOU control or manipulate (the cause)</p>
        <p><strong>Dependent Variable (Y):</strong> The variable that RESPONDS to changes in X (the effect)</p>
        <p><strong>Key Insight:</strong> X → Y (X influences Y, not the other way around)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example selector
    example = st.selectbox("Choose a scenario to explore:", [
        "🍦 Temperature → Ice Cream Sales",
        "💰 Advertising → Sales Revenue", 
        "🎓 Study Hours → Test Scores",
        "🏥 Drug Dosage → Recovery Rate",
        "🌱 Fertilizer → Plant Growth"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if "Temperature" in example:
            st.markdown("""
            <div class="example-box">
                <h4>🍦 Ice Cream Business Scenario</h4>
                <p><strong>Question:</strong> How does temperature affect ice cream sales?</p>
                <p><strong>Independent Variable:</strong> Temperature (°F)</p>
                <p><strong>Dependent Variable:</strong> Ice cream sales ($)</p>
                <p><strong>Hypothesis:</strong> Higher temperatures → Higher sales</p>
            </div>
            """, unsafe_allow_html=True)
            
            temp_min = st.slider("Minimum Temperature (°F)", 40, 80, 50)
            temp_max = st.slider("Maximum Temperature (°F)", 80, 110, 100)
            sample_size = st.slider("Sample Size", 50, 500, 200)
            add_noise = st.slider("Add Random Variation", 0, 50, 20)
            
        elif "Advertising" in example:
            st.markdown("""
            <div class="example-box">
                <h4>💰 Marketing Campaign Scenario</h4>
                <p><strong>Question:</strong> How does ad spending affect sales?</p>
                <p><strong>Independent Variable:</strong> Ad Spend ($)</p>
                <p><strong>Dependent Variable:</strong> Sales Revenue ($)</p>
                <p><strong>Hypothesis:</strong> More advertising → More sales</p>
            </div>
            """, unsafe_allow_html=True)
            
            ad_budget = st.slider("Max Ad Budget ($)", 1000, 10000, 5000)
            roi_rate = st.slider("Return on Investment Rate", 1.5, 5.0, 3.0)
            market_saturation = st.slider("Market Saturation Effect", 0.0, 1.0, 0.3)
            
        elif "Study Hours" in example:
            st.markdown("""
            <div class="example-box">
                <h4>🎓 Education Scenario</h4>
                <p><strong>Question:</strong> How do study hours affect test scores?</p>
                <p><strong>Independent Variable:</strong> Study Hours</p>
                <p><strong>Dependent Variable:</strong> Test Score (%)</p>
                <p><strong>Hypothesis:</strong> More study → Better scores (with diminishing returns)</p>
            </div>
            """, unsafe_allow_html=True)
            
            max_hours = st.slider("Maximum Study Hours", 10, 40, 25)
            base_ability = st.slider("Student Base Ability", 40, 80, 60)
            diminishing_returns = st.slider("Diminishing Returns Factor", 0.1, 0.9, 0.5)
        
        elif "Drug Dosage" in example:
            st.markdown("""
            <div class="example-box">
                <h4>🏥 Medical Research Scenario</h4>
                <p><strong>Question:</strong> How does drug dosage affect recovery rate?</p>
                <p><strong>Independent Variable:</strong> Drug Dosage (mg)</p>
                <p><strong>Dependent Variable:</strong> Recovery Rate (%)</p>
                <p><strong>Hypothesis:</strong> Optimal dosage exists (too little/much = less effective)</p>
            </div>
            """, unsafe_allow_html=True)
            
            optimal_dose = st.slider("Optimal Dosage (mg)", 50, 200, 100)
            max_recovery = st.slider("Maximum Recovery Rate (%)", 70, 95, 85)
            side_effect_threshold = st.slider("Side Effect Threshold", 120, 300, 200)
            
        else:  # Plant Growth
            st.markdown("""
            <div class="example-box">
                <h4>🌱 Agricultural Scenario</h4>
                <p><strong>Question:</strong> How does fertilizer affect plant growth?</p>
                <p><strong>Independent Variable:</strong> Fertilizer Amount (kg/acre)</p>
                <p><strong>Dependent Variable:</strong> Plant Height (cm)</p>
                <p><strong>Hypothesis:</strong> More fertilizer → Taller plants (up to a limit)</p>
            </div>
            """, unsafe_allow_html=True)
            
            fertilizer_max = st.slider("Max Fertilizer (kg/acre)", 20, 100, 50)
            growth_efficiency = st.slider("Growth Efficiency", 0.5, 2.0, 1.0)
            environmental_stress = st.slider("Environmental Stress", 0.0, 0.5, 0.2)
    
    with col2:
        # Generate data based on selected example
        np.random.seed(42)
        n = sample_size if "Temperature" in example else 200
        
        if "Temperature" in example:
            x = np.random.uniform(temp_min, temp_max, n)
            y = (x - 40) * 3 + np.random.normal(0, add_noise, n)
            y = np.clip(y, 0, None)
            x_label, y_label = "Temperature (°F)", "Ice Cream Sales ($)"
            
        elif "Advertising" in example:
            x = np.random.exponential(ad_budget/3, n)
            # Diminishing returns effect
            y = (x * roi_rate * (1 - market_saturation * x / ad_budget) + 
                 np.random.normal(0, x * 0.1, n))
            y = np.clip(y, 0, None)
            x_label, y_label = "Ad Spend ($)", "Sales Revenue ($)"
            
        elif "Study Hours" in example:
            x = np.random.exponential(max_hours/3, n)
            x = np.clip(x, 0, max_hours)
            # Logarithmic relationship (diminishing returns)
            y = base_ability + 30 * np.log(x + 1) * diminishing_returns + np.random.normal(0, 5, n)
            y = np.clip(y, 0, 100)
            x_label, y_label = "Study Hours", "Test Score (%)"
            
        elif "Drug Dosage" in example:
            x = np.random.uniform(10, 300, n)
            # Inverted U-shape (optimal dosage)
            y = max_recovery * np.exp(-((x - optimal_dose)**2) / (2 * 50**2))
            # Add side effects for high doses
            side_effects = np.where(x > side_effect_threshold, 
                                  (x - side_effect_threshold) * 0.2, 0)
            y = y - side_effects + np.random.normal(0, 5, n)
            y = np.clip(y, 0, 100)
            x_label, y_label = "Drug Dosage (mg)", "Recovery Rate (%)"
            
        else:  # Plant Growth
            x = np.random.uniform(0, fertilizer_max, n)
            # Growth with environmental limits
            y = (50 + x * growth_efficiency * 2 * 
                 np.exp(-environmental_stress * x) + 
                 np.random.normal(0, 10, n))
            y = np.clip(y, 30, None)
            x_label, y_label = "Fertilizer Amount (kg/acre)", "Plant Height (cm)"
        
        # Create interactive plot
        fig = px.scatter(x=x, y=y, title=f"{example}: Relationship Visualization",
                        labels={'x': x_label, 'y': y_label},
                        trendline='ols')
        
        # Customize plot
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        fig.update_layout(height=500, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display statistics
        correlation = np.corrcoef(x, y)[0, 1]
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Correlation", f"{correlation:.3f}")
        with col_b:
            st.metric("Data Points", len(x))
        with col_c:
            relationship = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"
            st.metric("Relationship", relationship)
    
    # Interactive experiment section
    st.markdown("### 🧪 Interactive Experiment")
    st.markdown("**Try this:** Adjust the controls above and observe how the relationship changes!")
    
    experiment_col1, experiment_col2 = st.columns(2)
    with experiment_col1:
        st.markdown("""
        **🔍 Things to notice:**
        - How does the correlation coefficient change?
        - What happens to the trend line?
        - Can you make the relationship stronger or weaker?
        """)
    
    with experiment_col2:
        st.markdown("""
        **🎯 Learning Goals:**
        - Identify which variable is independent vs dependent
        - Understand how changing X affects Y
        - Recognize different types of relationships
        """)

elif page == "🔄 Interaction Variables":
    st.markdown('<h2 class="section-header">🔄 Interaction Variables</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Interaction Variables:</strong> When the effect of one variable depends on the value of another variable</p>
        <p><strong>Mathematical Form:</strong> Y = β₀ + β₁X₁ + β₂X₂ + β₃(X₁ × X₂)</p>
        <p><strong>Key Insight:</strong> 1 + 1 ≠ 2 (the combined effect is different from the sum of individual effects)</p>
    </div>
    """, unsafe_allow_html=True)
    
    scenario = st.selectbox("Choose an interaction scenario:", [
        "🏪 Price × Quality → Sales",
        "☀️ Temperature × Humidity → Comfort",
        "💊 Drug A × Drug B → Effectiveness",
        "📚 Study Method × Difficulty → Performance",
        "🎯 Experience × Training → Productivity"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if "Price" in scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🏪 Retail Business Scenario</h4>
                <p><strong>Interaction:</strong> Price × Quality</p>
                <p><strong>Insight:</strong> High quality can justify high prices, but low quality + high price = disaster!</p>
            </div>
            """, unsafe_allow_html=True)
            
            price_effect = st.slider("Price Effect Strength", -2.0, 0.5, -1.0)
            quality_effect = st.slider("Quality Effect Strength", 0.0, 3.0, 1.5)
            interaction_strength = st.slider("Interaction Strength", 0.0, 2.0, 1.2)
            
        elif "Temperature" in scenario:
            st.markdown("""
            <div class="example-box">
                <h4>☀️ Climate Comfort Scenario</h4>
                <p><strong>Interaction:</strong> Temperature × Humidity</p>
                <p><strong>Insight:</strong> High temp + low humidity = comfortable, but high temp + high humidity = unbearable!</p>
            </div>
            """, unsafe_allow_html=True)
            
            temp_comfort_range = st.slider("Optimal Temperature", 65, 85, 75)
            humidity_tolerance = st.slider("Humidity Tolerance", 0.1, 1.0, 0.5)
            interaction_severity = st.slider("Interaction Severity", 0.5, 3.0, 2.0)
            
        elif "Drug" in scenario:
            st.markdown("""
            <div class="example-box">
                <h4>💊 Pharmaceutical Scenario</h4>
                <p><strong>Interaction:</strong> Drug A × Drug B</p>
                <p><strong>Insight:</strong> Drugs can have synergistic (positive) or antagonistic (negative) interactions</p>
            </div>
            """, unsafe_allow_html=True)
            
            drug_a_effect = st.slider("Drug A Solo Effect", 0.0, 50.0, 20.0)
            drug_b_effect = st.slider("Drug B Solo Effect", 0.0, 50.0, 25.0)
            synergy_type = st.selectbox("Interaction Type", ["Synergistic (+)", "Antagonistic (-)", "No Interaction"])
            
        elif "Study" in scenario:
            st.markdown("""
            <div class="example-box">
                <h4>📚 Education Scenario</h4>
                <p><strong>Interaction:</strong> Study Method × Material Difficulty</p>
                <p><strong>Insight:</strong> Advanced methods help more with difficult material than easy material</p>
            </div>
            """, unsafe_allow_html=True)
            
            method_effectiveness = st.slider("Advanced Method Bonus", 0.0, 30.0, 15.0)
            difficulty_penalty = st.slider("Difficulty Penalty", 0.0, 40.0, 20.0)
            method_difficulty_synergy = st.slider("Method-Difficulty Synergy", 0.0, 2.0, 1.0)
        
        else:  # Experience × Training
            st.markdown("""
            <div class="example-box">
                <h4>🎯 Workplace Scenario</h4>
                <p><strong>Interaction:</strong> Experience × Training</p>
                <p><strong>Insight:</strong> Training is more effective for experienced workers who can apply it better</p>
            </div>
            """, unsafe_allow_html=True)
            
            base_productivity = st.slider("Base Productivity", 50, 100, 70)
            experience_boost = st.slider("Experience Boost", 0.0, 2.0, 1.0)
            training_multiplier = st.slider("Training Multiplier", 1.0, 3.0, 1.5)
    
    with col2:
        # Generate interaction data
        np.random.seed(42)
        n = 300
        
        if "Price" in scenario:
            price = np.random.uniform(10, 100, n)
            quality = np.random.uniform(1, 10, n)
            
            # Base effects + interaction
            sales = (50 + 
                    price * price_effect + 
                    quality * quality_effect + 
                    (price * quality / 10) * interaction_strength +
                    np.random.normal(0, 10, n))
            sales = np.clip(sales, 0, None)
            
            # Create 3D plot
            fig = px.scatter_3d(x=price, y=quality, z=sales,
                               labels={'x': 'Price ($)', 'y': 'Quality (1-10)', 'z': 'Sales'},
                               title="Price × Quality Interaction Effect on Sales",
                               color=sales, color_continuous_scale='Viridis')
            
        elif "Temperature" in scenario:
            temp = np.random.uniform(60, 100, n)
            humidity = np.random.uniform(30, 90, n)
            
            # Comfort decreases with deviation from optimal temp and high humidity interaction
            comfort = (100 - 
                      abs(temp - temp_comfort_range) * 2 - 
                      (humidity - 50) * humidity_tolerance -
                      interaction_severity * (temp - 70) * (humidity - 50) / 100 +
                      np.random.normal(0, 5, n))
            comfort = np.clip(comfort, 0, 100)
            
            fig = px.scatter_3d(x=temp, y=humidity, z=comfort,
                               labels={'x': 'Temperature (°F)', 'y': 'Humidity (%)', 'z': 'Comfort Level'},
                               title="Temperature × Humidity Interaction on Comfort",
                               color=comfort, color_continuous_scale='RdYlBu_r')
            
        elif "Drug" in scenario:
            drug_a = np.random.uniform(0, 10, n)
            drug_b = np.random.uniform(0, 10, n)
            
            # Individual effects
            effectiveness = drug_a_effect * (drug_a / 10) + drug_b_effect * (drug_b / 10)
            
            # Add interaction
            if synergy_type == "Synergistic (+)":
                interaction = (drug_a / 10) * (drug_b / 10) * 30
            elif synergy_type == "Antagonistic (-)":
                interaction = -(drug_a / 10) * (drug_b / 10) * 20
            else:
                interaction = 0
            
            effectiveness += interaction + np.random.normal(0, 3, n)
            effectiveness = np.clip(effectiveness, 0, 100)
            
            fig = px.scatter_3d(x=drug_a, y=drug_b, z=effectiveness,
                               labels={'x': 'Drug A Dose', 'y': 'Drug B Dose', 'z': 'Effectiveness (%)'},
                               title=f"Drug Interaction: {synergy_type}",
                               color=effectiveness, color_continuous_scale='Plasma')
            
        elif "Study" in scenario:
            difficulty = np.random.uniform(1, 10, n)
            method = np.random.choice([0, 1], n)  # 0=basic, 1=advanced
            
            # Base performance decreases with difficulty
            performance = (80 - difficulty * difficulty_penalty/10 +
                          method * method_effectiveness +
                          method * difficulty * method_difficulty_synergy +
                          np.random.normal(0, 5, n))
            performance = np.clip(performance, 0, 100)
            
            df = pd.DataFrame({'Difficulty': difficulty, 'Method': method, 'Performance': performance})
            df['Method_Label'] = df['Method'].map({0: 'Basic', 1: 'Advanced'})
            
            fig = px.scatter(df, x='Difficulty', y='Performance', color='Method_Label',
                           title="Study Method × Difficulty Interaction",
                           labels={'Difficulty': 'Material Difficulty (1-10)', 
                                  'Performance': 'Test Performance (%)'},
                           trendline='ols')
            
        else:  # Experience × Training
            experience = np.random.uniform(0, 20, n)
            training = np.random.choice([0, 1], n)  # 0=no training, 1=training
            
            productivity = (base_productivity + 
                           experience * experience_boost +
                           training * 10 +
                           training * experience * (training_multiplier - 1) +
                           np.random.normal(0, 5, n))
            
            df = pd.DataFrame({'Experience': experience, 'Training': training, 'Productivity': productivity})
            df['Training_Label'] = df['Training'].map({0: 'No Training', 1: 'With Training'})
            
            fig = px.scatter(df, x='Experience', y='Productivity', color='Training_Label',
                           title="Experience × Training Interaction",
                           labels={'Experience': 'Years of Experience', 
                                  'Productivity': 'Productivity Score'},
                           trendline='ols')
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show interaction insights
        st.markdown("### 🔍 Interaction Insights")
        if "Price" in scenario:
            st.markdown(f"""
            - **High Quality + High Price:** Can work well (luxury market)
            - **Low Quality + Low Price:** Budget market strategy
            - **High Price + Low Quality:** Usually fails
            - **Current Interaction Strength:** {interaction_strength:.1f}
            """)
        elif "Temperature" in scenario:
            st.markdown(f"""
            - **Optimal Temperature:** {temp_comfort_range}°F
            - **Humidity becomes more uncomfortable at higher temperatures**
            - **The interaction effect is {interaction_severity:.1f}x stronger than individual effects**
            """)

elif page == "👻 Latent Variables":
    st.markdown('<h2 class="section-header">👻 Latent Variables</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Latent Variables:</strong> Hidden variables that you cannot directly observe but that influence what you can observe</p>
        <p><strong>Examples:</strong> Intelligence, Satisfaction, Motivation, Brand Loyalty</p>
        <p><strong>Key Challenge:</strong> We measure the effects, not the cause itself</p>
    </div>
    """, unsafe_allow_html=True)
    
    latent_scenario = st.selectbox("Choose a latent variable scenario:", [
        "😊 Customer Satisfaction → Survey Responses",
        "🧠 Intelligence → Test Scores", 
        "💪 Athletic Ability → Performance Metrics",
        "🏢 Company Culture → Employee Behaviors",
        "❤️ Brand Loyalty → Purchase Patterns"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if "Satisfaction" in latent_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>😊 Customer Satisfaction Research</h4>
                <p><strong>Latent Variable:</strong> True Customer Satisfaction (unobservable)</p>
                <p><strong>Observable Indicators:</strong> Survey responses, ratings, reviews</p>
                <p><strong>Challenge:</strong> Each indicator captures satisfaction differently</p>
            </div>
            """, unsafe_allow_html=True)
            
            true_satisfaction = st.slider("Average True Satisfaction Level", 1.0, 10.0, 7.0)
            measurement_noise = st.slider("Measurement Noise Level", 0.1, 3.0, 1.0)
            response_bias = st.slider("Response Bias (tendency to agree)", -2.0, 2.0, 0.5)
            
        elif "Intelligence" in latent_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🧠 Intelligence Assessment</h4>
                <p><strong>Latent Variable:</strong> General Intelligence (g-factor)</p>
                <p><strong>Observable Indicators:</strong> Math scores, verbal scores, logic tests</p>
                <p><strong>Challenge:</strong> No single test captures all intelligence</p>
            </div>
            """, unsafe_allow_html=True)
            
            avg_intelligence = st.slider("Average Intelligence Level", 80, 120, 100)
            test_difficulty = st.slider("Test Difficulty Variance", 0.5, 2.0, 1.0)
            domain_specificity = st.slider("Domain-Specific Skills", 0.0, 1.0, 0.3)
            
        elif "Athletic" in latent_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>💪 Athletic Performance Analysis</h4>
                <p><strong>Latent Variable:</strong> Overall Athletic Ability</p>
                <p><strong>Observable Indicators:</strong> Speed, strength, endurance, coordination</p>
                <p><strong>Challenge:</strong> Athletes excel differently across metrics</p>
            </div>
            """, unsafe_allow_html=True)
            
            athletic_base = st.slider("Average Athletic Ability", 40, 90, 65)
            specialization = st.slider("Specialization Factor", 0.0, 1.0, 0.4)
            training_effect = st.slider("Training Variation", 0.5, 2.0, 1.2)
            
        elif "Culture" in latent_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🏢 Company Culture Assessment</h4>
                <p><strong>Latent Variable:</strong> Company Culture Quality</p>
                <p><strong>Observable Indicators:</strong> Turnover, productivity, satisfaction surveys</p>
                <p><strong>Challenge:</strong> Culture affects everything but is hard to measure directly</p>
            </div>
            """, unsafe_allow_html=True)
            
            culture_strength = st.slider("Culture Strength", 1, 10, 7)
            external_factors = st.slider("External Market Factors", 0.0, 1.0, 0.3)
            leadership_influence = st.slider("Leadership Influence", 0.5, 2.0, 1.2)
            
        else:  # Brand Loyalty
            st.markdown("""
            <div class="example-box">
                <h4>❤️ Brand Loyalty Analysis</h4>
                <p><strong>Latent Variable:</strong> True Brand Loyalty</p>
                <p><strong>Observable Indicators:</strong> Repeat purchases, recommendations, price sensitivity</p>
                <p><strong>Challenge:</strong> Behavior vs. true feelings can differ</p>
            </div>
            """, unsafe_allow_html=True)
            
            loyalty_level = st.slider("Average Brand Loyalty", 1, 10, 6)
            price_sensitivity = st.slider("Price Sensitivity Impact", 0.0, 2.0, 1.0)
            competitor_influence = st.slider("Competitor Influence", 0.0, 1.0, 0.4)
    
    with col2:
        # Generate latent variable data
        np.random.seed(42)
        n = 200
        
        if "Satisfaction" in latent_scenario:
            # True latent satisfaction (unobserved)
            true_sat = np.random.normal(true_satisfaction, 1.5, n)
            true_sat = np.clip(true_sat, 1, 10)
            
            # Observable indicators influenced by latent variable
            survey_q1 = true_sat + np.random.normal(response_bias, measurement_noise, n)  # Overall rating
            survey_q2 = true_sat * 0.9 + np.random.normal(response_bias * 0.5, measurement_noise, n)  # Recommendation
            survey_q3 = true_sat * 1.1 + np.random.normal(response_bias * 0.8, measurement_noise, n)  # Repeat purchase
            
            # Clip to valid ranges
            survey_q1 = np.clip(survey_q1, 1, 10)
            survey_q2 = np.clip(survey_q2, 1, 10)
            survey_q3 = np.clip(survey_q3, 1, 10)
            
            latent_df = pd.DataFrame({
                'True_Satisfaction': true_sat,
                'Overall_Rating': survey_q1,
                'Recommendation_Score': survey_q2,
                'Repurchase_Intent': survey_q3,
                'Customer_ID': range(1, n+1)
            })
            
            # Create correlation matrix of observable variables
            obs_vars = ['Overall_Rating', 'Recommendation_Score', 'Repurchase_Intent']
            corr_matrix = latent_df[obs_vars].corr()
            
            fig1 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Correlation Between Observable Indicators<br>(All influenced by latent satisfaction)",
                           color_continuous_scale='RdBu')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Show scatter plot matrix
            fig2 = px.scatter_matrix(latent_df, dimensions=obs_vars,
                                   title="Relationships Between Observable Measures")
            fig2.update_layout(height=600)
            st.plotly_chart(fig2, use_container_width=True)
            
        elif "Intelligence" in latent_scenario:
            # True intelligence (latent)
            true_iq = np.random.normal(avg_intelligence, 15, n)
            
            # Different test types influenced by intelligence + domain-specific skills
            math_score = (true_iq + np.random.normal(0, 10 * test_difficulty, n) + 
                         np.random.normal(0, 20 * domain_specificity, n))
            verbal_score = (true_iq + np.random.normal(0, 10 * test_difficulty, n) + 
                           np.random.normal(0, 20 * domain_specificity, n))
            logic_score = (true_iq + np.random.normal(0, 10 * test_difficulty, n) + 
                          np.random.normal(0, 15 * domain_specificity, n))
            
            # Normalize scores
            math_score = np.clip(math_score, 60, 140)
            verbal_score = np.clip(verbal_score, 60, 140)
            logic_score = np.clip(logic_score, 60, 140)
            
            intel_df = pd.DataFrame({
                'True_Intelligence': true_iq,
                'Math_Score': math_score,
                'Verbal_Score': verbal_score,
                'Logic_Score': logic_score
            })
            
            # 3D plot showing relationships
            fig = px.scatter_3d(intel_df, x='Math_Score', y='Verbal_Score', z='Logic_Score',
                               color='True_Intelligence',
                               title="Test Scores Influenced by Latent Intelligence",
                               color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
        elif "Athletic" in latent_scenario:
            # True athletic ability (latent)
            true_ability = np.random.normal(athletic_base, 10, n)
            
            # Different performance metrics
            speed = (true_ability + np.random.normal(0, 5 * training_effect, n) + 
                    np.random.normal(0, 15 * specialization, n))
            strength = (true_ability + np.random.normal(0, 5 * training_effect, n) + 
                       np.random.normal(0, 15 * specialization, n))
            endurance = (true_ability + np.random.normal(0, 5 * training_effect, n) + 
                        np.random.normal(0, 15 * specialization, n))
            coordination = (true_ability + np.random.normal(0, 5 * training_effect, n) + 
                           np.random.normal(0, 15 * specialization, n))
            
            athletic_df = pd.DataFrame({
                'True_Ability': true_ability,
                'Speed_Score': np.clip(speed, 20, 100),
                'Strength_Score': np.clip(strength, 20, 100),
                'Endurance_Score': np.clip(endurance, 20, 100),
                'Coordination_Score': np.clip(coordination, 20, 100)
            })
            
            # Radar chart for athletic profiles
            fig = go.Figure()
            
            # Sample a few athletes for radar chart
            sample_athletes = athletic_df.sample(5)
            
            for idx, athlete in sample_athletes.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[athlete['Speed_Score'], athlete['Strength_Score'], 
                       athlete['Endurance_Score'], athlete['Coordination_Score']],
                    theta=['Speed', 'Strength', 'Endurance', 'Coordination'],
                    fill='toself',
                    name=f'Athlete {idx+1} (Ability: {athlete["True_Ability"]:.0f})'
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Athletic Performance Profiles<br>(Different expressions of latent athletic ability)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show key insights
        st.markdown("### 🔍 Key Insights About Latent Variables")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            **🎯 Why Latent Variables Matter:**
            - They explain why observable measures correlate
            - Help us understand the "why" behind data
            - Essential for surveys, psychology, marketing
            """)
        
        with insights_col2:
            st.markdown("""
            **⚠️ Challenges:**
            - Cannot be measured directly
            - Require multiple indicators
            - Subject to measurement error
            """)

elif page == "🌪️ Confounding Variables":
    st.markdown('<h2 class="section-header">🌪️ Confounding Variables</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Confounding Variables:</strong> Hidden variables that affect BOTH your independent and dependent variables, creating false relationships</p>
        <p><strong>The Problem:</strong> You think X causes Y, but actually Z causes both X and Y</p>
        <p><strong>Famous Example:</strong> Ice cream sales and drowning deaths (both caused by hot weather!)</p>
    </div>
    """, unsafe_allow_html=True)
    
    confound_scenario = st.selectbox("Choose a confounding scenario:", [
        "🍦 Ice Cream Sales vs Drownings (Temperature)",
        "📚 Shoe Size vs Reading Ability (Age)",
        "☕ Coffee Shops vs Crime Rate (Population Density)",
        "🏠 Home Size vs Income (Location)",
        "📱 Screen Time vs Poor Grades (Socioeconomic Status)"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ Confounding Alert!</h4>
            <p>This scenario shows how confounding variables can create misleading correlations.</p>
            <p><strong>Remember:</strong> Correlation ≠ Causation</p>
        </div>
        """, unsafe_allow_html=True)
        
        show_confounder = st.checkbox("Reveal the confounding variable", value=False)
        confounder_strength = st.slider("Confounding Effect Strength", 0.0, 3.0, 2.0)
        sample_size = st.slider("Sample Size", 100, 1000, 300)
        
        if "Ice Cream" in confound_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🍦 Classic Confounding Example</h4>
                <p><strong>Apparent Relationship:</strong> Ice cream sales → Drowning deaths</p>
                <p><strong>Confounding Variable:</strong> Temperature</p>
                <p><strong>Truth:</strong> Hot weather increases both ice cream sales AND swimming activity</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Shoe Size" in confound_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>📚 Development Confounding</h4>
                <p><strong>Apparent Relationship:</strong> Shoe size → Reading ability</p>
                <p><strong>Confounding Variable:</strong> Age</p>
                <p><strong>Truth:</strong> Older children have bigger feet AND better reading skills</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Coffee" in confound_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>☕ Urban Planning Confounding</h4>
                <p><strong>Apparent Relationship:</strong> Coffee shops → Crime rate</p>
                <p><strong>Confounding Variable:</strong> Population density</p>
                <p><strong>Truth:</strong> Dense areas have more coffee shops AND more crime opportunities</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Home Size" in confound_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🏠 Real Estate Confounding</h4>
                <p><strong>Apparent Relationship:</strong> Home size → Income</p>
                <p><strong>Confounding Variable:</strong> Location (urban vs suburban)</p>
                <p><strong>Truth:</strong> Location affects both home size AND income levels</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Screen Time
            st.markdown("""
            <div class="example-box">
                <h4>📱 Education Confounding</h4>
                <p><strong>Apparent Relationship:</strong> Screen time → Poor grades</p>
                <p><strong>Confounding Variable:</strong> Socioeconomic status</p>
                <p><strong>Truth:</strong> SES affects both device access AND educational resources</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Generate confounded data
        np.random.seed(42)
        n = sample_size
        
        if "Ice Cream" in confound_scenario:
            # Confounding variable: Temperature
            temperature = np.random.uniform(60, 100, n)
            
            if show_confounder:
                # Both variables affected by temperature
                ice_cream_sales = ((temperature - 60) * confounder_strength * 5 + 
                                 np.random.normal(100, 20, n))
                drowning_deaths = ((temperature - 60) * confounder_strength * 0.1 + 
                                 np.random.normal(2, 1, n))
            else:
                # Random relationship (no confounding shown)
                ice_cream_sales = np.random.normal(200, 50, n)
                drowning_deaths = np.random.normal(4, 2, n)
            
            ice_cream_sales = np.clip(ice_cream_sales, 0, None)
            drowning_deaths = np.clip(drowning_deaths, 0, None)
            
            confound_df = pd.DataFrame({
                'Temperature': temperature,
                'Ice_Cream_Sales': ice_cream_sales,
                'Drowning_Deaths': drowning_deaths
            })
            
            if show_confounder:
                # Show the confounding variable
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Ice Cream vs Drownings', 'Temperature vs Ice Cream', 
                                  'Temperature vs Drownings', 'Confounding Explanation'),
                    specs=[[{"colspan": 2}, None],
                           [{}, {}]]
                )
                
                # Main misleading relationship
                fig.add_trace(
                    go.Scatter(x=confound_df['Ice_Cream_Sales'], y=confound_df['Drowning_Deaths'],
                             mode='markers', name='Misleading Correlation', 
                             marker=dict(color='red', size=8)),
                    row=1, col=1
                )
                
                # Temperature vs Ice Cream
                fig.add_trace(
                    go.Scatter(x=confound_df['Temperature'], y=confound_df['Ice_Cream_Sales'],
                             mode='markers', name='Temp → Ice Cream', 
                             marker=dict(color='orange', size=6)),
                    row=2, col=1
                )
                
                # Temperature vs Drownings
                fig.add_trace(
                    go.Scatter(x=confound_df['Temperature'], y=confound_df['Drowning_Deaths'],
                             mode='markers', name='Temp → Drownings', 
                             marker=dict(color='blue', size=6)),
                    row=2, col=2
                )
                
                fig.update_layout(title="Confounding Variable Revealed: Temperature")
                
            else:
                # Show only the misleading correlation
                fig = px.scatter(confound_df, x='Ice_Cream_Sales', y='Drowning_Deaths',
                               title="Apparent Relationship: Ice Cream Sales vs Drowning Deaths",
                               trendline='ols')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlations
            corr_misleading = confound_df['Ice_Cream_Sales'].corr(confound_df['Drowning_Deaths'])
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Ice Cream vs Drowning Correlation", f"{corr_misleading:.3f}")
            
            if show_confounder:
                with col_b:
                    corr_temp_ice = confound_df['Temperature'].corr(confound_df['Ice_Cream_Sales'])
                    st.metric("Temperature vs Ice Cream", f"{corr_temp_ice:.3f}")
                with col_c:
                    corr_temp_drown = confound_df['Temperature'].corr(confound_df['Drowning_Deaths'])
                    st.metric("Temperature vs Drownings", f"{corr_temp_drown:.3f}")

        elif "Shoe Size" in confound_scenario:
            # Confounding variable: Age (5-12 years old)
            age = np.random.uniform(5, 12, n)
            
            if show_confounder:
                # Both affected by age
                shoe_size = (age * 1.5 + np.random.normal(0, 1, n))
                reading_score = (age * 8 + np.random.normal(0, 5, n))
            else:
                shoe_size = np.random.normal(6, 2, n)
                reading_score = np.random.normal(50, 15, n)
            
            shoe_size = np.clip(shoe_size, 3, 12)
            reading_score = np.clip(reading_score, 0, 100)
            
            confound_df = pd.DataFrame({
                'Age': age,
                'Shoe_Size': shoe_size,
                'Reading_Score': reading_score
            })
            
            if show_confounder:
                fig = px.scatter_3d(confound_df, x='Shoe_Size', y='Reading_Score', z='Age',
                                   color='Age', title="Age Confounds Shoe Size and Reading Ability",
                                   labels={'Shoe_Size': 'Shoe Size', 'Reading_Score': 'Reading Score'})
            else:
                fig = px.scatter(confound_df, x='Shoe_Size', y='Reading_Score',
                               title="Apparent Relationship: Shoe Size vs Reading Ability",
                               trendline='ols')
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Interactive exercise
    st.markdown("### 🧪 Confounding Detective Exercise")
    
    exercise_col1, exercise_col2 = st.columns(2)
    
    with exercise_col1:
        st.markdown("""
        **🔍 Your Mission:**
        1. Look at the apparent relationship when confounding is hidden
        2. Check the "Reveal confounding variable" box
        3. Observe how the relationship changes
        4. Adjust the confounding strength
        """)
    
    with exercise_col2:
        st.markdown("""
        **🎯 What to Learn:**
        - How strong correlations can be misleading
        - Why domain knowledge is crucial
        - The importance of controlled experiments
        - How to identify potential confounders
        """)
    
    if show_confounder:
        st.success("🎉 Great! You've revealed the confounding variable. Notice how the apparent relationship between the main variables is actually explained by the third variable affecting both.")
    else:
        st.info("🤔 The correlation looks strong, but is it real? Try revealing the confounding variable to see what's really happening.")

elif page == "🎛️ Control Variables":
    st.markdown('<h2 class="section-header">🎛️ Control Variables</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Control Variables:</strong> Variables you hold constant or account for to isolate the true effect of your independent variable</p>
        <p><strong>Purpose:</strong> Remove confounding effects to see the real relationship</p>
        <p><strong>Methods:</strong> Physical control, statistical control, matching, randomization</p>
    </div>
    """, unsafe_allow_html=True)
    
    control_scenario = st.selectbox("Choose a control scenario:", [
        "💊 Drug Testing (Control for Age)",
        "📚 Teaching Methods (Control for Prior Knowledge)",
        "🎯 Marketing Campaigns (Control for Seasonality)",
        "🏭 Productivity Study (Control for Experience)",
        "🌱 Fertilizer Experiment (Control for Soil Quality)"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Experimental Controls")
        
        control_active = st.checkbox("Apply Control Variable", value=True)
        control_strength = st.slider("Control Effectiveness", 0.0, 1.0, 0.8)
        
        if "Drug" in control_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>💊 Clinical Trial Design</h4>
                <p><strong>Goal:</strong> Test if new drug works</p>
                <p><strong>Problem:</strong> Age affects recovery rate</p>
                <p><strong>Solution:</strong> Control for age in analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            age_effect = st.slider("Age Effect on Recovery", 0.0, 2.0, 1.0)
            drug_true_effect = st.slider("True Drug Effect", 0, 30, 20)
            
        elif "Teaching" in control_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>📚 Education Research</h4>
                <p><strong>Goal:</strong> Compare teaching methods</p>
                <p><strong>Problem:</strong> Students have different prior knowledge</p>
                <p><strong>Solution:</strong> Control for baseline scores</p>
            </div>
            """, unsafe_allow_html=True)
            
            prior_knowledge_effect = st.slider("Prior Knowledge Impact", 0.0, 1.0, 0.7)
            method_effectiveness = st.slider("New Method Advantage", 0, 20, 12)
            
        elif "Marketing" in control_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🎯 Marketing Analysis</h4>
                <p><strong>Goal:</strong> Measure campaign effectiveness</p>
                <p><strong>Problem:</strong> Sales vary by season</p>
                <p><strong>Solution:</strong> Control for seasonal trends</p>
            </div>
            """, unsafe_allow_html=True)
            
            seasonal_amplitude = st.slider("Seasonal Effect Size", 0, 100, 50)
            campaign_lift = st.slider("Campaign Effect", 0, 80, 30)
            
        elif "Productivity" in control_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🏭 Workplace Study</h4>
                <p><strong>Goal:</strong> Test new training program</p>
                <p><strong>Problem:</strong> Experience affects productivity</p>
                <p><strong>Solution:</strong> Match participants by experience</p>
            </div>
            """, unsafe_allow_html=True)
            
            experience_impact = st.slider("Experience Effect", 0.0, 3.0, 1.5)
            training_benefit = st.slider("Training Improvement", 0, 25, 15)
            
        else:  # Fertilizer
            st.markdown("""
            <div class="example-box">
                <h4>🌱 Agricultural Research</h4>
                <p><strong>Goal:</strong> Test fertilizer effectiveness</p>
                <p><strong>Problem:</strong> Soil quality varies</p>
                <p><strong>Solution:</strong> Control plots with similar soil</p>
            </div>
            """, unsafe_allow_html=True)
            
            soil_quality_range = st.slider("Soil Quality Variation", 0.0, 2.0, 1.0)
            fertilizer_effect = st.slider("Fertilizer Benefit", 0, 40, 25)
    
    with col2:
        # Generate controlled experiment data
        np.random.seed(42)
        n = 200
        
        if "Drug" in control_scenario:
            # Create age groups
            age = np.random.uniform(20, 80, n)
            treatment = np.random.choice([0, 1], n)  # 0=placebo, 1=drug
            
            # Recovery rate affected by age and treatment
            base_recovery = 60
            age_effect_actual = (80 - age) * age_effect * 0.5  # Younger people recover better
            treatment_effect = treatment * drug_true_effect
            
            if control_active:
                # Control for age - remove age effect from analysis
                recovery_rate = (base_recovery + treatment_effect + 
                               age_effect_actual * (1 - control_strength) +
                               np.random.normal(0, 10, n))
            else:
                # No control - age effect confounds results
                recovery_rate = (base_recovery + treatment_effect + age_effect_actual +
                               np.random.normal(0, 10, n))
            
            recovery_rate = np.clip(recovery_rate, 0, 100)
            
            experiment_df = pd.DataFrame({
                'Age': age,
                'Treatment': treatment,
                'Recovery_Rate': recovery_rate,
                'Treatment_Label': ['Placebo', 'Drug'][treatment] if isinstance(treatment, int) else ['Placebo' if t == 0 else 'Drug' for t in treatment]
            })
            
            # Create comparison plots
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=('Recovery by Age & Treatment', 'Treatment Effect'))
            
            # Scatter plot colored by treatment
            for treat_val, treat_name, color in [(0, 'Placebo', 'red'), (1, 'Drug', 'blue')]:
                mask = experiment_df['Treatment'] == treat_val
                fig.add_trace(
                    go.Scatter(x=experiment_df[mask]['Age'], 
                             y=experiment_df[mask]['Recovery_Rate'],
                             mode='markers', name=treat_name, 
                             marker=dict(color=color, size=6)),
                    row=1, col=1
                )
            
            # Box plot of treatment effect
            fig.add_trace(
                go.Box(y=experiment_df[experiment_df['Treatment']==0]['Recovery_Rate'],
                      name='Placebo', marker_color='red'),
                row=1, col=2
            )
            fig.add_trace(
                go.Box(y=experiment_df[experiment_df['Treatment']==1]['Recovery_Rate'],
                      name='Drug', marker_color='blue'),
                row=1, col=2
            )
            
            fig.update_layout(title=f"Drug Trial Results {'(Age Controlled)' if control_active else '(Age Not Controlled)'}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate treatment effect
            placebo_mean = experiment_df[experiment_df['Treatment']==0]['Recovery_Rate'].mean()
            drug_mean = experiment_df[experiment_df['Treatment']==1]['Recovery_Rate'].mean()
            observed_effect = drug_mean - placebo_mean
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Placebo Group", f"{placebo_mean:.1f}%")
            with col_b:
                st.metric("Drug Group", f"{drug_mean:.1f}%")
            with col_c:
                st.metric("Observed Effect", f"{observed_effect:.1f}%")
                
            # Show the importance of control
            if control_active:
                st.success(f"✅ With age control: Observed effect ({observed_effect:.1f}%) is close to true effect ({drug_true_effect}%)")
            else:
                st.warning(f"⚠️ Without age control: Observed effect ({observed_effect:.1f}%) may be biased by age differences")

        elif "Teaching" in control_scenario:
            # Prior knowledge and teaching method
            prior_score = np.random.normal(70, 15, n)
            method = np.random.choice([0, 1], n)  # 0=traditional, 1=new method
            
            # Post-test score
            if control_active:
                # Control for prior knowledge
                post_score = (prior_score * prior_knowledge_effect * (1 - control_strength) +
                             method * method_effectiveness +
                             np.random.normal(0, 8, n))
            else:
                # No control
                post_score = (prior_score * prior_knowledge_effect +
                             method * method_effectiveness +
                             np.random.normal(0, 8, n))
            
            post_score = np.clip(post_score, 0, 100)
            
            teaching_df = pd.DataFrame({
                'Prior_Score': prior_score,
                'Method': method,
                'Post_Score': post_score,
                'Method_Label': ['Traditional', 'New Method'][method] if isinstance(method, int) else ['Traditional' if m == 0 else 'New Method' for m in method]
            })
            
            fig = px.scatter(teaching_df, x='Prior_Score', y='Post_Score', color='Method_Label',
                           title=f"Teaching Method Comparison {'(Prior Knowledge Controlled)' if control_active else '(No Control)'}",
                           trendline='ols')
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔗 Correlated Variables":
    st.markdown('<h2 class="section-header">🔗 Correlated Variables</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Correlated Variables:</strong> Variables that move together in a predictable pattern</p>
        <p><strong>Types:</strong> Positive correlation (both increase), Negative correlation (one increases, other decreases)</p>
        <p><strong>Key Warning:</strong> Correlation ≠ Causation (they might both be caused by something else!)</p>
    </div>
    """, unsafe_allow_html=True)
    
    correlation_scenario = st.selectbox("Choose a correlation scenario:", [
        "🌡️ Temperature & Energy Usage",
        "📈 Stock Prices & Market Index", 
        "🏠 House Size & Property Value",
        "🎓 Study Time & GPA",
        "📱 Social Media & Anxiety Levels"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Correlation Controls")
        
        correlation_strength = st.slider("Correlation Strength", -1.0, 1.0, 0.7)
        sample_size = st.slider("Sample Size", 50, 500, 200)
        noise_level = st.slider("Random Noise", 0.1, 2.0, 0.5)
        add_outliers = st.checkbox("Add Outliers", False)
        
        if "Temperature" in correlation_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🌡️ Energy Consumption Pattern</h4>
                <p><strong>Expected:</strong> Strong negative correlation</p>
                <p><strong>Why:</strong> Hot weather → More AC use → Higher energy bills</p>
                <p><strong>Cold weather:</strong> More heating → Higher energy bills</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Stock" in correlation_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>📈 Financial Market Behavior</h4>
                <p><strong>Expected:</strong> Strong positive correlation</p>
                <p><strong>Why:</strong> Individual stocks tend to follow market trends</p>
                <p><strong>Exception:</strong> Some stocks are "market contrarian"</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "House" in correlation_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🏠 Real Estate Relationships</h4>
                <p><strong>Expected:</strong> Strong positive correlation</p>
                <p><strong>Why:</strong> Bigger houses generally cost more</p>
                <p><strong>But:</strong> Location can override size effects</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Study" in correlation_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🎓 Academic Performance</h4>
                <p><strong>Expected:</strong> Moderate positive correlation</p>
                <p><strong>Why:</strong> More study usually improves grades</p>
                <p><strong>But:</strong> Quality of study matters too!</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Social Media
            st.markdown("""
            <div class="example-box">
                <h4>📱 Digital Wellbeing</h4>
                <p><strong>Expected:</strong> Positive correlation (controversial)</p>
                <p><strong>Why:</strong> Excessive use may increase anxiety</p>
                <p><strong>But:</strong> Could be reverse causation!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Generate correlated data
        np.random.seed(42)
        n = sample_size
        
        if "Temperature" in correlation_scenario:
            # Temperature and energy usage (U-shaped relationship)
            temperature = np.random.uniform(0, 100, n)
            
            # Energy usage is high when very hot or very cold
            optimal_temp = 72
            energy_base = 100
            energy_usage = (energy_base + 
                           ((temperature - optimal_temp) ** 2) * 0.05 * abs(correlation_strength) +
                           np.random.normal(0, noise_level * 20, n))
            
            if correlation_strength < 0:
                energy_usage = energy_base * 2 - energy_usage  # Invert for negative correlation
            
            x_data, y_data = temperature, energy_usage
            x_label, y_label = "Temperature (°F)", "Energy Usage (kWh)"
            
        elif "Stock" in correlation_scenario:
            # Stock price and market index
            market_index = np.random.normal(100, 20, n)
            market_changes = np.diff(np.concatenate([[100], market_index]))
            
            # Stock follows market with some independence
            stock_price_changes = (market_changes * correlation_strength + 
                                 np.random.normal(0, noise_level * 5, n))
            stock_price = np.cumsum(np.concatenate([[50], stock_price_changes]))
            
            x_data, y_data = market_index, stock_price[:n]
            x_label, y_label = "Market Index", "Stock Price ($)"
            
        elif "House" in correlation_scenario:
            # House size and value
            house_size = np.random.uniform(800, 4000, n)
            
            # Value correlates with size
            base_price = 100000
            price_per_sqft = 150
            house_value = (base_price + 
                          house_size * price_per_sqft * correlation_strength +
                          np.random.normal(0, noise_level * 50000, n))
            
            x_data, y_data = house_size, house_value
            x_label, y_label = "House Size (sq ft)", "House Value ($)"
            
        elif "Study" in correlation_scenario:
            # Study time and GPA
            study_hours = np.random.uniform(0, 40, n)
            
            # GPA with diminishing returns
            base_gpa = 2.0
            gpa = (base_gpa + 
                   np.log(study_hours + 1) * 0.5 * correlation_strength +
                   np.random.normal(0, noise_level * 0.3, n))
            gpa = np.clip(gpa, 0, 4.0)
            
            x_data, y_data = study_hours, gpa
            x_label, y_label = "Study Hours per Week", "GPA"
            
        else:  # Social Media
            # Social media usage and anxiety
            social_media_hours = np.random.exponential(3, n)
            
            # Anxiety increases with usage (if positive correlation)
            base_anxiety = 30
            anxiety_level = (base_anxiety + 
                           social_media_hours * 5 * correlation_strength +
                           np.random.normal(0, noise_level * 10, n))
            anxiety_level = np.clip(anxiety_level, 0, 100)
            
            x_data, y_data = social_media_hours, anxiety_level
            x_label, y_label = "Social Media Hours/Day", "Anxiety Level (0-100)"
        
        # Add outliers if requested
        if add_outliers:
            n_outliers = max(1, n // 20)  # 5% outliers
            outlier_indices = np.random.choice(n, n_outliers, replace=False)
            
            # Make outliers deviate significantly
            for idx in outlier_indices:
                if np.random.random() > 0.5:
                    y_data[idx] = np.random.uniform(np.min(y_data), np.max(y_data))
                else:
                    x_data[idx] = np.random.uniform(np.min(x_data), np.max(x_data))
        
        # Create DataFrame
        corr_df = pd.DataFrame({
            'X': x_data,
            'Y': y_data,
            'Outlier': False
        })
        
        if add_outliers:
            corr_df.loc[outlier_indices, 'Outlier'] = True
        
        # Create scatter plot
        fig = px.scatter(corr_df, x='X', y='Y', color='Outlier' if add_outliers else None,
                        title=f"{correlation_scenario}: Correlation Analysis",
                        labels={'X': x_label, 'Y': y_label},
                        trendline='ols')
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate actual correlation
        actual_correlation = np.corrcoef(x_data, y_data)[0, 1]
        
        # Display metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Target Correlation", f"{correlation_strength:.3f}")
        with col_b:
            st.metric("Actual Correlation", f"{actual_correlation:.3f}")
        with col_c:
            correlation_strength_text = (
                "Very Strong" if abs(actual_correlation) > 0.8 else
                "Strong" if abs(actual_correlation) > 0.6 else
                "Moderate" if abs(actual_correlation) > 0.4 else
                "Weak" if abs(actual_correlation) > 0.2 else
                "Very Weak"
            )
            st.metric("Strength", correlation_strength_text)
        
        # Correlation interpretation
        st.markdown("### 🔍 Correlation Insights")
        
        if actual_correlation > 0.7:
            st.success("🔗 **Strong Positive Correlation**: As one variable increases, the other tends to increase significantly.")
        elif actual_correlation > 0.3:
            st.info("📈 **Moderate Positive Correlation**: There's a noticeable tendency for both variables to increase together.")
        elif actual_correlation > -0.3:
            st.warning("🤷 **Weak/No Correlation**: The variables don't show a clear linear relationship.")
        elif actual_correlation > -0.7:
            st.info("📉 **Moderate Negative Correlation**: As one variable increases, the other tends to decrease.")
        else:
            st.error("🔗 **Strong Negative Correlation**: As one variable increases, the other decreases significantly.")

elif page == "💧 Leaky Variables":
    st.markdown('<h2 class="section-header">💧 Leaky Variables (Data Leakage)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Leaky Variables:</strong> Features that contain information about the target that wouldn't be available at prediction time</p>
        <p><strong>The Problem:</strong> Your model looks amazing in testing but fails completely in real-world use</p>
        <p><strong>Common Causes:</strong> Future information, target encoding, data preprocessing errors</p>
    </div>
    """, unsafe_allow_html=True)
    
    leaky_scenario = st.selectbox("Choose a data leakage scenario:", [
        "🏥 Medical Diagnosis (Test Results)",
        "💳 Credit Approval (Payment History)", 
        "🎯 Marketing Response (Purchase Date)",
        "📈 Stock Prediction (Future Prices)",
        "🚗 Insurance Claims (Claim Amount)"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚠️ Leakage Detection")
        
        leakage_amount = st.slider("Information Leakage Level", 0.0, 1.0, 0.4)
        add_legitimate_features = st.checkbox("Include Legitimate Features", True)
        show_performance_comparison = st.checkbox("Show Model Performance Impact", True)
        
        if "Medical" in leaky_scenario:
            st.markdown("""
            <div class="warning-box">
                <h4>🏥 Medical AI Disaster</h4>
                <p><strong>Goal:</strong> Predict if patient has disease</p>
                <p><strong>Leaky Feature:</strong> "Test ordered for disease X"</p>
                <p><strong>Problem:</strong> Doctors only order tests when they suspect the disease!</p>
                <p><strong>Result:</strong> Model learns to cheat, not diagnose</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Credit" in leaky_scenario:
            st.markdown("""
            <div class="warning-box">
                <h4>💳 Credit Scoring Failure</h4>
                <p><strong>Goal:</strong> Predict loan default risk</p>
                <p><strong>Leaky Feature:</strong> "Payment history next 6 months"</p>
                <p><strong>Problem:</strong> Future payment info used to predict future default!</p>
                <p><strong>Result:</strong> Perfect accuracy in testing, useless in production</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Marketing" in leaky_scenario:
            st.markdown("""
            <div class="warning-box">
                <h4>🎯 Marketing Model Mishap</h4>
                <p><strong>Goal:</strong> Predict customer response to campaign</p>
                <p><strong>Leaky Feature:</strong> "Purchase date" or "Response timestamp"</p>
                <p><strong>Problem:</strong> Using outcome timing to predict the outcome!</p>
                <p><strong>Result:</strong> Model finds patterns in data collection, not customer behavior</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Stock" in leaky_scenario:
            st.markdown("""
            <div class="warning-box">
                <h4>📈 Trading Algorithm Trap</h4>
                <p><strong>Goal:</strong> Predict tomorrow's stock price</p>
                <p><strong>Leaky Feature:</strong> "Next day's opening price" or "Future volume"</p>
                <p><strong>Problem:</strong> Using future information to predict the future!</p>
                <p><strong>Result:</strong> Backtest looks perfect, live trading loses money</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Insurance
            st.markdown("""
            <div class="warning-box">
                <h4>🚗 Insurance Fraud Detection Flaw</h4>
                <p><strong>Goal:</strong> Predict if claim is fraudulent</p>
                <p><strong>Leaky Feature:</strong> "Claim processing time" or "Investigation notes"</p>
                <p><strong>Problem:</strong> Fraudulent claims take longer to process!</p>
                <p><strong>Result:</strong> Model learns processing patterns, not fraud patterns</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Generate leaky data simulation
        np.random.seed(42)
        n = 1000
        
        if "Medical" in leaky_scenario:
            # Generate patient data
            age = np.random.normal(60, 15, n)
            symptoms_severity = np.random.uniform(1, 10, n)
            
            # True disease status (what we want to predict)
            disease_probability = 0.2 + (age - 40) * 0.005 + symptoms_severity * 0.05
            disease_probability = np.clip(disease_probability, 0, 1)
            has_disease = np.random.binomial(1, disease_probability, n)
            
            # Legitimate features
            legitimate_features = pd.DataFrame({
                'Age': age,
                'Symptoms_Severity': symptoms_severity,
                'Family_History': np.random.binomial(1, 0.3, n),
                'BMI': np.random.normal(25, 5, n)
            })
            
            # Leaky feature: Test was ordered (doctors order tests when they suspect disease)
            test_ordered_prob = has_disease * 0.8 + (1 - has_disease) * 0.1  # 80% if disease, 10% if no disease
            test_ordered = np.random.binomial(1, test_ordered_prob, n)
            
            # Add leakage
            leaky_feature = has_disease * leakage_amount + np.random.random(n) * (1 - leakage_amount)
            
            features_df = legitimate_features.copy()
            features_df['Test_Ordered'] = test_ordered
            features_df['Leaky_Feature'] = leaky_feature
            features_df['Has_Disease'] = has_disease
            
            target_col = 'Has_Disease'
            leaky_col = 'Test_Ordered'
            
        elif "Credit" in leaky_scenario:
            # Generate credit data
            income = np.random.lognormal(10, 0.5, n)
            credit_score = np.random.normal(650, 100, n)
            debt_ratio = np.random.uniform(0, 0.8, n)
            
            # True default probability
            default_prob = 0.1 + (700 - credit_score) * 0.001 + debt_ratio * 0.3
            default_prob = np.clip(default_prob, 0, 1)
            defaults = np.random.binomial(1, default_prob, n)
            
            # Legitimate features
            legitimate_features = pd.DataFrame({
                'Income': income,
                'Credit_Score': credit_score,
                'Debt_Ratio': debt_ratio,
                'Employment_Years': np.random.exponential(5, n)
            })
            
            # Leaky feature: Future payment behavior
            future_payments = (1 - defaults) * 0.9 + defaults * 0.1  # Good payers if no default
            
            # Add leakage
            leaky_feature = defaults * leakage_amount + np.random.random(n) * (1 - leakage_amount)
            
            features_df = legitimate_features.copy()
            features_df['Future_Payment_Score'] = future_payments
            features_df['Leaky_Feature'] = leaky_feature
            features_df['Default'] = defaults
            
            target_col = 'Default'
            leaky_col = 'Future_Payment_Score'
        
        # Create correlation heatmap
        if add_legitimate_features:
            correlation_cols = list(legitimate_features.columns) + [leaky_col, 'Leaky_Feature', target_col]
        else:
            correlation_cols = [leaky_col, 'Leaky_Feature', target_col]
        
        corr_matrix = features_df[correlation_cols].corr()
        
        fig1 = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                        title="Feature Correlations - Spot the Leakage!",
                        color_continuous_scale='RdBu', color_continuous_midpoint=0)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Show model performance comparison
        if show_performance_comparison:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            # Prepare data
            X_legit = legitimate_features
            X_with_leak = features_df[list(legitimate_features.columns) + [leaky_col]]
            y = features_df[target_col]
            
            # Split data
            X_legit_train, X_legit_test, y_train, y_test = train_test_split(X_legit, y, test_size=0.3, random_state=42)
            X_leak_train, X_leak_test, _, _ = train_test_split(X_with_leak, y, test_size=0.3, random_state=42)
            
            # Train models
            model_legit = RandomForestClassifier(random_state=42, n_estimators=50)
            model_leak = RandomForestClassifier(random_state=42, n_estimators=50)
            
            model_legit.fit(X_legit_train, y_train)
            model_leak.fit(X_leak_train, y_train)
            
            # Make predictions
            pred_legit = model_legit.predict(X_legit_test)
            pred_leak = model_leak.predict(X_leak_test)
            
            # Calculate metrics
            acc_legit = accuracy_score(y_test, pred_legit)
            acc_leak = accuracy_score(y_test, pred_leak)
            
            try:
                auc_legit = roc_auc_score(y_test, model_legit.predict_proba(X_legit_test)[:, 1])
                auc_leak = roc_auc_score(y_test, model_leak.predict_proba(X_leak_test)[:, 1])
            except:
                auc_legit = auc_leak = 0.5
            
            # Display comparison
            st.markdown("### 🎯 Model Performance Comparison")
            
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown("**🔬 Legitimate Model (No Leakage)**")
                st.metric("Accuracy", f"{acc_legit:.3f}")
                st.metric("AUC Score", f"{auc_legit:.3f}")
                st.success("✅ Realistic performance")
            
            with perf_col2:
                st.markdown("**💧 Leaky Model (With Leakage)**")
                st.metric("Accuracy", f"{acc_leak:.3f}")
                st.metric("AUC Score", f"{auc_leak:.3f}")
                if acc_leak > acc_legit + 0.1:
                    st.error("🚨 Suspiciously high performance!")
                else:
                    st.warning("⚠️ May still have subtle leakage")
        
        # Feature importance analysis
        if 'model_leak' in locals():
            importance_df = pd.DataFrame({
                'Feature': X_leak_train.columns,
                'Importance': model_leak.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig2 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title="Feature Importance - Leaky Features Often Dominate")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Learning exercise
        st.markdown("### 🎓 Leakage Detection Exercise")
        
        exercise_col1, exercise_col2 = st.columns(2)
        
        with exercise_col1:
            st.markdown("""
            **🔍 Red Flags to Look For:**
            - Suspiciously high model performance
            - Features with perfect correlation to target
            - Features that wouldn't be available at prediction time
            - Domain knowledge violations
            """)
        
        with exercise_col2:
            st.markdown("""
            **🛡️ Prevention Strategies:**
            - Time-based validation splits
            - Careful feature engineering review
            - Domain expert consultation
            - Gradual feature removal testing
            """)

elif page == "📈 Stationary vs Non-stationary":
    st.markdown('<h2 class="section-header">📈 Stationary vs Non-stationary Variables</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Stationary:</strong> Statistical properties (mean, variance) don't change over time</p>
        <p><strong>Non-stationary:</strong> Statistical properties change over time (trends, seasonality, changing variance)</p>
        <p><strong>Why It Matters:</strong> Many statistical models assume stationarity for valid predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    ts_scenario = st.selectbox("Choose a time series scenario:", [
        "📊 Stock Prices vs Returns",
        "🌡️ Global Temperature Trends", 
        "💰 GDP Growth Over Decades",
        "🏠 Housing Prices (Bubble & Crash)",
        "⚡ Energy Consumption Patterns"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Time Series Controls")
        
        add_trend = st.checkbox("Add Trend", True)
        trend_strength = st.slider("Trend Strength", 0.0, 5.0, 1.0)
        
        add_seasonality = st.checkbox("Add Seasonality", True)
        seasonal_amplitude = st.slider("Seasonal Effect", 0, 50, 20)
        
        add_volatility_change = st.checkbox("Changing Volatility", False)
        noise_level = st.slider("Base Noise Level", 0.1, 5.0, 1.0)
        
        time_periods = st.slider("Time Periods", 100, 1000, 365)
        
        if "Stock" in ts_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>📊 Financial Time Series</h4>
                <p><strong>Stock Prices:</strong> Non-stationary (trending, changing volatility)</p>
                <p><strong>Stock Returns:</strong> Often stationary (mean-reverting)</p>
                <p><strong>Key Insight:</strong> Price changes, not prices, are usually stationary</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Temperature" in ts_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🌡️ Climate Data Analysis</h4>
                <p><strong>Daily Temps:</strong> Seasonal (stationary around trend)</p>
                <p><strong>Global Average:</strong> Non-stationary (climate change trend)</p>
                <p><strong>Key Insight:</strong> Detrending can make data stationary</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "GDP" in ts_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>💰 Economic Indicators</h4>
                <p><strong>GDP Level:</strong> Non-stationary (grows over time)</p>
                <p><strong>GDP Growth Rate:</strong> More stationary (fluctuates around mean)</p>
                <p><strong>Key Insight:</strong> Growth rates vs levels behave differently</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Housing" in ts_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🏠 Real Estate Cycles</h4>
                <p><strong>Normal Times:</strong> Trending upward (non-stationary)</p>
                <p><strong>Bubble Periods:</strong> Extreme non-stationarity</p>
                <p><strong>Key Insight:</strong> Structural breaks change stationarity</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Energy
            st.markdown("""
            <div class="example-box">
                <h4>⚡ Energy Usage Patterns</h4>
                <p><strong>Daily Usage:</strong> Strong seasonality (non-stationary)</p>
                <p><strong>Deseasonalized:</strong> May be stationary</p>
                <p><strong>Key Insight:</strong> Multiple seasonal patterns possible</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Generate time series data
        np.random.seed(42)
        t = np.arange(time_periods)
        
        # Base stationary series (white noise around mean)
        stationary_base = np.random.normal(0, noise_level, time_periods)
        
        # Non-stationary components
        trend_component = 0
        seasonal_component = 0
        volatility_component = 1
        
        if add_trend:
            if "Stock" in ts_scenario:
                # Stock-like random walk with drift
                trend_component = np.cumsum(np.random.normal(trend_strength/100, 0.1, time_periods))
            else:
                # Linear trend
                trend_component = trend_strength * t / 100
        
        if add_seasonality:
            if "Temperature" in ts_scenario:
                # Annual temperature cycle
                seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * t / 365)
            elif "Energy" in ts_scenario:
                # Energy has both daily and annual cycles
                seasonal_component = (seasonal_amplitude * np.sin(2 * np.pi * t / 365) +
                                   seasonal_amplitude * 0.3 * np.sin(2 * np.pi * t / 7))
            else:
                # Generic seasonal pattern
                seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * t / 50)
        
        if add_volatility_change:
            # Changing volatility over time
            volatility_component = 1 + 0.5 * np.sin(2 * np.pi * t / 200)
        
        # Combine components
        if "Stock" in ts_scenario:
            # Stock prices (non-stationary)
            stock_prices = 100 + trend_component + seasonal_component + stationary_base * volatility_component
            stock_prices = np.maximum(stock_prices, 1)  # Prices can't be negative
            
            # Stock returns (more stationary)
            stock_returns = np.diff(stock_prices) / stock_prices[:-1] * 100
            
            series_data = pd.DataFrame({
                'Time': t,
                'Stock_Price': stock_prices,
                'Stock_Return': np.concatenate([[0], stock_returns])  # Pad first return
            })
            
            # Plot both prices and returns
            fig = make_subplots(rows=2, cols=1,
                               subplot_titles=('Stock Prices (Non-stationary)', 'Stock Returns (More Stationary)'))
            
            fig.add_trace(
                go.Scatter(x=series_data['Time'], y=series_data['Stock_Price'],
                          mode='lines', name='Stock Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=series_data['Time'], y=series_data['Stock_Return'],
                          mode='lines', name='Stock Return (%)', line=dict(color='red')),
                row=2, col=1
            )
            
        else:
            # Other scenarios - single series
            if "Temperature" in ts_scenario:
                base_temp = 60
                series_value = base_temp + trend_component + seasonal_component + stationary_base * volatility_component
                series_name = "Temperature (°F)"
                
            elif "GDP" in ts_scenario:
                base_gdp = 1000
                series_value = base_gdp + trend_component * 50 + seasonal_component + stationary_base * volatility_component * 20
                series_name = "GDP (Billions $)"
                
            elif "Housing" in ts_scenario:
                base_price = 200000
                # Add bubble effect
                bubble_effect = 0
                if time_periods > 200:
                    bubble_peak = time_periods * 0.7
                    bubble_effect = 50000 * np.exp(-((t - bubble_peak) ** 2) / (2 * (time_periods * 0.1) ** 2))
                
                series_value = (base_price + trend_component * 1000 + seasonal_component * 1000 + 
                               bubble_effect + stationary_base * volatility_component * 5000)
                series_name = "House Price ($)"
                
            else:  # Energy
                base_energy = 100
                series_value = base_energy + trend_component * 10 + seasonal_component + stationary_base * volatility_component * 5
                series_name = "Energy Usage (kWh)"
            
            series_data = pd.DataFrame({
                'Time': t,
                'Value': series_value
            })
            
            # Single plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=series_data['Time'], y=series_data['Value'],
                          mode='lines', name=series_name, line=dict(color='blue'))
            )
            
            # Add trend line if trend is present
            if add_trend:
                trend_line = series_data['Value'].iloc[0] + trend_component
                fig.add_trace(
                    go.Scatter(x=series_data['Time'], y=trend_line,
                              mode='lines', name='Trend', line=dict(color='red', dash='dash'))
                )
        
        fig.update_layout(title=f"{ts_scenario}: Stationarity Analysis", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical tests and insights
        st.markdown("### 📊 Stationarity Analysis")
        
        # Calculate rolling statistics
        if "Stock" in ts_scenario:
            analysis_series = series_data['Stock_Price']
        else:
            analysis_series = series_data['Value']
        
        window = min(50, len(analysis_series) // 4)
        rolling_mean = analysis_series.rolling(window=window).mean()
        rolling_std = analysis_series.rolling(window=window).std()
        
        # Display rolling statistics
        fig_stats = make_subplots(rows=2, cols=1,
                                 subplot_titles=('Rolling Mean', 'Rolling Standard Deviation'))
        
        fig_stats.add_trace(
            go.Scatter(x=series_data['Time'], y=rolling_mean,
                      mode='lines', name='Rolling Mean'),
            row=1, col=1
        )
        
        fig_stats.add_trace(
            go.Scatter(x=series_data['Time'], y=rolling_std,
                      mode='lines', name='Rolling Std'),
            row=2, col=1
        )
        
        fig_stats.update_layout(title="Rolling Statistics (Constant for Stationary Series)")
        st.plotly_chart(fig_stats, use_container_width=True)
        
        # Stationarity assessment
        mean_change = abs(rolling_mean.iloc[-1] - rolling_mean.iloc[window]) if len(rolling_mean) > window else 0
        std_change = abs(rolling_std.iloc[-1] - rolling_std.iloc[window]) if len(rolling_std) > window else 0
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.metric("Mean Change", f"{mean_change:.2f}")
        with stat_col2:
            st.metric("Std Change", f"{std_change:.2f}")
        with stat_col3:
            if mean_change < 5 and std_change < 2:
                st.success("Likely Stationary")
            elif mean_change < 20 and std_change < 10:
                st.warning("Weakly Stationary")
            else:
                st.error("Non-stationary")

elif page == "⏰ Lagged Variables":
    st.markdown('<h2 class="section-header">⏰ Lagged Variables</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-box">
        <h4>📚 Core Concept:</h4>
        <p><strong>Lagged Variables:</strong> Using past values of variables to predict current or future values</p>
        <p><strong>Notation:</strong> X(t-1), X(t-2), ... represent values from 1, 2, ... periods ago</p>
        <p><strong>Applications:</strong> Time series forecasting, econometrics, sequential data analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    lag_scenario = st.selectbox("Choose a lagged variable scenario:", [
        "📈 Stock Price Prediction",
        "🌤️ Weather Forecasting", 
        "💰 Sales Forecasting",
        "🏥 Patient Health Monitoring",
        "🚗 Traffic Flow Prediction"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Lag Configuration")
        
        max_lags = st.slider("Maximum Lags to Include", 1, 20, 5)
        autocorr_strength = st.slider("Autocorrelation Strength", 0.0, 0.95, 0.6)
        seasonal_lags = st.checkbox("Include Seasonal Lags", False)
        show_prediction = st.checkbox("Show Prediction Example", True)
        
        if "Stock" in lag_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>📈 Financial Forecasting</h4>
                <p><strong>Question:</strong> Can past prices predict future prices?</p>
                <p><strong>Lag Features:</strong> Price(t-1), Price(t-2), Volume(t-1)</p>
                <p><strong>Challenge:</strong> Markets are often unpredictable!</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Weather" in lag_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🌤️ Meteorological Prediction</h4>
                <p><strong>Question:</strong> How does yesterday's weather predict tomorrow's?</p>
                <p><strong>Lag Features:</strong> Temp(t-1), Pressure(t-1), Humidity(t-1)</p>
                <p><strong>Insight:</strong> Weather has strong short-term persistence</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Sales" in lag_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>💰 Business Forecasting</h4>
                <p><strong>Question:</strong> How do past sales predict future sales?</p>
                <p><strong>Lag Features:</strong> Sales(t-1), Marketing(t-2), Season effects</p>
                <p><strong>Insight:</strong> Business cycles create predictable patterns</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif "Health" in lag_scenario:
            st.markdown("""
            <div class="example-box">
                <h4>🏥 Medical Monitoring</h4>
                <p><strong>Question:</strong> Can vital signs predict health changes?</p>
                <p><strong>Lag Features:</strong> Heart_rate(t-1), BP(t-1), Temp(t-1)</p>
                <p><strong>Insight:</strong> Health changes gradually, creating predictable trends</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Traffic
            st.markdown("""
            <div class="example-box">
                <h4>🚗 Urban Planning</h4>
                <p><strong>Question:</strong> How does past traffic predict future congestion?</p>
                <p><strong>Lag Features:</strong> Traffic(t-1), Weather(t-1), Events</p>
                <p><strong>Insight:</strong> Traffic patterns are highly regular and predictable</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Generate autocorrelated time series
        np.random.seed(42)
        n = 200
        
        # Create base time series with autocorrelation
        series = np.zeros(n)
        series[0] = np.random.normal(50, 10)
        
        # Generate series with specified autocorrelation
        for t in range(1, n):
            series[t] = (autocorr_strength * series[t-1] + 
                        (1 - autocorr_strength) * np.random.normal(50, 10))
            
            # Add scenario-specific patterns
            if "Weather" in lag_scenario:
                # Add seasonal temperature pattern
                series[t] += 15 * np.sin(2 * np.pi * t / 365) if t % 365 == 0 else 0
                
            elif "Sales" in lag_scenario:
                # Add monthly sales cycle
                series[t] += 20 * np.sin(2 * np.pi * t / 30)
                
            elif "Traffic" in lag_scenario:
                # Add weekly traffic pattern
                series[t] += 25 * np.sin(2 * np.pi * t / 7)
        
        # Create lagged features
        lag_data = pd.DataFrame({'Value': series, 'Time': range(n)})
        
        # Add lag columns
        for lag in range(1, max_lags + 1):
            lag_data[f'Lag_{lag}'] = lag_data['Value'].shift(lag)
        
        # Add seasonal lags if requested
        if seasonal_lags:
            if "Sales" in lag_scenario:
                lag_data['Lag_30'] = lag_data['Value'].shift(30)  # Monthly
            elif "Traffic" in lag_scenario:
                lag_data['Lag_7'] = lag_data['Value'].shift(7)   # Weekly
            elif "Weather" in lag_scenario:
                lag_data['Lag_365'] = lag_data['Value'].shift(365) if n > 365 else lag_data['Value'].shift(n//2)
        
        # Remove rows with NaN values
        lag_data_clean = lag_data.dropna()
        
        # Plot 1: Original time series
        fig1 = px.line(lag_data, x='Time', y='Value', 
                      title=f"{lag_scenario}: Original Time Series")
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Plot 2: Autocorrelation function
        autocorr_values = []
        for lag in range(1, max_lags + 1):
            if f'Lag_{lag}' in lag_data_clean.columns:
                corr = lag_data_clean['Value'].corr(lag_data_clean[f'Lag_{lag}'])
                autocorr_values.append({'Lag': lag, 'Autocorrelation': corr})
        
        if autocorr_values:
            autocorr_df = pd.DataFrame(autocorr_values)
            
            fig2 = px.bar(autocorr_df, x='Lag', y='Autocorrelation',
                         title="Autocorrelation Function (ACF)")
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            fig2.add_hline(y=0.2, line_dash="dot", line_color="gray", 
                          annotation_text="Weak correlation threshold")
            fig2.add_hline(y=-0.2, line_dash="dot", line_color="gray")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Plot 3: Lag scatter plots
        if len(autocorr_values) >= 3:
            fig3 = make_subplots(rows=1, cols=3,
                                subplot_titles=[f'Current vs Lag {i+1}' for i in range(3)])
            
            colors = ['red', 'blue', 'green']
            for i in range(3):
                lag_col = f'Lag_{i+1}'
                if lag_col in lag_data_clean.columns:
                    fig3.add_trace(
                        go.Scatter(x=lag_data_clean[lag_col], y=lag_data_clean['Value'],
                                  mode='markers', name=f'Lag {i+1}', 
                                  marker=dict(color=colors[i], size=4)),
                        row=1, col=i+1
                    )
            
            fig3.update_layout(title="Lag Relationships", height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Prediction example
        if show_prediction and len(lag_data_clean) > 20:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Prepare features and target
            feature_cols = [col for col in lag_data_clean.columns if col.startswith('Lag_')]
            X = lag_data_clean[feature_cols]
            y = lag_data_clean['Value']
            
            # Split into train/test
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Plot predictions
            test_time = lag_data_clean['Time'].iloc[split_point:].values
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=test_time, y=y_test, mode='lines+markers',
                                     name='Actual', line=dict(color='blue')))
            fig4.add_trace(go.Scatter(x=test_time, y=y_pred, mode='lines+markers',
                                     name='Predicted', line=dict(color='red', dash='dash')))
            
            fig4.update_layout(title=f"Prediction Results (R² = {r2:.3f})")
            st.plotly_chart(fig4, use_container_width=True)
            
            # Display metrics
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            with pred_col1:
                st.metric("R² Score", f"{r2:.3f}")
            with pred_col2:
                st.metric("RMSE", f"{np.sqrt(mse):.2f}")
            with pred_col3:
                quality = "Excellent" if r2 > 0.8 else "Good" if r2 > 0.6 else "Fair" if r2 > 0.4 else "Poor"
                st.metric("Prediction Quality", quality)
        
        # Feature importance
        if show_prediction and 'model' in locals():
            importance_df = pd.DataFrame({
                'Lag': range(1, len(model.coef_) + 1),
                'Coefficient': model.coef_
            })
            
            fig5 = px.bar(importance_df, x='Lag', y='Coefficient',
                         title="Lag Importance (Model Coefficients)")
            st.plotly_chart(fig5, use_container_width=True)

elif page == "🎲 Variable Relationships Quiz":
    st.markdown('<h2 class="section-header">🎲 Test Your Knowledge!</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="concept-box">
        <h4>🎯 Interactive Learning Quiz</h4>
        <p>Test your understanding of different variable types and relationships!</p>
        <p>Each question presents a scenario - identify the correct variable type and relationship.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quiz questions with hints and demo data
    quiz_questions = [
        {
            "question": "🍦 Ice cream sales are high when drowning incidents are high. What explains this relationship?",
            "options": ["Ice cream causes drowning", "Drowning causes ice cream sales", "Temperature affects both", "Pure coincidence"],
            "correct": 2,
            "explanation": "This is a classic confounding variable example. Hot weather increases both ice cream sales (people want cold treats) and drowning incidents (more people swim).",
            "concept": "Confounding Variable",
            "hint": "Think about what happens in summer that could increase both swimming and ice cream sales.",
            "demo": "confounder"
        },
        {
            "question": "📚 A model predicts student success using 'final exam score' as a feature. What's wrong?",
            "options": ["Nothing wrong", "Data leakage", "Confounding variable", "Interaction effect"],
            "correct": 1,
            "explanation": "This is data leakage! The final exam score wouldn't be available when making predictions about student success - it's part of what determines success.",
            "concept": "Leaky Variable",
            "hint": "Is the 'final exam score' available before the outcome?",
            "demo": None
        },
        {
            "question": "🎯 The effect of advertising depends on the season (holiday vs regular). This is an example of:",
            "options": ["Confounding", "Interaction", "Correlation", "Lagged effect"],
            "correct": 1,
            "explanation": "This is an interaction effect. The impact of advertising (X₁) depends on the season (X₂). Holiday advertising might be more effective than regular season advertising.",
            "concept": "Interaction Variable",
            "hint": "Does the effect of advertising change depending on another variable?",
            "demo": "interaction"
        },
        {
            "question": "🏥 To test a new drug, researchers match patients by age before assigning treatments. Age is a:",
            "options": ["Dependent variable", "Independent variable", "Control variable", "Latent variable"],
            "correct": 2,
            "explanation": "Age is a control variable. Researchers are controlling for age to isolate the true effect of the drug treatment.",
            "concept": "Control Variable",
            "hint": "What is being held constant to ensure a fair comparison?",
            "demo": None
        },
        {
            "question": "📈 Stock prices today strongly correlate with stock prices yesterday. This suggests:",
            "options": ["Causation", "Autocorrelation", "Confounding", "Data leakage"],
            "correct": 1,
            "explanation": "This is autocorrelation - a variable correlated with its own past values. Stock prices exhibit 'momentum' or persistence over short periods.",
            "concept": "Lagged/Autocorrelation",
            "hint": "Is the variable related to its own past values?",
            "demo": "lagged"
        }
    ]

    # Quiz interface
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.answers = []
        st.session_state.show_hint = False

    if not st.session_state.quiz_started:
        st.markdown("### 🚀 Ready to test your knowledge?")
        if st.button("Start Quiz", type="primary"):
            st.session_state.quiz_started = True
            st.rerun()

    else:
        current_q = st.session_state.current_question
        if current_q < len(quiz_questions):
            question = quiz_questions[current_q]
            progress = (current_q + 1) / len(quiz_questions)
            st.progress(progress)
            st.markdown(f"**Question {current_q + 1} of {len(quiz_questions)}**")
            st.markdown(f"### {question['question']}")

            # Show demo DataFrame or plot if available
            if question.get('demo') == 'confounder':
                # Show a small DataFrame for temperature, ice cream sales, drownings
                np.random.seed(42)
                temp = np.random.randint(70, 101, 8)
                ice_cream = temp * 5 + np.random.normal(0, 10, 8)
                drownings = temp * 0.2 + np.random.normal(0, 1, 8)
                df = pd.DataFrame({'Temperature (F)': temp, 'Ice Cream Sales': ice_cream.round(1), 'Drownings': drownings.round(2)})
                st.dataframe(df)
                fig = px.scatter(df, x='Temperature (F)', y=['Ice Cream Sales', 'Drownings'], title='Both Sales and Drownings Rise with Temperature')
                st.plotly_chart(fig)
            elif question.get('demo') == 'interaction':
                # Show a DataFrame for ad spend, season, and sales
                np.random.seed(0)
                ad_spend = np.repeat([1000, 5000, 10000], 2)
                season = ['Regular', 'Holiday'] * 3
                sales = [2000, 3000, 4000, 7000, 6000, 12000]
                df = pd.DataFrame({'Ad Spend': ad_spend, 'Season': season, 'Sales': sales})
                st.dataframe(df)
                fig = px.bar(df, x='Ad Spend', y='Sales', color='Season', barmode='group', title='Ad Effect by Season')
                st.plotly_chart(fig)
            elif question.get('demo') == 'lagged':
                # Show a time series with lag
                np.random.seed(1)
                days = pd.date_range('2023-01-01', periods=10)
                price = np.cumsum(np.random.normal(0, 2, 10)) + 100
                df = pd.DataFrame({'Day': days, 'Price': price})
                df['Price (Yesterday)'] = df['Price'].shift(1)
                st.dataframe(df)
                fig = px.line(df, x='Day', y=['Price', 'Price (Yesterday)'], title='Stock Price and Lagged Value')
                st.plotly_chart(fig)

            # Show hint button
            if not st.session_state.show_hint:
                if st.button("Show Hint", key=f"hint_{current_q}"):
                    st.session_state.show_hint = True
                    st.experimental_rerun()
            else:
                st.info(f"💡 Hint: {question['hint']}")

            selected_answer = st.radio(
                "Choose your answer:",
                options=range(len(question['options'])),
                format_func=lambda x: question['options'][x],
                key=f"q_{current_q}"
            )

            if st.button("Submit Answer", key=f"submit_{current_q}"):
                is_correct = selected_answer == question['correct']
                st.session_state.answers.append({
                    'question': current_q,
                    'selected': selected_answer,
                    'correct': question['correct'],
                    'is_correct': is_correct
                })
                if is_correct:
                    st.session_state.score += 1
                    st.success(f"✅ Correct! {question['explanation']}")
                else:
                    st.error(f"❌ Incorrect. {question['explanation']}")
                st.info(f"💡 **Concept:** {question['concept']}")
                st.session_state.show_hint = False
                st.session_state.auto_advance = True
                st.session_state.auto_advance_time = time.time()
                st.experimental_rerun()

            # Auto-advance logic
            if st.session_state.get('auto_advance', False):
                if time.time() - st.session_state.get('auto_advance_time', 0) > 1.5:
                    st.session_state.current_question += 1
                    st.session_state.show_hint = False
                    st.session_state.auto_advance = False
                    st.experimental_rerun()
                else:
                    st.info("Moving to next question...")
        else:
            st.markdown("## 🎉 Quiz Completed!")
            score_percentage = (st.session_state.score / len(quiz_questions)) * 100
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Score", f"{st.session_state.score}/{len(quiz_questions)}")
            with col2:
                st.metric("Percentage", f"{score_percentage:.0f}%")
            with col3:
                if score_percentage >= 80:
                    st.success("Excellent! 🌟")
                elif score_percentage >= 60:
                    st.info("Good Job! 👍")
                else:
                    st.warning("Keep Learning! 📚")
            st.markdown("### 📊 Detailed Results")
            for i, answer in enumerate(st.session_state.answers):
                question = quiz_questions[answer['question']]
                st.markdown(f"**Q{i+1}: {question['question']}**")
                st.markdown(f"- Your answer: {question['options'][answer['selected']]}")
                st.markdown(f"- Correct answer: {question['options'][question['correct']]}")
                if answer['is_correct']:
                    st.success(f"✅ {question['concept']}")
                else:
                    st.error(f"❌ {question['concept']} - Review this concept!")
                st.info(f"Explanation: {question['explanation']}")
                if question.get('demo') == 'confounder':
                    np.random.seed(42)
                    temp = np.random.randint(70, 101, 8)
                    ice_cream = temp * 5 + np.random.normal(0, 10, 8)
                    drownings = temp * 0.2 + np.random.normal(0, 1, 8)
                    df = pd.DataFrame({'Temperature (F)': temp, 'Ice Cream Sales': ice_cream.round(1), 'Drownings': drownings.round(2)})
                    st.dataframe(df)
                elif question.get('demo') == 'interaction':
                    np.random.seed(0)
                    ad_spend = np.repeat([1000, 5000, 10000], 2)
                    season = ['Regular', 'Holiday'] * 3
                    sales = [2000, 3000, 4000, 7000, 6000, 12000]
                    df = pd.DataFrame({'Ad Spend': ad_spend, 'Season': season, 'Sales': sales})
                    st.dataframe(df)
                elif question.get('demo') == 'lagged':
                    np.random.seed(1)
                    days = pd.date_range('2023-01-01', periods=10)
                    price = np.cumsum(np.random.normal(0, 2, 10)) + 100
                    df = pd.DataFrame({'Day': days, 'Price': price})
                    df['Price (Yesterday)'] = df['Price'].shift(1)
                    st.dataframe(df)
            if st.button("Restart Quiz"):
                st.session_state.quiz_started = False
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.answers = []
                st.experimental_rerun()

# Footer with learning resources
st.markdown("---")
st.markdown("""
### 🎓 **Congratulations on Exploring Variables!**

You've now experienced interactive examples of all major variable types in data analysis. Here's what you've learned:

**🎯 Core Variable Types:**
- **Independent & Dependent**: The foundation of cause-and-effect analysis
- **Interaction Variables**: When effects depend on combinations of factors  
- **Latent Variables**: Hidden factors that influence what we observe
- **Confounding Variables**: The sneaky culprits behind false correlations

**🛠️ Analysis Techniques:**
- **Control Variables**: Isolating true effects through careful design
- **Correlated Variables**: Understanding relationships without assuming causation
- **Leaky Variables**: Avoiding the pitfall of using future information
- **Time Series Properties**: Stationarity and temporal dependencies
- **Lagged Variables**: Using the past to predict the future

**💡 Key Takeaways:**
1. **Always question relationships** - Correlation ≠ Causation
2. **Consider hidden variables** - What might be affecting both variables?
3. **Think about timing** - Are you accidentally using future information?
4. **Control for confounders** - Good experimental design is crucial
5. **Domain knowledge matters** - Understanding your field prevents mistakes

### 🚀 **Next Steps:**
- Practice identifying variable types in your own projects
- Always check for potential confounders in your analyses
- Be vigilant about data leakage in machine learning models
- Consider interactions when relationships seem complex

**Remember:** Understanding your variables is the foundation of good data science! 🌟
""")