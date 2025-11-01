import streamlit as st
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import os
from fpdf import FPDF
import plotly.graph_objects as go

# --- App Configuration ---
st.set_page_config(page_title="Skill Gap Analyzer", page_icon="🎓", layout="wide")

# --- Data Dictionaries & Constants ---

# 1. Job Skill Requirements
JOB_SKILL_REQUIREMENTS = {
    'Software Developer': ['Java', 'Python', 'SQL', 'Data Structures', 'Problem-Solving', 'Teamwork', 'Git'],
    'AI Engineer': ['Python', 'Machine Learning', 'TensorFlow', 'PyTorch', 'Deep Learning', 'Statistics', 'NLP'],
    'Data Scientist': ['Python', 'R', 'SQL', 'Machine Learning', 'Statistics', 'Data Visualization', 'Communication'],
    'Cloud Architect': ['Cloud Computing', 'AWS', 'Azure', 'Networking', 'Docker', 'Kubernetes', 'Leadership'],
    'Cybersecurity Expert': ['Cybersecurity', 'Networking', 'Python', 'Cryptography', 'Ethical Hacking', 'Penetration Testing'],
    'Data Analyst': ['SQL', 'Excel', 'Tableau', 'Power BI', 'Statistics', 'Data Cleaning', 'Communication']
}

# 2. Learning Resources for Missing Skills (UPDATED)
LEARNING_RESOURCES = {
    'java': 'https://www.coursera.org/learn/java-programming',
    'python': 'https://www.coursera.org/learn/python-for-everybody',
    'sql': 'https://www.udemy.com/course/the-complete-sql-bootcamp/',
    'data structures': 'https://www.geeksforgeeks.org/data-structures/',
    'problem-solving': 'https://leetcode.com/problemset/all/',
    'teamwork': 'https://www.coursera.org/learn/teamwork-skills-communication-and-collaboration',
    'git': 'https://www.udemy.com/course/git-expert-4-hours/',
    'machine learning': 'https://www.coursera.org/learn/machine-learning',
    'tensorflow': 'https://www.coursera.org/specializations/tensorflow-in-practice',
    'pytorch': 'https://www.udemy.com/course/pytorch-for-deep-learning-with-python-bootcamp/',
    'deep learning': 'https://www.coursera.org/specializations/deep-learning',
    'statistics': 'https://www.khanacademy.org/math/statistics-probability',
    'nlp': 'https://www.coursera.org/learn/natural-language-processing-specialization',
    'r': 'https://www.coursera.org/learn/r-programming',
    'data visualization': 'https://www.coursera.org/learn/datavisualization',
    'communication': 'https://www.coursera.org/learn/communication-skills-for-engineers',
    'cloud computing': 'https://www.coursera.org/learn/introduction-to-cloud',
    'aws': 'https://aws.amazon.com/training/digital/',
    'azure': 'https://learn.microsoft.com/en-us/training/azure/',
    'networking': 'https://www.coursera.org/professional-certificates/google-it-support',
    'docker': 'https://www.udemy.com/course/docker-for-the-absolute-beginner/',
    'kubernetes': 'https://www.udemy.com/course/learn-kubernetes/',
    'leadership': 'https://www.coursera.org/learn/leadership-management',
    'cybersecurity': 'https://www.coursera.org/professional-certificates/google-cybersecurity',
    'cryptography': 'https://www.coursera.org/learn/crypto',
    'ethical hacking': 'https://www.coursera.org/learn/introduction-cyber-security-foundations',
    'penetration testing': 'https://www.coursera.org/learn/penetration-testing',
    'excel': 'https://www.coursera.org/specializations/excel-skills-for-business',
    'tableau': 'https://www.coursera.org/learn/tableau-for-data-science',
    'power bi': 'https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/',
    'data cleaning': 'https://www.coursera.org/learn/data-cleaning' # ADDED THIS
}

# 3. Project Ideas for Practice (UPDATED)
PROJECT_IDEAS = {
    'python': 'Build a simple web scraper to get headlines from a news website.',
    'sql': 'Analyze a sample sales dataset to find the top 5 most profitable products.',
    'tableau': 'Create an interactive dashboard for COVID-19 cases by country.',
    'machine learning': 'Build a model to predict house prices based on features like area and number of rooms.',
    'java': 'Create a simple command-line banking application.',
    'aws': 'Deploy a simple static website using an S3 bucket.',
    'docker': 'Containerize a simple Python web application.',
    'data cleaning': 'Find a messy dataset on Kaggle and clean it for analysis (e.g., handle missing values, correct data types).', # ADDED THIS
    'data visualization': 'Create a compelling story using charts and graphs from a public dataset (e.g., Titanic dataset).'
}


# 4. List of common skills for multi-select
ALL_SKILLS_LIST = sorted(list(set(skill for sublist in JOB_SKILL_REQUIREMENTS.values() for skill in sublist)))


# --- Data Loading and ML Model Training ---

@st.cache_data
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: The file was not found at {file_path}")
        return None
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('_X', '', regex=False).str.strip()
    df.dropna(subset=['job_role_aspiration', 'technical_skills', 'soft_skills'], inplace=True)

    def analyze_skill_gap(row):
        job_role = row['job_role_aspiration']
        if job_role not in JOB_SKILL_REQUIREMENTS:
            return 0
        required = set(s.lower() for s in JOB_SKILL_REQUIREMENTS[job_role])
        student_skills = set(s.lower().strip() for s in re.split(r'[,\s]+', str(row['technical_skills']) + ',' + str(row['soft_skills'])) if s)
        missing = required - student_skills
        return 1 if len(required) > 0 and (len(missing) / len(required)) > 0.5 else 0

    df['skill_gap'] = df.apply(analyze_skill_gap, axis=1)
    return df

@st.cache_resource
def train_model(df):
    df['all_skills'] = df['technical_skills'].str.cat(df['soft_skills'], sep=',')
    X = df[['current_course', 'year', 'all_skills']]
    y = df['skill_gap']

    X.loc[:, 'all_skills'] = X['all_skills'].apply(lambda x: [skill.strip() for skill in str(x).split(',')])

    mlb = MultiLabelBinarizer()
    one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    skills_encoded = mlb.fit_transform(X['all_skills'])
    cats_encoded = one_hot.fit_transform(X[['current_course', 'year']])

    skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_, index=X.index)
    cats_df = pd.DataFrame(cats_encoded, columns=one_hot.get_feature_names_out(), index=X.index)

    X_processed = pd.concat([cats_df, skills_df], axis=1)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_processed, y)

    return model, one_hot, mlb, X_processed.columns

# --- PDF Report Generation ---

def generate_report(job_aspiration, course, year, user_skills_list, prediction, proba, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt="Skill Gap Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 8, txt="Student Details", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt=f"Aspired Job Role: {job_aspiration}", ln=True)
    pdf.cell(200, 8, txt=f"Current Course: {course}", ln=True)
    pdf.cell(200, 8, txt=f"Year of Study: {year}", ln=True)
    pdf.multi_cell(0, 8, txt=f"Your Skills: {', '.join(user_skills_list)}", align='L')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 8, txt="Analysis Result", ln=True)
    pdf.set_font("Arial", size=12)

    if prediction == 1:
        pdf.cell(200, 8, txt="Prediction: Skill Gap Exists", ln=True)
        pdf.cell(200, 8, txt=f"Confidence: {proba[1]*100:.2f}%", ln=True)
        if recommendations:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(200, 8, txt="Recommendations to Bridge the Gap:", ln=True)
            pdf.set_font("Arial", size=12)
            for skill in recommendations:
                pdf.cell(200, 8, txt=f"- Learn {skill.title()}", ln=True)
                resource = LEARNING_RESOURCES.get(skill.lower(), "N/A")
                idea = PROJECT_IDEAS.get(skill.lower(), "N/A")
                pdf.cell(200, 8, txt=f"  - Resource: {resource}", ln=True)
                pdf.cell(200, 8, txt=f"  - Project Idea: {idea}", ln=True)
    else:
        pdf.cell(200, 8, txt="Prediction: No Significant Skill Gap", ln=True)
        pdf.cell(200, 8, txt=f"Confidence: {proba[0]*100:.2f}%", ln=True)
        pdf.cell(200, 8, txt="You are on the right track! Keep building your profile.", ln=True)

    return pdf.output(dest='S').encode('latin1')

# --- Main Application Logic ---

def main():
    st.title("🎓 AI-Powered Skill Gap Analyzer")
    st.write("Enter your details to predict your skill gap, get learning resources, and see a visual analysis.")

    file_path = r'C:\Users\debup\OneDrive\Desktop\skill_gap_project\Dataset\Final_Updated_DMA_DATASET_Indian_Names.xlsx'
    df = load_and_preprocess_data(file_path)
    if df is None:
        return
        
    model, one_hot, mlb, X_columns = train_model(df)

    # --- Sidebar for User Input ---
    st.sidebar.header("👤 Enter Your Details")
    job_aspiration = st.sidebar.selectbox("Aspired Job Role", options=list(JOB_SKILL_REQUIREMENTS.keys()))
    
    course_options = sorted(list(df['current_course'].unique()) + ['MCA'])
    course = st.sidebar.selectbox("Current Course", options=course_options)
    
    year = st.sidebar.selectbox("Year of Study", options=sorted(df['year'].unique()))
    
    # NEW: Multi-select for skills
    user_skills_list = st.sidebar.multiselect("Select Your Skills", options=ALL_SKILLS_LIST, default=["Python", "SQL", "Communication"])

    if st.sidebar.button("Analyze Skill Gap", type="primary"):
        if not user_skills_list:
            st.sidebar.warning("Please select at least one skill.")
            return

        user_df = pd.DataFrame({
            'current_course': [course], 'year': [year], 'all_skills': [user_skills_list]
        })

        user_cats_encoded = one_hot.transform(user_df[['current_course', 'year']])
        user_skills_encoded = mlb.transform(user_df['all_skills'])

        user_cats_df = pd.DataFrame(user_cats_encoded, columns=one_hot.get_feature_names_out())
        user_skills_df = pd.DataFrame(user_skills_encoded, columns=mlb.classes_)

        user_processed_df = pd.concat([user_cats_df, user_skills_df], axis=1)
        user_processed_df = user_processed_df.reindex(columns=X_columns, fill_value=0)

        prediction = model.predict(user_processed_df)[0]
        proba = model.predict_proba(user_processed_df)[0]

        # --- Display Results ---
        st.header("📊 Analysis Result")
        col1, col2 = st.columns(2)

        required_skills_set = set(s.lower() for s in JOB_SKILL_REQUIREMENTS[job_aspiration])
        owned_skills_set = set(s.lower() for s in user_skills_list)
        recommendations = list(required_skills_set - owned_skills_set)

        if prediction == 1:
            col1.error("**Prediction: Skill Gap Exists**")
            col2.warning(f"Confidence: {proba[1]*100:.2f}%")
        else:
            col1.success("**Prediction: No Significant Skill Gap**")
            col2.info(f"Confidence: {proba[0]*100:.2f}%")

        # --- NEW: Interactive Radar Chart ---
        st.write("---")
        st.subheader("Visual Skill Analysis")

        radar_labels = list(required_skills_set)
        owned_values = [1 if skill in owned_skills_set else 0 for skill in radar_labels]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[1]*len(radar_labels), # Required skills line
            theta=radar_labels,
            fill='toself',
            name='Required Skills'
        ))
        fig.add_trace(go.Scatterpolar(
            r=owned_values, # Your skills line
            theta=radar_labels,
            fill='toself',
            name='Your Skills'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1])
            ),
            showlegend=True,
            title=f"Skill Comparison for {job_aspiration}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- NEW: Recommendations with Resources and Projects ---
        st.write("---")
        if recommendations:
            st.subheader("💡 Your Personalized Learning Path")
            for skill in recommendations:
                with st.expander(f"Learn {skill.title()}"):
                    st.markdown(f"**Why it's needed:** A key skill for a {job_aspiration}.")
                    
                    resource_link = LEARNING_RESOURCES.get(skill.lower(), "https://www.google.com/search?q=learn+" + skill.replace(' ', '+'))
                    st.markdown(f"**🎓 Recommended Resource:** [Click here to learn]({resource_link})")
                    
                    project_idea = PROJECT_IDEAS.get(skill.lower(), "Try to build a small application using this skill.")
                    st.markdown(f"**🛠️ Practice Project:** {project_idea}")
        else:
            st.subheader("🚀 You're on the right track!")
            st.markdown("You possess all the key skills for this role. Keep honing them by building more complex projects!")

        # --- Download Report Button ---
        st.write("---")
        report_pdf = generate_report(job_aspiration, course, year, user_skills_list, prediction, proba, recommendations)
        st.download_button(
            label="📥 Download Full Report (PDF)",
            data=report_pdf,
            file_name=f"{job_aspiration.replace(' ', '_')}_Skill_Report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()