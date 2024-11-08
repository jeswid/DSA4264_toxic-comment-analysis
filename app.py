import numpy as np 
import pandas as pd 
import streamlit as st  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import plotly.graph_objects as go
import datetime

st.set_page_config(
    page_title="Toxic Reddit Topic Tracker",
    page_icon="🌐",
    layout="wide",
)
# Title
st.title("🌐 Toxic Reddit Topic Tracker")

################ Inputs ################
# Sidebar
st.sidebar.header("Chart Input Parameters")

# Set min and max dates
min_date = datetime.date(2020, 1, 1)
max_date = datetime.date(2023, 12, 31)

# Sidebar - Start date input
start_date = st.sidebar.date_input(
    "Start Date",
    min_date,  # default to min date
    min_value=min_date,
    max_value=max_date,
)

# Sidebar - End date input
end_date = st.sidebar.date_input(
    "End Date",
    max_date,  # default to max date
    min_value=min_date,
    max_value=max_date,
)

## Sidebar - Choose topics
policy_options = ["Religion", "SG Politics", "Covid", "Housing", "Transport"]
policy_chosen = st.sidebar.pills("Social and Governance Topics ", policy_options, selection_mode="multi")

entertainment_options = ["Sports", "Music", "Gaming", "Media"]
entertainment_chosen = st.sidebar.pills("Entertainment Topics ", entertainment_options, selection_mode="multi")

################ Backend ################
df = pd.read_csv("updated_full_comments_new_topic.csv", index_col=0)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['Month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
df['Toxic'] = df['predicted_label'].apply(lambda x: 1 if x == 'toxic' else 0)


monthly_toxicity = df.groupby(['Month', 'new_topic'])['Toxic'].mean().reset_index()
monthly_toxicity['Month'] = monthly_toxicity['Month'].dt.to_timestamp()

monthly_toxicity_filtered = monthly_toxicity[(monthly_toxicity['Month'] >= pd.to_datetime(start_date)) 
                                             & (monthly_toxicity['Month'] <= pd.to_datetime(end_date))]

names_to_number = {"Religion": 1, "SG Politics": 2, "Covid": 3, "Housing": 5, "Transport": 8,
                   "Sports": 4, "Music": 6, "Gaming": 7, "Media": 9}

policy_govt = monthly_toxicity_filtered[(monthly_toxicity_filtered['new_topic'].isin([1, 2, 3, 5, 8]))]
policy_govt['Toxic_MA'] = policy_govt.groupby('new_topic')['Toxic'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
policy_names_to_number = {"Religion": 1, "SG Politics": 2, "Covid": 3, "Housing": 5, "Transport": 8}
policy_topic_colors = {1: "#8E44AD", 2: "#E74C3C", 3: "#3498DB", 5: "#F39C12", 8: "#27AE60"}

entertainment = monthly_toxicity_filtered[(monthly_toxicity_filtered['new_topic'].isin([4, 6, 7, 9]))]
entertainment['Toxic_MA'] = entertainment.groupby('new_topic')['Toxic'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
entertainment_names_to_number = {"Sports": 4, "Music": 6, "Gaming": 7, "Media": 9}
entertainment_topic_colors = {4: "#27AE60", 6: "#E74C3C", 7: "#3498DB", 9: "#F39C12"}

################ Frontend ################
with st.container(border=True):
    st.subheader("Charts")
    st.write(f"Charts tracking the number of toxic comments from {start_date} to {end_date} for selected topics")
    tab1, tab2 = st.tabs(["Policy and Governance", "Entertainment"])

    with tab1:
        # Initialize figure
        fig = go.Figure()

        # Add each topic line as a separate trace for full customization
        for topic in policy_chosen:
            topic_number = policy_names_to_number.get(topic)
            topic_data = policy_govt[policy_govt['new_topic'] == topic_number]
            label = topic
            color = policy_topic_colors.get(topic_number) 
            
            # Add a line trace for each topic
            fig.add_trace(
                go.Scatter(
                    x=topic_data['Month'],
                    y=topic_data['Toxic_MA'],
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                    opacity=0.9,
                    hoverinfo="x+y+name"  # Customize hover info if needed
                )
            )

        # Customize layout
        fig.update_layout(
            title="Proportion of Toxic Comments: Social & Governance (Smoothed)",
            title_font=dict(size=18, color="#333333"),
            xaxis_title="Month",
            yaxis_title="Proportion of Toxic Comments",
            legend_title="Topic",
            font=dict(size=12, color="#333333"),
            xaxis_tickangle=45,
            legend=dict(
                title_font_size=12,
                font_size=10,
                bgcolor="rgba(255, 255, 255, 0.5)",  # Slightly transparent background
                bordercolor="Black",
                borderwidth=1,
            ),
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2: 
        # Initialize figure
        fig = go.Figure()

        # Add each topic line as a separate trace for full customization
        for topic in entertainment_chosen:
            topic_number = entertainment_names_to_number.get(topic)
            topic_data = entertainment[entertainment['new_topic'] == topic_number]
            label = topic
            color = entertainment_topic_colors.get(topic_number) 
            
            # Add a line trace for each topic
            fig.add_trace(
                go.Scatter(
                    x=topic_data['Month'],
                    y=topic_data['Toxic_MA'],
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                    opacity=0.9,
                    hoverinfo="x+y+name"  # Customize hover info if needed
                )
            )

        # Customize layout
        fig.update_layout(
            title="Proportion of Toxic Comments: Entertainment (Smoothed)",
            title_font=dict(size=18, color="#333333"),
            xaxis_title="Month",
            yaxis_title="Proportion of Toxic Comments",
            legend_title="Topic",
            font=dict(size=12, color="#333333"),
            xaxis_tickangle=45,
            legend=dict(
                title_font_size=12,
                font_size=10,
                bgcolor="rgba(255, 255, 255, 0.5)",  # Slightly transparent background
                bordercolor="Black",
                borderwidth=1,
            ),
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)


with st.container(border=True):
    st.subheader("Toxic Comments")
    st.write("Retrieve toxic comments for selected topic in a particular month")

    #all_options = ["Religion", "SG Politics", "Covid", "Housing", "Transport", "Sports", "Music", "Gaming", "Media"] 
    all_options = policy_chosen + entertainment_chosen
    topics_chosen = st.pills("Selected Topics", all_options, selection_mode= "multi")
    numeric_topics_chosen = [names_to_number[topic] for topic in topics_chosen]

    # Month and year options
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    years = [2020, 2021, 2022, 2023]
    
    # Month and year selectors
    col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
    with col1:
        selected_month = st.selectbox("Month", months)
    with col2: 
        selected_year = st.selectbox("Year", years)

    table_data = df.copy()
    table_data['Month'] = table_data['Month'].dt.to_timestamp()

    # Extract year and month for filtering
    table_data['Year'] = table_data['Month'].dt.year
    table_data['Month_Name'] = table_data['Month'].dt.strftime('%B')  # Converts month to full name (e.g., "January")

    filtered_df = table_data[(table_data['Year'] == selected_year) & 
                 (table_data['Month_Name'] == selected_month) & 
                 (table_data['new_topic'].isin(numeric_topics_chosen)) &
                 (table_data['Toxic'] == 1)].reset_index()
    
    st.dataframe(filtered_df[['timestamp', 'original text', 'new_topic']], use_container_width=True)


    

