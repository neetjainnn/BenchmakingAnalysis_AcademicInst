import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import re

st.set_page_config(page_title="University Rankings Analysis", layout="wide")
st.title("University Rankings Analysis Dashboard")

def clean_university_name(name):
    if pd.isna(name):
        return ""
    name = name.lower().strip()
    name = re.sub(r"\(.*?\)", "", name)
    return name.strip()

def clean_qs_ranking(df):
    df.columns = df.columns.str.strip().str.lower()
    
    column_mapping = {
        "institution name": "institution",
        "academic reputation score": "academic reputation",
        "employer reputation score": "employer reputation",
        "faculty student score": "faculty student",
        "citations per faculty score": "citations per faculty",
        "international faculty score": "international faculty",
        "international students score": "international students",
        "international research network score": "international research network",
        "employment outcomes score": "employment outcomes",
        "sustainability score": "sustainability"
    }
    df.rename(columns=column_mapping, inplace=True)
    
    if "institution" not in df.columns:
        st.error("Institution column missing in QS Ranking dataset")
        return None
    
    ranking_criteria = [col for col in column_mapping.values() if col in df.columns and col != "institution"]
    df.dropna(subset=['institution'], inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, ax=ax)
    plt.title("QS Ranking Dataset - Null Values Before Cleaning")
    st.pyplot(fig)
    plt.close()
    
    for col in ranking_criteria:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, ax=ax)
    plt.title("QS Ranking Dataset - Null Values After Cleaning")
    st.pyplot(fig)
    plt.close()
    
    weights = {
        "academic reputation": 0.4,
        "employer reputation": 0.1,
        "faculty student": 0.2,
        "citations per faculty": 0.15,
        "international faculty": 0.05,
        "international students": 0.05,
        "international research network": 0.025,
        "employment outcomes": 0.025,
        "sustainability": 0.025
    }
    
    scaler = MinMaxScaler()
    df[ranking_criteria] = scaler.fit_transform(df[ranking_criteria])
    df["final_score"] = np.tanh(df[ranking_criteria].mul(pd.Series(weights)).sum(axis=1))
    df["rank_percentile"] = df["final_score"].rank(pct=True) * 100
    # Add explicit rank calculation
    df["rank"] = df["final_score"].rank(ascending=False, method='min').astype(int)
    
    return df, ranking_criteria, weights

def clean_nirf_ranking(df):
    df.columns = df.columns.str.strip().str.lower()
    
    if "institution" not in df.columns:
        st.error("Institution column missing in NIRF Ranking dataset")
        return None
    
    ranking_criteria = ['tlr', 'rpc', 'go', 'oi', 'perception']
    ranking_criteria = [col for col in ranking_criteria if col in df.columns]
    
    df.dropna(subset=['institution'], inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, ax=ax)
    plt.title("NIRF Ranking Dataset - Null Values Before Cleaning")
    st.pyplot(fig)
    plt.close()
    
    for col in ranking_criteria:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, ax=ax)
    plt.title("NIRF Ranking Dataset - Null Values After Cleaning")
    st.pyplot(fig)
    plt.close()
    
    weights = {
        "tlr": 0.3,
        "rpc": 0.3,
        "go": 0.2,
        "oi": 0.1,
        "perception": 0.1
    }
    
    scaler = MinMaxScaler()
    df[ranking_criteria] = scaler.fit_transform(df[ranking_criteria])
    df["final_score"] = np.tanh(df[ranking_criteria].mul(pd.Series(weights)).sum(axis=1))
    df["rank_percentile"] = df["final_score"].rank(pct=True) * 100
    
    return df, ranking_criteria, weights

def visualize_rankings(df, ranking_criteria, title, weights):
    if "final_score" not in df.columns or "institution" not in df.columns:
        st.error(f"Skipping visualization for {title} due to missing columns.")
        return
    
    # Add number input for top N universities
    num_universities = st.number_input(
        f"Select number of top universities to display for {title}",
        min_value=1,
        max_value=len(df),
        value=10,
        step=1
    )
    
    # Add weight adjustment section with default button
    st.subheader("Adjust Weights")
    
    # Create columns for default button and weight adjustment
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button(f"Reset to Default Weights ({title})"):
            # Keep the original weights
            adjusted_weights = weights.copy()
        else:
            # Allow weight adjustment
            adjusted_weights = {}
            for criterion, default_weight in weights.items():
                adjusted_weights[criterion] = st.slider(
                    f"{criterion.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
                    step=0.05,
                    key=f"{title}_{criterion}"
                )
    
    # Normalize weights to sum to 1
    total_weight = sum(adjusted_weights.values())
    if total_weight > 0:
        adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
    
    # Recalculate scores with new weights
    df["final_score"] = np.tanh(df[ranking_criteria].mul(pd.Series(adjusted_weights)).sum(axis=1))
    df["rank_percentile"] = df["final_score"].rank(pct=True) * 100
    df["rank"] = df["final_score"].rank(ascending=False, method='min').astype(int)
    
    top_n = df.nlargest(num_universities, "final_score").copy()
    
    # Add button to show bar plot
    if st.button(f"Show Top Universities Bar Plot ({title})"):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(y=top_n["institution"], x=top_n["final_score"], palette="Blues_r", ax=ax)
        plt.xlabel("Final Score")
        plt.ylabel("Institution")
        plt.title(f"Top {num_universities} Institutions by Final Score ({title})")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
        plt.close()
    
    st.write(f"Top {num_universities} {title} Institutions:")
    st.dataframe(top_n[["institution", "final_score", "rank_percentile"]].reset_index(drop=True).style.format({
        "final_score": "{:.3f}",
        "rank_percentile": "{:.2f}%"
    }))
    
    # Add button to show correlation heatmap
    if st.button(f"Show Correlation Heatmap ({title})"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[ranking_criteria].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.title("Correlation Heatmap of Ranking Criteria")
        st.pyplot(fig)
        plt.close()
    
    # Add button to show weights distribution pie chart
    if st.button(f"Show Weights Distribution ({title})"):
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.pie(adjusted_weights.values(), labels=adjusted_weights.keys(), autopct='%1.0f%%', colors=sns.color_palette("pastel"))
        plt.title(f"Weight Distribution for {title} Criteria")
        st.pyplot(fig)
        plt.close()
        
    return df  # Return the dataframe with updated scores

def comparative_analysis(qs_df, nirf_df):
    st.subheader("Compare Your University with Dream University")
    
    # First select the ranking system
    ranking_system = st.radio("Select Ranking System:", ["QS", "NIRF"], horizontal=True)
    
    # Get the dataframe based on selected ranking system
    df = qs_df if ranking_system == "QS" else nirf_df
    
    # Add rank column if not exists
    if "rank" not in df.columns:
        df["rank"] = df["final_score"].rank(ascending=False, method='min').astype(int)
    
    # Show available ranks for reference
    st.write(f"Available ranks in {ranking_system} rankings: 1 to {len(df)}")
    
    # Create two columns for rank inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Current University")
        current_rank = st.number_input(
            "Enter your current university rank:",
            min_value=1,
            max_value=len(df),
            value=1
        )
        current_univ = df[df["rank"] == current_rank]
        if not current_univ.empty:
            st.info(f"Selected: {current_univ['institution'].iloc[0]}")
    
    with col2:
        st.write("### Dream University")
        dream_rank = st.number_input(
            "Enter your dream university rank:",
            min_value=1,
            max_value=len(df),
            value=1
        )
        dream_univ = df[df["rank"] == dream_rank]
        if not dream_univ.empty:
            st.info(f"Selected: {dream_univ['institution'].iloc[0]}")
    
    if st.button("Compare Universities"):
        if current_univ.empty or dream_univ.empty:
            st.error("One or both ranks not found in the dataset.")
            return
        
        # Get comparison columns based on ranking system
        if ranking_system == "QS":
            compare_cols = [
                "academic reputation", "employer reputation",
                "faculty student", "citations per faculty",
                "international faculty", "international students",
                "international research network", "employment outcomes",
                "sustainability"
            ]
        else:
            compare_cols = ["tlr", "rpc", "go", "oi", "perception"]
        
        current_name = current_univ["institution"].iloc[0]
        dream_name = dream_univ["institution"].iloc[0]
        
        # Create comparison dataframe
        comparison_data = {
            "Criteria": compare_cols,
            f"{current_name} (Rank {current_rank})": current_univ[compare_cols].values[0],
            f"{dream_name} (Rank {dream_rank})": dream_univ[compare_cols].values[0]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Show comparison visualization
        st.write("### Comparison Analysis")
        
        # Bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        comparison_melted = pd.melt(comparison_df, id_vars="Criteria", var_name="University", value_name="Score")
        sns.barplot(data=comparison_melted, x="Criteria", y="Score", hue="University", ax=ax)
        plt.title(f"Comparison Between {current_name} (Rank {current_rank}) and {dream_name} (Rank {dream_rank})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show numerical differences
        st.write("### Score Differences")
        diff_df = comparison_df.copy()
        diff_df["Difference"] = diff_df[f"{dream_name} (Rank {dream_rank})"] - diff_df[f"{current_name} (Rank {current_rank})"]
        st.dataframe(diff_df.style.format({col: "{:.3f}" for col in diff_df.columns if col != "Criteria"}))

# File Upload Section
st.header("Upload Ranking Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("QS Rankings")
    qs_file = st.file_uploader("Upload QS Rankings CSV", type="csv")
    if qs_file is not None:
        qs_df = pd.read_csv(qs_file)
        qs_results = clean_qs_ranking(qs_df)
        if qs_results:
            qs_df_cleaned, qs_ranking_criteria, qs_weights = qs_results
            st.success("QS Rankings data processed successfully!")
            st.subheader("QS Rankings Analysis")
            visualize_rankings(qs_df_cleaned, qs_ranking_criteria, "QS World Ranking", qs_weights)

with col2:
    st.subheader("NIRF Rankings")
    nirf_file = st.file_uploader("Upload NIRF Rankings CSV", type="csv")
    if nirf_file is not None:
        nirf_df = pd.read_csv(nirf_file)
        nirf_results = clean_nirf_ranking(nirf_df)
        if nirf_results:
            nirf_df_cleaned, nirf_ranking_criteria, nirf_weights = nirf_results
            st.success("NIRF Rankings data processed successfully!")
            st.subheader("NIRF Rankings Analysis")
            visualize_rankings(nirf_df_cleaned, nirf_ranking_criteria, "NIRF Ranking", nirf_weights)

# Comparative Analysis Section
if 'qs_df_cleaned' in locals() and 'nirf_df_cleaned' in locals():
    st.header("Comparative Analysis")
    comparative_analysis(qs_df_cleaned, nirf_df_cleaned)
    
    # Common Universities Analysis
    st.header("Common Universities Analysis")
    
    # Clean university names for better matching
    qs_df_cleaned['clean_name'] = qs_df_cleaned['institution'].apply(clean_university_name)
    nirf_df_cleaned['clean_name'] = nirf_df_cleaned['institution'].apply(clean_university_name)
    
    # Find common universities
    common_unis = set(qs_df_cleaned['clean_name']).intersection(set(nirf_df_cleaned['clean_name']))
    
    if common_unis:
        st.write(f"Found {len(common_unis)} universities present in both QS and NIRF rankings")
        
        # Create comparison dataframe for common universities
        common_data = []
        for uni in common_unis:
            qs_data = qs_df_cleaned[qs_df_cleaned['clean_name'] == uni].iloc[0]
            nirf_data = nirf_df_cleaned[nirf_df_cleaned['clean_name'] == uni].iloc[0]
            
            common_data.append({
                'University': qs_data['institution'],
                'QS Score': qs_data['final_score'],
                'QS Percentile': qs_data['rank_percentile'],
                'NIRF Score': nirf_data['final_score'],
                'NIRF Percentile': nirf_data['rank_percentile']
            })
        
        common_df = pd.DataFrame(common_data)
        common_df = common_df.sort_values(by=['QS Score', 'NIRF Score'], ascending=[False, False])
        
        # Display common universities table
        st.dataframe(common_df.style.format({
            'QS Score': '{:.3f}',
            'QS Percentile': '{:.2f}%',
            'NIRF Score': '{:.3f}',
            'NIRF Percentile': '{:.2f}%'
        }))
        
        # Visualization of common universities
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.scatter(common_df['QS Score'], common_df['NIRF Score'])
        
        # Add university labels to points
        for idx, row in common_df.iterrows():
            plt.annotate(row['University'], (row['QS Score'], row['NIRF Score']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('QS Score')
        plt.ylabel('NIRF Score')
        plt.title('Comparison of QS vs NIRF Scores for Common Universities')
        
        # Add correlation line
        z = np.polyfit(common_df['QS Score'], common_df['NIRF Score'], 1)
        p = np.poly1d(z)
        plt.plot(common_df['QS Score'], p(common_df['QS Score']), "r--", alpha=0.8)
        
        correlation = common_df['QS Score'].corr(common_df['NIRF Score'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No common universities found between QS and NIRF rankings")