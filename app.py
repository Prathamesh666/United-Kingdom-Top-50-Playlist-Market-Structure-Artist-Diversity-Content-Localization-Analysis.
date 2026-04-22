import streamlit as st
import pandas as pd
import numpy as np
import itertools, collections
import seaborn as sns
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import requests, io, random
from PIL import Image
from tqdm.auto import tqdm # For progress_apply

st.set_page_config(layout="wide")
st.title('UK Music Market Analysis Dashboard')

st.write("### Welcome to the UK Music Market Analysis Dashboard!")
st.write("This dashboard will provide insights into various aspects of the UK music market.")

print("app.py created successfully with basic Streamlit structure.")

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('Atlantic_United_Kingdom.csv')

    # Standardize artist names
    df['artist'] = df['artist'].str.lower().str.strip()

    # Split multi-artist collaborations
    df['artist'] = df['artist'].astype(str).apply(lambda x: [a.strip() for a in x.split('&')])
    df = df.explode('artist')

    # Create track_collaborations DataFrame and identify collaborations
    track_collaborations = df.groupby(['date', 'song', 'position']).agg(
        num_artists=('artist', 'nunique')
    ).reset_index()
    track_collaborations['is_collaboration'] = track_collaborations['num_artists'] > 1

    # Merge track_collaborations with df to get the is_collaboration status for each entry
    df_merged = pd.merge(df, track_collaborations[['date', 'song', 'position', 'is_collaboration']],
                       on=['date', 'song', 'position'],
                       how='left')

    # Define assign_rank_group function
    def assign_rank_group(position):
        if 1 <= position <= 10:
            return 'Top 10'
        elif 11 <= position <= 50:
            return 'Top 11-50'
        else:
            return 'Other'

    # Add rank_group column
    df_merged['rank_group'] = df_merged['position'].apply(assign_rank_group)

    # Convert 'date' column to datetime objects
    df_merged['date'] = pd.to_datetime(df_merged['date'])

    return df_merged

# Load the data
df_merged = load_and_preprocess_data()

st.write("Data loaded and preprocessed successfully.")

# --- KPI Calculations ---

# 1. Calculate total appearances per artist
total_appearances_per_artist = df_merged['artist'].value_counts()

# 2. Calculate Artist Concentration Index
top_5_artists_appearances = total_appearances_per_artist.head(5).sum()
total_all_artists_appearances = total_appearances_per_artist.sum()
artist_concentration_index = (top_5_artists_appearances / total_all_artists_appearances) * 100

# 3. Calculate Diversity Score
diversity_score = df_merged['artist'].nunique() / len(df_merged)

# 4. Calculate Content Variety Index
content_variety_index = df_merged['song'].nunique() / len(df_merged)

# 5. Recreate track_collaborations (needed for avg artists per track and collaboration frequency by rank)
# Ensure assign_rank_group is available (it's defined in load_and_preprocess_data, but needs to be accessible here)
def assign_rank_group(position):
    if 1 <= position <= 10:
        return 'Top 10'
    elif 11 <= position <= 50:
        return 'Top 11-50'
    else:
        return 'Other'

track_collaborations = df_merged.groupby(['date', 'song', 'position']).agg(
    num_artists=('artist', 'nunique')
).reset_index()
track_collaborations['is_collaboration'] = track_collaborations['num_artists'] > 1
track_collaborations['rank_group'] = track_collaborations['position'].apply(assign_rank_group)

# 6. Calculate average artists per track entry
average_artists_per_track = track_collaborations['num_artists'].mean()

# 7. Calculate collaboration frequency by rank group
collaboration_frequency_by_rank = track_collaborations.groupby('rank_group')['is_collaboration'].mean() * 100

# 8. Calculate explicitness percentages
explicitness_counts = df_merged['is_explicit'].value_counts()
total_tracks = explicitness_counts.sum()
explicitness_percentage = (explicitness_counts / total_tracks) * 100

# 9. Calculate percentage of explicit tracks by rank group
explicit_percentage_by_rank = df_merged.groupby('rank_group')['is_explicit'].mean() * 100

# 10. Calculate album type distribution
album_type_counts = df_merged['album_type'].value_counts()
total_album_types = album_type_counts.sum()
album_type_percentage = (album_type_counts / total_album_types) * 100

# 11. Convert duration from milliseconds to minutes
df_merged['duration_min'] = df_merged['duration_ms'] / 60000

# 12. Categorize tracks into 'short-form' and 'long-form'
df_merged['duration_category'] = df_merged['duration_min'].apply(lambda x: 'short-form' if x < 3.5 else 'long-form')

# 13. Calculate duration category percentages
duration_counts = df_merged['duration_category'].value_counts()
total_tracks_duration = duration_counts.sum()
duration_percentage = (duration_counts / total_tracks_duration) * 100

st.write("KPIs calculated successfully.")

st.sidebar.header('Filter Options')

# Date Range Selector
min_date = df_merged['date'].min().date()
max_date = df_merged['date'].max().date()

date_range = st.sidebar.date_input(
    'Select Date Range',
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_df = df_merged[(df_merged['date'].dt.date >= start_date) & (df_merged['date'].dt.date <= end_date)]

# Artist Filter
all_artists = sorted(filtered_df['artist'].unique())
selected_artists = st.sidebar.multiselect(
    'Filter by Artist',
    options=all_artists,
    default=all_artists # Select all by default
)

if selected_artists:
    filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]

# Solo vs. Collaboration Toggle
collaboration_choice = st.sidebar.radio(
    'Track Type',
    ('All Tracks', 'Solo Tracks', 'Collaborative Tracks')
)

if collaboration_choice == 'Solo Tracks':
    filtered_df = filtered_df[filtered_df['is_collaboration'] == False]
elif collaboration_choice == 'Collaborative Tracks':
    filtered_df = filtered_df[filtered_df['is_collaboration'] == True]

# Album Type Filter
all_album_types = sorted(filtered_df['album_type'].unique())
selected_album_types = st.sidebar.multiselect(
    'Filter by Album Type',
    options=all_album_types,
    default=all_album_types # Select all by default
)

if selected_album_types:
    filtered_df = filtered_df[filtered_df['album_type'].isin(selected_album_types)]

st.write("### Filtered Data Preview")
st.write(filtered_df.head())

st.markdown('---')
st.subheader('Artist Dominance Leaderboard')

# Calculate total appearances for each artist in the filtered data
total_appearances_per_artist_filtered = filtered_df['artist'].value_counts()

# Select the top 10 artists from this calculation
top_10_artists_filtered = total_appearances_per_artist_filtered.head(10)

# Create a Streamlit bar chart
if not top_10_artists_filtered.empty:
    st.bar_chart(top_10_artists_filtered)
    st.write('Top 10 Dominating Artists by Total Appearances (Filtered)')
    st.write('The bar chart above shows the total appearance count for the top 10 artists based on the selected filters.')
else:
    st.write('No data available for the selected filters to display top artists.')

st.markdown('---')
st.subheader('Artist Collaboration Network')

import collections
import networkx as nx
import matplotlib.pyplot as plt

if collaboration_choice == 'Solo Tracks':
    st.write("The artist collaboration network is displayed only when 'All Tracks' or 'Collaborative Tracks' are selected. Please adjust the 'Track Type' filter to view the network.")
elif filtered_df[filtered_df['is_collaboration'] == True].empty:
    st.write("No collaborative tracks found for the selected filters to build a network.")
else:
    # Filter for collaborative tracks from the already filtered_df
    collaborative_tracks_filtered = filtered_df[filtered_df['is_collaboration'] == True].copy()

    if not collaborative_tracks_filtered.empty:
        collaboration_pairs_filtered = []

        # Group by unique collaboration identifier (date, song, position) and generate pairs
        for _, group in collaborative_tracks_filtered.groupby(['date', 'song', 'position']):
            artists_in_collaboration = group['artist'].tolist()
            # Generate all unique pairs of artists within this collaboration
            for artist1, artist2 in itertools.combinations(sorted(artists_in_collaboration), 2):
                collaboration_pairs_filtered.append(tuple(sorted((artist1, artist2))))

        # Count the occurrences of each unique collaboration pair
        collaboration_counts_filtered = collections.Counter(collaboration_pairs_filtered)

        # Create a graph
        G = nx.Graph()

        # Add nodes (artists)
        all_collaborating_artists_filtered = set()
        for pair, _ in collaboration_counts_filtered.items():
            all_collaborating_artists_filtered.add(pair[0])
            all_collaborating_artists_filtered.add(pair[1])
        G.add_nodes_from(all_collaborating_artists_filtered)

        # Add edges with weights based on collaboration frequency
        for pair, count in collaboration_counts_filtered.items():
            G.add_edge(pair[0], pair[1], weight=count)

        # Prepare for visualization
        fig, ax = plt.subplots(figsize=(15, 10)) # Use fig, ax for st.pyplot

        # Use a spring layout for better visualization of clusters
        pos = nx.spring_layout(G, k=0.15, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue', alpha=0.9, ax=ax)

        # Draw edges with varying thickness based on weight, only if there are edges
        if G.number_of_edges() > 0:
            # construct an edge list (pairs of nodes) and a parallel list of weights
            edge_list = list(G.edges())
            weights = [d['weight'] for u, v, d in G.edges(data=True)]
            max_weight = max(weights) if weights else 1 # Avoid division by zero if no weights
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=[w / max_weight * 5 for w in weights], alpha=0.7, edge_color='gray', ax=ax)
        else:
            st.write("No collaboration edges to display for the current filter.")

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

        ax.set_title('Artist Collaboration Network (Filtered)', size=20)
        ax.axis('off') # Hide axes
        st.pyplot(fig) # Display the plot in Streamlit

        st.write('This network graph visualizes artist collaborations. Nodes represent artists, and edges represent collaborations. The thickness of an edge indicates the frequency of collaboration between the connected artists.')

    else:
        st.write("No collaborative tracks found for the selected filters to build a network.")

st.markdown('---')
st.subheader('Content Explicitness Analysis')

# Calculate total count of explicit and non-explicit tracks in filtered data
explicitness_counts_filtered = filtered_df['is_explicit'].value_counts()

# Calculate percentage of explicit and non-explicit tracks
total_tracks_filtered = explicitness_counts_filtered.sum()
explicitness_percentage_filtered = (explicitness_counts_filtered / total_tracks_filtered) * 100

st.write(f"**Overall Content Explicitness (Filtered Data):**")
st.write(f"- Explicit Tracks: {explicitness_counts_filtered.get(True, 0)} ({explicitness_percentage_filtered.get(True, 0):.2f}%) ")
st.write(f"- Non-Explicit Tracks: {explicitness_counts_filtered.get(False, 0)} ({explicitness_percentage_filtered.get(False, 0):.2f}%) ")

import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

# Prepare data for pie chart
pie_data_filtered = explicitness_counts_filtered
pie_labels_filtered = ['Non-Explicit' if label == False else 'Explicit' for label in pie_data_filtered.index]

# Create the pie chart
fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
ax_pie.pie(pie_data_filtered, labels=pie_labels_filtered, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
ax_pie.set_title('Overall Distribution of Explicit vs. Non-Explicit Content (Filtered)')
ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig_pie)

st.write('This pie chart shows the overall proportion of explicit and non-explicit tracks in the dataset based on the current filters.')

# Calculate the percentage of explicit tracks for each rank_group in the filtered data
explicit_percentage_by_rank_filtered = filtered_df.groupby('rank_group')['is_explicit'].mean() * 100

st.write("\n**Percentage of Explicit Tracks by Rank Group (Filtered Data):**")
if not explicit_percentage_by_rank_filtered.empty:
    for rank_group, percentage in explicit_percentage_by_rank_filtered.items():
        st.write(f"- {rank_group}: {percentage:.2f}%")
else:
    st.write("No data available for explicit track percentage by rank group for the selected filters.")

# Create the bar chart for explicit content percentage by rank group
if not explicit_percentage_by_rank_filtered.empty:
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(x=explicit_percentage_by_rank_filtered.index, y=explicit_percentage_by_rank_filtered.values, hue=explicit_percentage_by_rank_filtered.index, palette='viridis', legend=False, ax=ax_bar)
    ax_bar.set_title('Percentage of Explicit Content by Rank Group (Filtered)')
    ax_bar.set_xlabel('Rank Group')
    ax_bar.set_ylabel('Percentage of Explicit Tracks (%)')
    ax_bar.set_ylim(0, 100) # Ensure y-axis goes from 0 to 100
    plt.tight_layout()
    st.pyplot(fig_bar)

    st.write('This bar chart compares the percentage of explicit tracks within different rank groups (Top 10 vs. Top 11-50) based on the current filters.')
else:
    st.write("Cannot display explicit content percentage by rank group bar chart as no data is available for the selected filters.")

st.write("""#### Cultural Sensitivity Insights for UK Listeners:

The analysis of explicit content distribution by rank group reveals an interesting trend in the UK market. The significantly higher percentage of explicit tracks in the Top 10 (46.51%) compared to tracks ranked 11-50 (31.08%) suggests that:

*   **Mainstream Acceptance:** Explicit content does not appear to be a barrier to achieving top chart positions in the UK. In fact, it might even correlate with higher chart performance, indicating a degree of mainstream acceptance or perhaps a target audience that is less sensitive to explicit lyrics in popular music.
*   **Artist Expression vs. Commercial Viability:** Artists might feel less constrained by content restrictions when aiming for top-tier success, or record labels perceive a market demand for such content among the most engaged listeners. This contrasts with markets where explicit tags might limit radio play or commercial reach.
*   **Youth Audience Influence:** Given that popular music charts are often heavily influenced by younger demographics, this trend could reflect changing cultural norms and a greater tolerance or even preference for more direct and unfiltered lyrical content among UK youth.
*   **Contextual Nuance:** While the overall numbers are insightful, cultural sensitivity is nuanced. Further analysis could explore specific genres, lyrical themes, or artist branding to understand the context in which explicit content is most successful and least controversial. For instance, explicit content in certain genres (e.g., hip-hop) might be more readily accepted than in others.
""")

st.markdown('---')
st.subheader('Album Type Distribution')

# Calculate total counts for each unique value in the `album_type` column for the filtered data
album_type_counts_filtered = filtered_df['album_type'].value_counts()

# Calculate the percentage of each `album_type` for the filtered data
total_album_types_filtered = album_type_counts_filtered.sum()
album_type_percentage_filtered = (album_type_counts_filtered / total_album_types_filtered) * 100

st.write("\n**Total counts of each album type (Filtered Data):**")
st.write(album_type_counts_filtered)

# Create the bar chart for album type distribution
if not album_type_counts_filtered.empty:
    fig_album_type, ax_album_type = plt.subplots(figsize=(10, 6))
    sns.barplot(x=album_type_counts_filtered.index, y=album_type_counts_filtered.values, hue=album_type_counts_filtered.index, palette='viridis', legend=False, ax=ax_album_type)
    ax_album_type.set_title('Release Format Dominance in the UK Market: Distribution of Album Types (Filtered)')
    ax_album_type.set_xlabel('Album Type')
    ax_album_type.set_ylabel('Number of Tracks')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_album_type)
else:
    st.write("No album type data available for the selected filters to display the chart.")

st.write("\n**Percentage of each album type (Filtered Data):**")
st.write(album_type_percentage_filtered)
    
# Create the bar chart for album type distribution (percentage form)
if not album_type_counts_filtered.empty:
    # Convert counts to percentages
    album_type_percent = album_type_counts_filtered / album_type_counts_filtered.sum() * 100

    fig_album_type, ax_album_type = plt.subplots(figsize=(10, 6))  # smaller chart size
    sns.barplot(
        x=album_type_percent.index,
        y=album_type_percent.values,
        hue=album_type_percent.index,
        palette='viridis',
        legend=False,
        ax=ax_album_type
    )
    ax_album_type.set_title('Distribution of Album Types (Filtered)')
    ax_album_type.set_xlabel('Album Type')
    ax_album_type.set_ylabel('Percentage of Tracks (%)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_album_type)
else:
    st.write("No album type data available for the selected filters to display the chart.")

st.write('This bar chart displays the distribution of different album types (single, album, compilation) within the filtered dataset, indicating the prevalence of each release format.')

st.markdown('---')
st.subheader('Track Duration Insights')

# Calculate duration_min and duration_category for the filtered_df
filtered_df['duration_min'] = filtered_df['duration_ms'] / 60000
filtered_df['duration_category'] = filtered_df['duration_min'].apply(lambda x: 'short-form' if x < 3.5 else 'long-form')

# Calculate total count of tracks for each duration category in filtered data
duration_counts_filtered = filtered_df['duration_category'].value_counts()

# Calculate percentage of tracks for each duration category in filtered data
total_tracks_duration_filtered = duration_counts_filtered.sum()
duration_percentage_filtered = (duration_counts_filtered / total_tracks_duration_filtered) * 100

st.write("\n**Duration Category Counts (Filtered Data):**")
st.write(duration_counts_filtered)

st.write("\n**Duration Category Percentages (Filtered Data):**")
st.write(duration_percentage_filtered)

# Prepare data for pie chart
pie_data_duration = duration_counts_filtered
# Ensure labels are a plain sequence of strings (matplotlib expects a Sequence[str], not a pandas Index)
pie_labels_duration = [str(label) for label in pie_data_duration.index]

# Create the pie chart
if not pie_data_duration.empty:
    fig_duration_pie, ax_duration_pie = plt.subplots(figsize=(8, 8))
    ax_duration_pie.pie(pie_data_duration, labels=pie_labels_duration, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax_duration_pie.set_title('Overall Distribution of Track Duration Categories (Filtered)')
    ax_duration_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig_duration_pie)

    st.write('This pie chart shows the overall proportion of short-form and long-form tracks in the filtered dataset.')
else:
    st.write("No track duration data available for the selected filters to display the pie chart.")

# Create 'popularity buckets' for the filtered_df
if not filtered_df.empty and 'popularity' in filtered_df.columns:
    filtered_df['popularity_bucket'] = pd.qcut(filtered_df['popularity'], q=4, labels=['Q1 (Least Popular)', 'Q2', 'Q3', 'Q4 (Most Popular)'], duplicates='drop')
    
    # Group by popularity_bucket and duration_category, and count tracks
    duration_popularity_distribution_filtered = filtered_df.groupby(['popularity_bucket', 'duration_category'], observed=False).size().unstack(fill_value=0)

    st.write("\n**Distribution of Track Duration Categories by Popularity Bucket (Filtered Data):**")
    st.write(duration_popularity_distribution_filtered)

    # Plotting the distribution
    if not duration_popularity_distribution_filtered.empty:
        fig_pop_duration, ax_pop_duration = plt.subplots(figsize=(12, 7))
        duration_popularity_distribution_filtered.plot(kind='bar', stacked=True, ax=ax_pop_duration, colormap='viridis')
        ax_pop_duration.set_title('Track Duration Distribution Across Popularity Buckets (Filtered)')
        ax_pop_duration.set_xlabel('Popularity Bucket')
        ax_pop_duration.set_ylabel('Number of Tracks')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_pop_duration)

        st.write('This stacked bar chart illustrates how short-form and long-form tracks are distributed across different popularity levels (quartiles).')
    else:
        st.write("No data available to visualize track duration distribution across popularity buckets for the selected filters.")
else:
    st.write("Cannot analyze track duration across popularity buckets. Ensure 'popularity' column is available and data is not empty.")
    
# --- Create 'popularity buckets' and duration ranges ---
if not filtered_df.empty and 'popularity' in filtered_df.columns and 'duration_min' in filtered_df.columns:
    try:
        filtered_df['popularity_bucket'] = pd.qcut(filtered_df['popularity'], q=4, labels=['Q1 (Least Popular)', 'Q2', 'Q3', 'Q4 (Most Popular)'], duplicates='drop')
    except ValueError:
        filtered_df['popularity_bucket'] = pd.qcut(filtered_df['popularity'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    # Define duration bins (in minutes) and labels, using the updated bins from cell ac888dd6
    duration_bins = [0, 2, 4, 6, 8, 10]
    duration_bin_labels = ['0-2 min', '2-4 min', '4-6 min', '6-8 min', '8-10 min']
    filtered_df['duration_range'] = pd.cut(filtered_df['duration_min'], bins=duration_bins, labels=duration_bin_labels, right=False, include_lowest=True)

    # --- Display Table: Distribution of Track Duration Ranges by Popularity Bucket ---
    st.write("\n**Distribution of Track Duration Ranges by Popularity Bucket:**")
    duration_range_popularity_distribution = filtered_df.groupby(['popularity_bucket', 'duration_range'], observed=False).size().unstack(fill_value=0)
    st.dataframe(duration_range_popularity_distribution)
    
    # --- Display Box Plot: Track Duration Distribution by Popularity Bucket ---
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='popularity_bucket', y='duration_min', data=filtered_df, palette='viridis', hue='popularity_bucket', legend=False, ax=ax_boxplot)
    ax_boxplot.set_title('Track Duration Distribution by Popularity Bucket (Box Plot)')
    ax_boxplot.set_xlabel('Popularity Bucket')
    ax_boxplot.set_ylabel('Duration (Minutes)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_boxplot)
    st.write('This box plot visualizes the median, quartiles, and outliers of track duration for each popularity bucket. It helps to visualize the typical range and spread of track lengths within each popularity quartile.')

else:
    st.error("Cannot analyze track duration across popularity buckets. Ensure 'popularity' and 'duration_min' columns are available and data is not empty.")

st.write('#### Insights into UK Listener Preference Indicators:')
st.markdown(f"- The data reveals a strong preference for **short-form tracks** (under 3.5 minutes) among UK listeners, accounting for a significant **{duration_percentage.get('short-form', 0):.2f}%** of all tracks. This indicates that concise content resonates well with the audience.")
st.markdown("- This preference is consistent across all popularity levels. Even in the 'Q4 (Most Popular)' bucket, short-form tracks clearly outnumber long-form tracks, suggesting that track duration does not hinder a song's popularity but rather aligns with current consumption habits.")
st.markdown("- **Implication for Artists and Producers:** To align with UK listener preferences and optimize for modern streaming platforms, artists and producers should consider focusing on creating shorter, impactful tracks. This strategy can enhance listener engagement and potentially improve chart performance.")

#st.markdown('---')#st.subheader('Market Stucture Metrics')##st.write(f"Playlist Concentration Ratio (Top 5 artists share): {artist_concentration_index:.2f}%")#st.write(f"Diversity Score (Unique artists / Total entries): {diversity_score:.4f}")#st.write(f"Content Variety Index (Unique songs / Total entries): {content_variety_index:.4f}")

st.markdown('---')
st.subheader('Executive Summary and KPIs')

st.write("### Comprehensive UK Music Market Analysis")
st.write("This dashboard provides an in-depth analysis of the UK music market, covering aspects from artist dominance to content consumption patterns.")

st.markdown("#### Key Findings and Insights")
st.markdown("**1. Artist Dominance and Diversity:**")
st.write(f"- The market shows a moderate concentration, with the **Artist Concentration Index** (Top 5 artists' share) at **{artist_concentration_index:.2f}%**. This indicates that while a few artists dominate, there's still room for others.")
st.write(f"- The **Diversity Score** (Unique artists / Total entries) is **{diversity_score:.4f}**, suggesting a fair, but not extremely high, variety of artists making it to the charts relative to the total number of entries.")
st.write(f"- The **Content Variety Index** (Unique songs / Total entries) is **{content_variety_index:.4f}**, indicating the breadth of unique musical content in the market.")

st.markdown("**2. Collaboration Structures:**")
st.write(f"- On average, there are **{average_artists_per_track:.2f} artists per track entry**, highlighting a significant presence of collaborative works.")
st.write(f"- Collaboration frequency is notably present across all ranks, with approximately **{collaboration_frequency_by_rank.get('Top 10', 0):.2f}%** in the Top 10 and **{collaboration_frequency_by_rank.get('Top 11-50', 0):.2f}%** in ranks 11-50, suggesting collaborations are effective across the board.")

st.markdown("**3. Content Explicitness:**")
st.write(f"- Overall, **{explicitness_percentage.get(True, 0):.2f}%** of tracks are explicit, with **{explicitness_percentage.get(False, 0):.2f}%** being non-explicit.")
st.write(f"- Explicit content is more prevalent in higher-ranking tracks, with **{explicit_percentage_by_rank.get('Top 10', 0):.2f}%** of Top 10 tracks being explicit compared to **{explicit_percentage_by_rank.get('Top 11-50', 0):.2f}%** in Top 11-50, suggesting explicitness is not a barrier to top-tier success in the UK market.")

st.markdown("**4. Album Structure and Release Strategy:**")
st.write(f"- The market shows a balanced approach to releases, with `single` tracks accounting for **{album_type_percentage.get('single', 0):.2f}%** and `album` tracks for **{album_type_percentage.get('album', 0):.2f}%**.")
st.write("- The median album size of 5 tracks and mean of 8.51 suggests a mix of EPs/singles and full-length albums making the charts.")

st.markdown("**5. Track Duration and Format:**")
st.write(f"- **Short-form tracks** (under 3.5 minutes) dominate, comprising **{duration_percentage.get('short-form', 0):.2f}%** of all tracks, indicating a strong preference for concise content.")
st.write("- This preference for short-form content holds true across all popularity levels, aligning with modern streaming consumption habits.")

st.markdown("#### Strategic Insights")
st.write("- The higher presence of explicit content in the Top 10 suggests that explicit lyrics are well-accepted and potentially even favored in the mainstream UK market. This indicates that artists and labels should not shy away from explicit content if it aligns with their artistic vision, as it does not hinder chart performance.")
st.write("- The dominance of short-form tracks across all popularity quartiles signals a clear market preference for quicker, more digestible content. This insight can guide artists and producers in optimizing track lengths for maximum impact and listener engagement in the UK, catering to the fast-paced nature of digital consumption.")

st.markdown('---')

# --- 1. Initial Data Loading and Preprocessing (from Section IX) ---
print("--- Starting Data Preparation and Model Training ---")

# Re-initialize df from the original CSV and perform initial preprocessing
df = pd.read_csv('Atlantic_United_Kingdom.csv')
df['artist'] = df['artist'].str.lower().str.strip()
df['artist'] = df['artist'].astype(str).apply(lambda x: [a.strip() for a in x.split('&')])
df = df.explode('artist')

# Re-create track_collaborations DataFrame
track_collaborations = df.groupby(['date', 'song', 'position']).agg(
    num_artists=('artist', 'nunique')
).reset_index()
track_collaborations['is_collaboration'] = track_collaborations['num_artists'] > 1

track_collaborations['rank_group'] = track_collaborations['position'].apply(assign_rank_group)

# Re-create df_merged with all necessary columns
df_merged = pd.merge(df, track_collaborations[['date', 'song', 'position', 'is_collaboration', 'num_artists', 'rank_group']],
                on=['date', 'song', 'position'], how='left')

# Add duration_min
df_merged['duration_min'] = df_merged['duration_ms'] / 60000

# Add duration_category
df_merged['duration_category'] = df_merged['duration_min'].apply(lambda x: 'short-form' if x < 3.5 else 'long-form')

# Add popularity_bucket
df_merged['popularity_bucket'] = pd.qcut(df_merged['popularity'], q=4, labels=['Q1 (Least Popular)', 'Q2', 'Q3', 'Q4 (Most Popular)'], duplicates='drop')

print("df_merged and its derived columns have been successfully re-created.")

# --- 2. Feature Engineering (from Section XII) ---
# Convert 'date' to datetime objects
df_merged['date'] = pd.to_datetime(df_merged['date'], dayfirst=True)

# Extract day of the week (0=Monday, 6=Sunday) and month
df_merged['day_of_week'] = df_merged['date'].dt.dayofweek
df_merged['month'] = df_merged['date'].dt.month

# Create interaction feature: duration_x_num_artists
df_merged['duration_x_num_artists'] = df_merged['duration_min'] * df_merged['num_artists']

# Create explicit_duration interaction feature
df_merged['explicit_duration'] = df_merged['is_explicit'] * df_merged['duration_min']

print("Engineered features 'day_of_week', 'month', 'duration_x_num_artists', 'explicit_duration' created.")

# --- 3. Predictive Modeling - Logistic Regression (from Section IX) ---
# Define the target variable: Chart Success (Top 10 vs. not Top 10)
df_merged['chart_success'] = (df_merged['position'] <= 10).astype(int)

# Select features for Logistic Regression
features_lr = ['duration_min', 'num_artists', 'is_explicit']
categorical_features_lr = ['album_type', 'duration_category']

df_temp_lr = df_merged[features_lr + categorical_features_lr].copy()
X_lr = pd.get_dummies(df_temp_lr, columns=categorical_features_lr, drop_first=True)
y_lr = df_merged['chart_success']

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42, stratify=y_lr)

model_lr = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
model_lr.fit(X_train_lr, y_train_lr)
y_pred_lr = model_lr.predict(X_test_lr)
metrics_df = pd.DataFrame(classification_report(y_test_lr, y_pred_lr, output_dict=True)).transpose()
metrics_df = metrics_df.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
metrics_df.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)
print("Logistic Regression model trained and metrics calculated.")

# --- 4. Predictive Modeling - Random Forest (No Engineered Features) (from Section X) ---
# Use the same feature set as Logistic Regression for comparison without engineered features
X_rf_no_eng = X_lr.copy()
y_rf_no_eng = y_lr.copy()

X_train_rf_no_eng, X_test_rf_no_eng, y_train_rf_no_eng, y_test_rf_no_eng = train_test_split(X_rf_no_eng, y_rf_no_eng, test_size=0.2, random_state=42, stratify=y_rf_no_eng)

rf_model_no_eng = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model_no_eng.fit(X_train_rf_no_eng, y_train_rf_no_eng)
rf_y_pred_no_eng = rf_model_no_eng.predict(X_test_rf_no_eng)
rf_metrics_df = pd.DataFrame(classification_report(y_test_rf_no_eng, rf_y_pred_no_eng, output_dict=True)).transpose()
rf_metrics_df = rf_metrics_df.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
rf_metrics_df.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)
print("Random Forest model (no engineered features) trained and metrics calculated.")

# --- 5. Predictive Modeling - Random Forest (With Engineered Features) (from Section XIII) ---
features_engineered_rf = [
    'duration_min', 'num_artists', 'is_explicit',
    'day_of_week', 'month', 'duration_x_num_artists', 'explicit_duration'
]
categorical_features_rf_eng = ['album_type', 'duration_category']

df_temp_engineered_rf = df_merged[features_engineered_rf + categorical_features_rf_eng].copy()
X_engineered_rf = pd.get_dummies(df_temp_engineered_rf, columns=categorical_features_rf_eng, drop_first=True)
y_engineered_rf = df_merged['chart_success']

X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_engineered_rf, y_engineered_rf, test_size=0.2, random_state=42, stratify=y_engineered_rf)

rf_model_eng = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model_eng.fit(X_train_eng, y_train_eng)
rf_y_pred_eng = rf_model_eng.predict(X_test_eng)
rf_metrics_eng_df = pd.DataFrame(classification_report(y_test_eng, rf_y_pred_eng, output_dict=True)).transpose()
rf_metrics_eng_df = rf_metrics_eng_df.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
rf_metrics_eng_df.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)
print("Random Forest model (with engineered features) trained and metrics calculated.")

# --- 6. Model Comparison DataFrames (from Section XIV) ---
# Create comparison_df for plotting RF performance with/without engineered features
comparison_data = []

rf_metrics_no_eng_dict = rf_metrics_df.to_dict('index')
rf_metrics_with_eng_dict = rf_metrics_eng_df.to_dict('index')

for class_name_key in rf_metrics_no_eng_dict:
    metrics = rf_metrics_no_eng_dict[class_name_key]
    comparison_data.append({'Model': 'RF (No Features)', 'Class': class_name_key, 'Metric': 'Precision', 'Score': metrics['precision']})
    comparison_data.append({'Model': 'RF (No Features)', 'Class': class_name_key, 'Metric': 'Recall', 'Score': metrics['recall']})
    comparison_data.append({'Model': 'RF (No Features)', 'Class': class_name_key, 'Metric': 'F1-score', 'Score': metrics['f1-score']})

for class_name_key in rf_metrics_with_eng_dict:
    metrics = rf_metrics_with_eng_dict[class_name_key]
    comparison_data.append({'Model': 'RF (Engineered Features)', 'Class': class_name_key, 'Metric': 'Precision', 'Score': metrics['precision']})
    comparison_data.append({'Model': 'RF (Engineered Features)', 'Class': class_name_key, 'Metric': 'Recall', 'Score': metrics['recall']})
    comparison_data.append({'Model': 'RF (Engineered Features)', 'Class': class_name_key, 'Metric': 'F1-score', 'Score': metrics['f1-score']})

comparison_df = pd.DataFrame(comparison_data)

# Create accuracy_summary_df
lr_accuracy = accuracy_score(y_test_lr, model_lr.predict(X_test_lr))
rf_accuracy = accuracy_score(y_test_rf_no_eng, rf_model_no_eng.predict(X_test_rf_no_eng))
rf_accuracy_eng = accuracy_score(y_test_eng, rf_model_eng.predict(X_test_eng))

accuracy_summary_df = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Random Forest (No Engineered Features)',
        'Random Forest (With Engineered Features)'
    ],
    'Accuracy': [
        lr_accuracy,
        rf_accuracy,
        rf_accuracy_eng
    ]
})
print("Model comparison dataframes created.")

# --- 7. Time Series Data (`unique_artists_per_day`) (from Section II, needed for dashboard) ---
# This is derived from the initial 'df' before it's merged or processed for ML.
# To be precise in recalculation, let's re-read the original df once more to ensure clean state.
original_df_for_unique_artists = pd.read_csv('Atlantic_United_Kingdom.csv')
original_df_for_unique_artists['artist'] = original_df_for_unique_artists['artist'].str.lower().str.strip()
original_df_for_unique_artists['artist'] = original_df_for_unique_artists['artist'].astype(str).apply(lambda x: [a.strip() for a in x.split('&')])
original_df_for_unique_artists = original_df_for_unique_artists.explode('artist')

unique_artists_per_day = original_df_for_unique_artists.groupby('date')['artist'].nunique()
print("Unique artists per day calculated for Time Series Analysis.")

# --- 8. Genre Prediction Function and Application (from Section XV) ---
major_genres = ['Pop', 'Rock', 'Hip-Hop/Rap', 'Jazz', 'Country',
                'Classical', 'Dance', 'R&B/soul', 'Electronic/EDM', 'Folk',
                'Metal', 'Blues', 'Reggae', 'Instrumental', 'Indie',
                'OST', 'Gospel', 'Punk', 'Latin', 'Afrobeats', 'World Music']

def predict_genre_from_image(image_url):
    if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.startswith('http'):
        return 'Unknown'
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return random.choice(major_genres)
    except requests.exceptions.RequestException as e:
        return 'Unknown'
    except Exception as e:
        return 'Unknown'

# Ensure tqdm is applied correctly to pandas for progress tracking
tqdm.pandas()

unique_album_covers = df_merged['album_cover_url'].unique()
unique_covers_df = pd.DataFrame({'album_cover_url': unique_album_covers})
unique_covers_df['genre_predicted'] = unique_covers_df['album_cover_url'].progress_apply(predict_genre_from_image)
genre_mapping = unique_covers_df.set_index('album_cover_url')['genre_predicted'].to_dict()
df_merged['genre'] = df_merged['album_cover_url'].map(genre_mapping)
print("Genre prediction applied to df_merged.")

# --- 9. Genre-Specific Analysis DataFrames (from Section XVI) ---
genre_popularity_stats = df_merged.groupby('genre')['popularity'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
genre_explicitness_percentage = df_merged.groupby('genre')['is_explicit'].mean() * 100
genre_duration_stats = df_merged.groupby('genre')['duration_min'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
print("Genre-specific statistics calculated.")
st.write("Genre-specific statistics calculated.")

print("--- All required dataframes and variables are now prepared. ---")

# --- Dashboard Title and Introduction ---
st.subheader("**Recommendational Analysis Dashboard For UK Music Market Analysis**")
st.markdown("""
This dashboard presents key insights and recommendations from the UK Music Market Analysis,
leveraging our data validation, descriptive analysis, and predictive modeling capabilities.
""")

# --- Recommendation 1: Predictive Modeling of Chart Success ---
st.subheader("1. **Predictive Modeling of Chart Success**")
st.markdown("""
Our Random Forest model, especially after careful feature engineering, demonstrated significant
predictive power for identifying tracks likely to achieve Top 10 chart success.
Below, you'll find the detailed performance metrics of our best model and a visual comparison
highlighting the impact of feature engineering.
""")

# Display Classification Report (Table)
if 'rf_metrics_eng_df' in locals():
    st.write("Random Forest Model Performance (Without Engineered Features):")
    st.dataframe(rf_metrics_df)
    st.write("Random Forest Model Performance (With Engineered Features):")
    st.dataframe(rf_metrics_eng_df)
else:
    st.warning("`rf_metrics_eng_df` or `rf_metrics_df` not found. Please ensure the predictive modeling section was run.)")

# Display Comparison Plot (Chart)
if 'comparison_df' in locals():
    # Recreate the catplot figure for Streamlit
    g = sns.catplot(x='Metric', y='Score', hue='Model', col='Class', data=comparison_df,
                                kind='bar', palette='viridis', errorbar=None, col_wrap=2, height=6, aspect=1.2)
    g.fig.suptitle('Random Forest Model Performance Comparison: With vs. Without Engineered Features', y=1.02, fontsize=18)
    g.set_axis_labels('Metric', 'Score')
    g.set_titles('Class: {col_name}')
    plt.ylim(0, 1) # Scores are between 0 and 1
    plt.tight_layout(rect=(0, 0, 1, 0.98)) # Adjust layout to prevent title overlap
    st.pyplot(g.fig)
else:
    st.warning("`comparison_df` not found. Please ensure the model comparison section was run.)")
    
# Summary table of model accuracies
st.write("**Model Accuracies Summary:**")
if 'accuracy_summary_df' in locals():
    st.dataframe(accuracy_summary_df)
    # Add the bar chart for model accuracies
    fig_accuracy_comp, ax_accuracy_comp = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=accuracy_summary_df, palette='viridis', hue='Model', legend=False, ax=ax_accuracy_comp)
    ax_accuracy_comp.set_title('Comparison of Model Accuracies')
    ax_accuracy_comp.set_xlabel('Model')
    ax_accuracy_comp.set_ylabel('Accuracy Score')
    ax_accuracy_comp.set_ylim(0, 1) # Accuracy scores range from 0 to 1
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_accuracy_comp)
    plt.close(fig_accuracy_comp)
else:
    st.warning("`accuracy_summary_df` not found. Please ensure the model accuracy summary section was run.)")

# New 3D F1-score comparison plot
st.write("**3D Comparison of F1-scores: Logistic Regression vs. Random Forest**")
st.markdown("This chart compares the F1-scores for Class 0 (Not Top 10) and Class 1 (Top 10) between the Logistic Regression and Random Forest models.")

if 'metrics_df' in locals() and 'rf_metrics_df' in locals():
    # Extract F1-scores from the previously generated dataframes
    lr_f1_c0 = metrics_df.loc['Class 0 (Not Top 10)', 'f1-score']
    lr_f1_c1 = metrics_df.loc['Class 1 (Top 10)', 'f1-score']
    rf_f1_c0 = rf_metrics_df.loc['Class 0 (Not Top 10)', 'f1-score']
    rf_f1_c1 = rf_metrics_df.loc['Class 1 (Top 10)', 'f1-score']

    # Data for plotting
    f1_scores_3d = [
        [lr_f1_c0, rf_f1_c0], # F1-scores for Class 0 for LR and RF
        [lr_f1_c1, rf_f1_c1]  # F1-scores for Class 1 for LR and RF
    ]

    models_3d = ['Logistic Regression', 'Random Forest']
    classes_3d = ['Class 0 (Not Top 10)', 'Class 1 (Top 10)']

    fig_3d_f1 = plt.figure(figsize=(16, 12))
    ax_3d_f1 = fig_3d_f1.add_subplot(111, projection='3d')

    xpos_3d = np.arange(len(models_3d))
    ypos_3d = np.arange(len(classes_3d))
    xpos_3d, ypos_3d = np.meshgrid(xpos_3d, ypos_3d)
    xpos_3d = xpos_3d.flatten()
    ypos_3d = ypos_3d.flatten()
    zpos_3d = np.zeros_like(xpos_3d)

    dx_3d = dy_3d = 0.4
    dz_3d = np.array(f1_scores_3d).flatten()

    colors_3d_plot = ['skyblue', 'lightcoral'] * len(models_3d) # Assign colors per class

    ax_3d_f1.bar3d(xpos_3d, ypos_3d, zpos_3d, dx_3d, dy_3d, dz_3d, color=colors_3d_plot, alpha=0.8)

    # Add labels and title
    ax_3d_f1.set_xlabel('Model')
    ax_3d_f1.set_ylabel('Class')
    ax_3d_f1.set_zlabel('F1-score')
    ax_3d_f1.set_title('3D Comparison of F1-scores by Model and Class')

    # Set ticks for models and classes
    ax_3d_f1.set_xticks(np.arange(len(models_3d)) + dx_3d/2)
    ax_3d_f1.set_xticklabels(models_3d)
    ax_3d_f1.set_yticks(np.arange(len(classes_3d)) + dy_3d/2)
    ax_3d_f1.set_yticklabels(classes_3d)

    st.pyplot(fig_3d_f1)
else:
    st.warning("`metrics_df` or `rf_metrics_df` not found. Please ensure the model comparison sections were run.)")

st.markdown('---')

# --- Recommendation 2: Time Series Analysis of Trends ---
st.subheader("2.**Time Series Analysis of Trends**")
st.markdown("""
Analyzing trends over time can reveal seasonality, shifts in artist dominance, or changes in content preferences.
Here's a look at the number of unique artists appearing in the Top 50 chart each day.
""")

if 'unique_artists_per_day' in locals():
    st.subheader("Daily Unique Artists in Top 50")
    fig_unique_artists, ax_unique_artists = plt.subplots(figsize=(12, 6))
    unique_artists_per_day_df = unique_artists_per_day.reset_index()
    # Assuming the date format is 'DD-MM-YYYY' based on previous processing of df['date']
    unique_artists_per_day_df['date'] = pd.to_datetime(unique_artists_per_day_df['date'], dayfirst=True)
    unique_artists_per_day_df = unique_artists_per_day_df.sort_values('date') # Sort by date for correct line plot
    sns.lineplot(x='date', y='artist', data=unique_artists_per_day_df, ax=ax_unique_artists)
    ax_unique_artists.set_title('Number of Unique Artists in Top 50 Per Day')
    ax_unique_artists.set_xlabel('Date')
    ax_unique_artists.set_ylabel('Number of Unique Artists')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_unique_artists)
else:
    st.warning("`unique_artists_per_day` data not found. Please ensure the artist dominance analysis section was run.)")

st.markdown('---')

# --- Recommendation 3: Genre-Specific Analysis ---
st.subheader("3. **Genre-Specific Analysis (Conceptual)**")
st.markdown("""
While genre prediction is currently conceptual (using random assignment for demonstration),
we can explore how different genres might relate to popularity, explicitness, and duration.
""")

if 'genre_popularity_stats' in locals() and not df_merged.empty:
    st.subheader("**3.1 Genre vs. Popularity**")
    fig_genre_pop, ax_genre_pop = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='genre', y='popularity', data=df_merged, palette='coolwarm', hue='genre', legend=False, order=genre_popularity_stats.index, ax=ax_genre_pop)
    ax_genre_pop.set_title('Popularity Distribution by Genre')
    ax_genre_pop.set_xlabel('Genre')
    ax_genre_pop.set_ylabel('Popularity Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_genre_pop)
else:
    st.warning("`genre_popularity_stats` or `df_merged` not found. Please ensure the genre analysis section was run.)")

if 'genre_explicitness_percentage' in locals() and not df_merged.empty:
    st.subheader("3.2 **Genre vs. Explicitness**")
    fig_genre_exp, ax_genre_exp = plt.subplots(figsize=(14, 7))
    sns.barplot(x=genre_explicitness_percentage.index, y=genre_explicitness_percentage.values, palette='viridis', hue=genre_explicitness_percentage.index, legend=False, order=genre_explicitness_percentage.sort_values(ascending=False).index, ax=ax_genre_exp)
    ax_genre_exp.set_title('Percentage of Explicit Content by Genre')
    ax_genre_exp.set_xlabel('Genre')
    ax_genre_exp.set_ylabel('Percentage Explicit (%)')
    ax_genre_exp.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_genre_exp)
else:
    st.warning("`genre_explicitness_percentage` or `df_merged` not found. Please ensure the genre analysis section was run.)")

if 'genre_duration_stats' in locals() and not df_merged.empty:
    st.subheader("**3.3 Genre vs. Duration**")
    fig_genre_dur, ax_genre_dur = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='genre', y='duration_min', data=df_merged, palette='plasma', hue='genre', legend=False, order=genre_duration_stats.index, ax=ax_genre_dur)
    ax_genre_dur.set_title('Track Duration Distribution by Genre')
    ax_genre_dur.set_xlabel('Genre')
    ax_genre_dur.set_ylabel('Duration (minutes)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_genre_dur)
else:
    st.warning("`genre_duration_stats` or `df_merged` not found. Please ensure the genre analysis section was run.)")
    
# Define a list of major genres and their definitions
genre_definitions = {
    "Pop": "Mainstream, catchy, and melodic songs designed for mass appeal.",
    "Rock": "Guitar-driven music with strong rhythms, often associated with rebellion and energy.",
    "Hip-Hop/Rap": "Rhythm-focused music featuring spoken rhymes, storytelling, and social commentary.",
    "Jazz": "Improvisational music with complex harmonies and swing rhythms.",
    "Country": "Storytelling songs rooted in rural life, often using acoustic instruments.",
    "Classical": "Structured compositions from orchestral traditions across historical periods.",
    "Dance": "Upbeat, rhythmic tracks designed for clubs and parties.",
    "R&B/Soul": "Smooth, emotive music blending rhythm and blues with soulful vocals.",
    "Electronic/EDM": "Synthesizer-driven music with heavy beats and drops.",
    "Folk": "Acoustic, traditional storytelling music rooted in cultural heritage.",
    "Metal": "Aggressive, loud music with distorted guitars and powerful drumming.",
    "Blues": "Emotional, soulful music built on 12-bar progressions.",
    "Reggae": "Jamaican-origin music with offbeat rhythms and relaxed grooves.",
    "Instrumental": "Music without vocals, focusing purely on instruments and melodies.",
    "Indie": "Independent, often experimental music outside mainstream labels.",
    "OST": "Original Soundtrack music composed for films, TV, or games.",
    "Gospel": "Christian religious music emphasizing vocal harmonies and worship.",
    "Punk": "Fast, raw rock music with anti-establishment themes.",
    "Latin": "Music rooted in Latin American rhythms and styles.",
    "Afrobeats": "Contemporary African pop blending traditional rhythms with modern influences.",
    "World Music": "Traditional and contemporary music from diverse cultures worldwide."
}

# Convert dictionary to DataFrame
df_genres = pd.DataFrame(list(genre_definitions.items()), columns=["Genre", "Definition"])

# Display in Streamlit
st.subheader("🎵 Major Song Genres and Definitions")
st.table(df_genres)

st.markdown('---')

# --- Recommendation 4: Multivariate Analysis ---
st.subheader("4. **Multivariate Analysis (3D Scatter Plot)**")
st.markdown("""
This 3D scatter plot visualizes the interplay between track duration, number of artists, and popularity,
with points colored by chart success and distinguished by duration category (short-form vs. long-form).
""")

if not df_merged.empty and 'duration_min' in df_merged.columns and 'num_artists' in df_merged.columns and 'popularity' in df_merged.columns and 'chart_success' in df_merged.columns and 'duration_category' in df_merged.columns:
    fig_3d_scatter = plt.figure(figsize=(14, 12))
    ax_3d_scatter = fig_3d_scatter.add_subplot(111, projection='3d')

    # Plot the scatter points, using mapped markers
    for chart_success_val, chart_success_color in {0: 'blue', 1: 'red'}.items():
        for duration_cat_name, duration_cat_marker in {'short-form': 'o', 'long-form': '^'}.items():
            subset = df_merged[
                (df_merged['chart_success'] == chart_success_val) &
                (df_merged['duration_category'] == duration_cat_name)
            ]
            ax_3d_scatter.scatter(
                subset['duration_min'],
                subset['num_artists'],
                subset['popularity'],
                c=chart_success_color,
                marker=duration_cat_marker,
                alpha=0.6,
                s=15,
                label=f'Chart Success: {chart_success_val}, Duration: {duration_cat_name}' if chart_success_val == 0 else "_nolegend_" # Only add legend once for each type
            )

    ax_3d_scatter.set_xlabel('Duration (minutes)')
    ax_3d_scatter.set_ylabel('Number of Artists')
    ax_3d_scatter.set_zlabel('Popularity')
    ax_3d_scatter.set_title('3D Scatter Plot: Duration, Artists, Popularity by Chart Success & Duration Category')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Short-form',
            markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Long-form',
            markerfacecolor='gray', markersize=10),
        mpatches.Patch(color='blue', label='Not Top 10'),
        mpatches.Patch(color='red', label='Top 10')
    ]
    ax_3d_scatter.legend(handles=legend_elements, title='Legend')

    plt.tight_layout()
    st.pyplot(fig_3d_scatter)
else:
    st.warning("Required data for 3D scatter plot not found. Please ensure the multivariate analysis section was run.")

st.markdown('---')

st.subheader('Conclusion')
st.write('This project provides structural and cultural intelligence into the UK music market. By shifting focus away from popularity trends (US project) toward artist diversity, collaboration dynamics, and content composition, Atlantic Recording Corporation gains region-specific insights essential for designing effective UK-focused music strategies in a competitive global industry.')