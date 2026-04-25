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
import torch, requests, io
from PIL import Image
from tqdm.auto import tqdm # For progress_apply

st.set_page_config(layout="wide")
st.title('United Kingdom Music Market Analysis Dashboard')

st.write("### Welcome to the UK Music Market Analysis Dashboard!")
st.write("This dashboard will provide insights into various aspects of the UK music market.")

print("Dashboard created successfully with basic Streamlit structure. Please wait for some time (about 3-4 mins) to get the advanced Streamlit structure with recommendations & conclusion.")

def assign_rank_group(position):
    if 1 <= position <= 10:
        return 'Top 10'
    elif 11 <= position <= 50:
        return 'Top 11-50'
    else:
        return 'Other'

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
                        on=['date', 'song', 'position'], how='left')

    # Add rank_group column
    df_merged['rank_group'] = df_merged['position'].apply(assign_rank_group)

    # Convert 'date' column to datetime objects
    df_merged['date'] = pd.to_datetime(df_merged['date'], dayfirst=True)

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

explicit_top10 = explicit_percentage_by_rank_filtered.get('Top 10', 0.0)
explicit_11_50 = explicit_percentage_by_rank_filtered.get('Top 11-50', 0.0)
explicit_other = explicit_percentage_by_rank_filtered.get('Other', None)
explicit_difference = explicit_top10 - explicit_11_50

st.write('#### Cultural Sensitivity Insights for UK Listeners:')
if total_tracks_filtered == 0:
    st.write('No explicitness data is available for the current filter selection, so insights cannot be generated at this time.')
else:
    st.markdown(f"- In the current filtered dataset, **{explicit_top10:.2f}%** of Top 10 tracks are explicit while **{explicit_11_50:.2f}%** of Top 11-50 tracks are explicit.")
    if explicit_top10 > explicit_11_50:
        st.markdown(f"- Explicit content is currently more prevalent in the Top 10 by **{explicit_difference:.2f} percentage points**, suggesting top-ranked UK chart entries may have a higher tolerance for direct or mature lyrical themes.")
        st.markdown("- This higher explicitness in top ranks may reflect audience preferences for authentic, unfiltered expressions in popular music.")
    elif explicit_top10 < explicit_11_50:
        st.markdown(f"- Explicit content is currently more prevalent in ranks 11-50 by **{abs(explicit_difference):.2f} percentage points**, which may indicate that the highest-ranked tracks are slightly less explicit in the selected sample.")
        st.markdown("- Lower explicitness in top ranks could suggest mainstream appeal prioritizing broader accessibility over edgy content.")
    else:
        st.markdown("- The explicitness rate is very similar between Top 10 and Top 11-50 tracks, suggesting a stable cultural acceptance across those chart tiers.")
        st.markdown("- Consistent explicitness levels indicate balanced cultural norms in UK music consumption.")

    st.markdown("- These insights are based on the currently selected filters and may shift as the dataset changes. Genre, artist profile, and release strategy all affect how explicit content performs in the UK market.")
    st.markdown("- For a more granular picture, examine explicitness by genre or by artist cohort, since some musical styles and fanbases are more accepting of explicit material than others.")
    if explicit_other is not None:
        st.markdown(f"- Tracks outside the Top 50 currently show **{explicit_other:.2f}%** explicit content, which can highlight longer-tail differences in cultural sensitivity beyond the main chart.")
        st.markdown("- Higher explicitness in lower ranks may signal niche markets or emerging trends in UK listener demographics.")

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
    # Use constrained_layout=True to avoid tight_layout warnings
    fig_album_type, ax_album_type = plt.subplots(figsize=(10, 6), constrained_layout=True)

    sns.barplot(
        x=album_type_counts_filtered.index, y=album_type_counts_filtered.values,
        hue=album_type_counts_filtered.index, palette='viridis', legend=False, ax=ax_album_type
    )
    ax_album_type.set_title('Release Format Dominance in the UK Market: Distribution of Album Types (Filtered)')
    ax_album_type.set_xlabel('Album Type')
    ax_album_type.set_ylabel('Number of Tracks')
    plt.xticks(rotation=45, ha='right')
    # Render and close cleanly
    st.pyplot(fig_album_type)
    plt.close(fig_album_type)

else:
    st.write("No album type data available for the selected filters to display the chart.")

st.write("\n**Percentage of each album type (Filtered Data):**")
st.write(album_type_percentage_filtered)
    
# Create the bar chart for album type distribution (percentage form)
if not album_type_counts_filtered.empty:
    # Convert counts to percentages
    album_type_percent = album_type_counts_filtered / album_type_counts_filtered.sum() * 100

    # Use constrained_layout=True to avoid tight_layout warnings
    fig_album_type, ax_album_type = plt.subplots(figsize=(10, 6), constrained_layout=True)

    sns.barplot(
        x=album_type_percent.index, y=album_type_percent.values,
        hue=album_type_percent.index, palette='viridis', legend=False, ax=ax_album_type
    )
    ax_album_type.set_title('Distribution of Album Types (Filtered)')
    ax_album_type.set_xlabel('Album Type')
    ax_album_type.set_ylabel('Percentage of Tracks (%)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    # Render and close cleanly
    st.pyplot(fig_album_type)
    plt.close(fig_album_type)

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
    # Use constrained_layout=True to avoid tight_layout warnings
    fig_duration_pie, ax_duration_pie = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax_duration_pie.pie( pie_data_duration, labels=pie_labels_duration, autopct='%1.1f%%', startangle=90, 
                        colors=sns.color_palette('pastel')
    )
    ax_duration_pie.set_title('Overall Distribution of Track Duration Categories (Filtered)')
    ax_duration_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig_duration_pie)
    plt.close(fig_duration_pie)

    st.write('This pie chart shows the overall proportion of short-form and long-form tracks in the filtered dataset.')
else:
    st.write("No track duration data available for the selected filters to display the pie chart.")

# Create 'popularity buckets' for the filtered_df
if not filtered_df.empty and 'popularity' in filtered_df.columns:
    filtered_df['popularity_bucket'] = pd.qcut( filtered_df['popularity'], q=4, 
                                                labels=['Q1 (Least Popular)', 'Q2', 'Q3', 'Q4 (Most Popular)'], duplicates='drop'
    )
    
    # Group by popularity_bucket and duration_category, and count tracks
    duration_popularity_distribution_filtered = (
        filtered_df.groupby(['popularity_bucket', 'duration_category'], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    st.write("\n**Distribution of Track Duration Categories by Popularity Bucket (Filtered Data):**")
    st.write(duration_popularity_distribution_filtered)

    # Plotting the distribution
    if not duration_popularity_distribution_filtered.empty:
        fig_pop_duration, ax_pop_duration = plt.subplots(figsize=(12, 7), constrained_layout=True)
        duration_popularity_distribution_filtered.plot(
            kind='bar', stacked=True, ax=ax_pop_duration, colormap='viridis'
        )
        ax_pop_duration.set_title('Track Duration Distribution Across Popularity Buckets (Filtered)')
        ax_pop_duration.set_xlabel('Popularity Bucket')
        ax_pop_duration.set_ylabel('Number of Tracks')
        plt.xticks(rotation=45, ha='right')

        st.pyplot(fig_pop_duration)
        plt.close(fig_pop_duration)

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

overall_short_form_pct = duration_percentage.get('short-form', 0.0)
overall_long_form_pct = duration_percentage.get('long-form', 0.0)

filtered_short_form_pct = duration_percentage_filtered.get('short-form', 0.0)
filtered_long_form_pct = duration_percentage_filtered.get('long-form', 0.0)

short_form_delta = filtered_short_form_pct - overall_short_form_pct
long_form_delta = filtered_long_form_pct - overall_long_form_pct

if total_tracks_duration_filtered == 0:
    st.write('No duration data is available for the current filter selection, so insights cannot be generated at this time.')
else:
    st.markdown(f"- In the current filtered selection, **{filtered_short_form_pct:.2f}%** are short-form and **{filtered_long_form_pct:.2f}%** are long-form.")
    
    full_min_date = df_merged['date'].min().date()
    full_max_date = df_merged['date'].max().date()
    is_date_range_different = (start_date != full_min_date) or (end_date != full_max_date)
    
    if is_date_range_different:
        st.markdown(f"- Across the full dataset, **{overall_short_form_pct:.2f}%** of tracks are short-form and **{overall_long_form_pct:.2f}%** are long-form.")
        
        if short_form_delta > 0:
            st.markdown(f"- The filtered selection shows a stronger short-form bias than the full-date-range baseline by **{short_form_delta:.2f} percentage points**, suggesting the current filter captures an audience or chart segment that prefers shorter tracks.")
            st.markdown("- This suggests evolving listener preferences toward concise content, possibly due to shorter attention spans in digital consumption.")
        elif short_form_delta < 0:
            st.markdown(f"- The filtered selection is slightly less dominated by short-form tracks than the full dataset by **{abs(short_form_delta):.2f} percentage points**, indicating the current filters favor comparatively longer material.")
            st.markdown("- This indicates a niche for longer-form music in the selected period, potentially appealing to dedicated listeners seeking immersive experiences.")
        else:
            st.markdown("- The short-form share in the filtered selection matches the overall dataset, indicating the filter does not materially change the duration preference profile.")
            st.markdown("- This reflects stable duration preferences across the timeframe, suggesting consistent UK listener habits.")
        
        if filtered_short_form_pct >= filtered_long_form_pct:
            st.markdown("- Short-form tracks remain dominant in the filtered view, reinforcing the broader UK preference for concise content under the selected settings.")
        else:
            st.markdown("- Long-form tracks are more prominent in the filtered view than in the overall dataset, which may reflect a niche listener preference or a specific subset of tracks that are more popular under the selected filters.")
        
        avg_duration_overall = df_merged['duration_min'].mean()
        avg_duration_filtered = filtered_df['duration_min'].mean() if not filtered_df.empty else 0.0
        duration_delta = avg_duration_filtered - avg_duration_overall
        
        st.markdown(f"- The full dataset average duration is **{avg_duration_overall:.2f} minutes**, while the filtered average is **{avg_duration_filtered:.2f} minutes**.")
        if duration_delta > 0:
            st.markdown(f"- This means the current filtered selection is slightly longer on average by **{duration_delta:.2f} minutes**, indicating a relative move toward longer-form listening within the selected subset.")
            st.markdown("- This shift toward longer tracks may indicate a trend in the selected period toward more elaborate musical compositions.")
        elif duration_delta < 0:
            st.markdown(f"- This means the current filtered selection is slightly shorter on average by **{abs(duration_delta):.2f} minutes**, reinforcing the short-form preference in the selected subset.")
            st.markdown("- This reinforces the dominance of short-form content, likely driven by streaming platform algorithms favoring quick engagement.")
        else:
            st.markdown("- Average track duration in the filtered selection is the same as the full dataset, suggesting the filter preserves the general length distribution.")
            st.markdown("- This consistency suggests that duration preferences are stable across the UK market over time.")
        
        st.markdown("- These comparisons make the dashboard insights dynamic: they now show whether the current filter view is more or less short-form than the underlying UK market baseline.")
    else:
        if filtered_short_form_pct >= filtered_long_form_pct:
            st.markdown("- Short-form tracks are dominant in the current selection, aligning with typical UK listener preferences for concise content.")
        else:
            st.markdown("- Long-form tracks are more prominent in the current selection, indicating a preference for longer musical experiences.")
        st.markdown(f"- The average track duration in the selection is **{filtered_df['duration_min'].mean() if not filtered_df.empty else 0.0:.2f} minutes**.")
        st.markdown("- Since the full date range is selected, no comparison is needed; this reflects the baseline UK market duration preferences.")

#st.markdown('---')#st.subheader('Market Stucture Metrics')##st.write(f"Playlist Concentration Ratio (Top 5 artists share): {artist_concentration_index:.2f}%")#st.write(f"Diversity Score (Unique artists / Total entries): {diversity_score:.4f}")#st.write(f"Content Variety Index (Unique songs / Total entries): {content_variety_index:.4f}")

st.markdown('---')
st.subheader('Executive Summary and KPIs')

st.write("#### Comprehensive UK Music Market Analysis")
st.write("This dashboard analyzes the UK music market by comparing the current filter view against the full dataset baseline to show how the selected subset differs from the overall market.")

filtered_total_appearances = filtered_df['artist'].value_counts()
filtered_top_5_appearances = filtered_total_appearances.head(5).sum()
filtered_total_all_appearances = filtered_total_appearances.sum()
filtered_artist_concentration_index = (filtered_top_5_appearances / filtered_total_all_appearances) * 100 if filtered_total_all_appearances else 0
filtered_diversity_score = (filtered_df['artist'].nunique() / len(filtered_df)) if len(filtered_df) else 0
filtered_content_variety_index = (filtered_df['song'].nunique() / len(filtered_df)) if len(filtered_df) else 0
filtered_track_collaborations = filtered_df.groupby(['date', 'song', 'position']).agg(num_artists=('artist', 'nunique')).reset_index()
filtered_track_collaborations['is_collaboration'] = filtered_track_collaborations['num_artists'] > 1
filtered_track_collaborations['rank_group'] = filtered_track_collaborations['position'].apply(assign_rank_group)
filtered_collaboration_frequency_by_rank = filtered_track_collaborations.groupby('rank_group')['is_collaboration'].mean() * 100
filtered_explicitness_counts = filtered_df['is_explicit'].value_counts()
filtered_explicitness_percentage = (filtered_explicitness_counts / filtered_explicitness_counts.sum()) * 100 if filtered_explicitness_counts.sum() else pd.Series({True: 0.0, False: 0.0})
filtered_explicit_percentage_by_rank = filtered_df.groupby('rank_group')['is_explicit'].mean() * 100
filtered_album_type_percentage = (filtered_df['album_type'].value_counts() / filtered_df['album_type'].value_counts().sum() * 100) if not filtered_df['album_type'].empty else pd.Series(dtype=float)

st.markdown("##### Key Findings and Insights")

full_min_date = df_merged['date'].min().date()
full_max_date = df_merged['date'].max().date()
is_date_range_different = (start_date != full_min_date) or (end_date != full_max_date)

# Get top artist for filtered data
filtered_top_artist = filtered_df['artist'].value_counts().index[0] if not filtered_df.empty and len(filtered_df['artist'].value_counts()) > 0 else 'Unknown'
filtered_top_artist_count = filtered_df['artist'].value_counts().iloc[0] if not filtered_df.empty and len(filtered_df['artist'].value_counts()) > 0 else 0

# Get top artist for full dataset
full_top_artist = df_merged['artist'].value_counts().index[0] if not df_merged.empty and len(df_merged['artist'].value_counts()) > 0 else 'Unknown'
full_top_artist_count = df_merged['artist'].value_counts().iloc[0] if not df_merged.empty and len(df_merged['artist'].value_counts()) > 0 else 0

st.markdown("**1. Artist Dominance and Diversity:**")
st.write(f"- Artist Concentration Index = **{filtered_artist_concentration_index:.2f}%**, Diversity Score = **{filtered_diversity_score:.4f}**, Content Variety Index = **{filtered_content_variety_index:.4f}**.")
st.write(f"- Top artist: **{filtered_top_artist}** with **{filtered_top_artist_count}** appearances.")
if is_date_range_different:
    st.write(f"- Full dataset: Artist Concentration Index = **{artist_concentration_index:.2f}%**, Diversity Score = **{diversity_score:.4f}**, Content Variety Index = **{content_variety_index:.4f}**.")
    st.write(f"- Full dataset top artist: **{full_top_artist}** with **{full_top_artist_count}** appearances.")
    if filtered_artist_concentration_index > artist_concentration_index:
        st.write(f"- The current selection is more concentrated than the overall dataset by **{filtered_artist_concentration_index - artist_concentration_index:.2f} percentage points**, suggesting fewer artists dominate in this view.")
        st.markdown("- This increased concentration may indicate a period of hit-driven dominance, where fewer artists capture the majority of attention.")
    elif filtered_artist_concentration_index < artist_concentration_index:
        st.write(f"- The current selection is less concentrated than the overall dataset by **{artist_concentration_index - filtered_artist_concentration_index:.2f} percentage points**, suggesting a broader set of artists is represented in this view.")
        st.markdown("- This broader representation suggests a more diverse market in the selected period, potentially due to emerging artists gaining traction.")
    else:
        st.write("- The current selection and full dataset show the same artist concentration level, indicating the filter preserves the overall market structure.")
        st.markdown("- This consistency reflects stable market dynamics across the timeframe.")
else:
    st.write("- Since the full date range is selected, these metrics represent the baseline UK market artist dominance and diversity.")

# Get highest collaboration pair for filtered data
if not filtered_df.empty:
    filtered_collab_pairs = []
    for _, group in filtered_df.groupby(['date', 'song', 'position']):
        if group['is_collaboration'].iloc[0]:
            artists = sorted(group['artist'].unique())
            for i in range(len(artists)-1):
                filtered_collab_pairs.append(tuple(sorted([artists[i], artists[i+1]])))
    filtered_collab_counter = collections.Counter(filtered_collab_pairs)
    filtered_highest_collab = filtered_collab_counter.most_common(1)[0] if filtered_collab_counter else (None, 0)
else:
    filtered_highest_collab = (None, 0)

# Get highest collaboration pair for full dataset
if not df_merged.empty:
    full_collab_pairs = []
    for _, group in df_merged.groupby(['date', 'song', 'position']):
        if group['is_collaboration'].iloc[0]:
            artists = sorted(group['artist'].unique())
            for i in range(len(artists)-1):
                full_collab_pairs.append(tuple(sorted([artists[i], artists[i+1]])))
    full_collab_counter = collections.Counter(full_collab_pairs)
    full_highest_collab = full_collab_counter.most_common(1)[0] if full_collab_counter else (None, 0)
else:
    full_highest_collab = (None, 0)
    
# Highest collaboration network (max number of artists per track in filtered_df)
if not filtered_df.empty:
    # Group by song/date/position and count unique artists
    collab_networks = (
        filtered_df.groupby(['date', 'song', 'position'])
        .agg(num_artists=('artist', 'nunique'))
        .reset_index()
    )

    # Find the max number of artists
    max_artists = collab_networks['num_artists'].max()

    # Get all tracks with that max number of artists
    highest_network_tracks = collab_networks[collab_networks['num_artists'] == max_artists]

    # Build mapping: artist group → unique songs
    network_groups = {}
    for _, row in highest_network_tracks.iterrows():
        artists = tuple(sorted(filtered_df[
            (filtered_df['date'] == row['date']) &
            (filtered_df['song'] == row['song']) &
            (filtered_df['position'] == row['position'])
        ]['artist'].unique()))
        network_groups.setdefault(artists, set()).add(row['song'])

    # Display results
    st.markdown("**2. Collaboration Structures:**")
    st.write(f"- Average artists per track = **{filtered_track_collaborations['num_artists'].mean():.2f}**, "
             f"Top 10 collaboration frequency = **{filtered_collaboration_frequency_by_rank.get('Top 10', 0):.2f}%**, "
             f"Top 11-50 = **{filtered_collaboration_frequency_by_rank.get('Top 11-50', 0):.2f}%**.")

    if filtered_highest_collab[0]:
        st.write(f"- Highest collaboration pair: **{filtered_highest_collab[0][0]}** & **{filtered_highest_collab[0][1]}** "
                 f"with **{filtered_highest_collab[1]}** collaborations.")

    if max_artists > 0:
        st.write(f"- Highest collaboration network: **{max_artists} artists** together.")
        st.markdown(" Artist groups with the highest number of collaborators and their unique songs:")
        for artists, songs in network_groups.items():
            st.markdown(f" - **{', '.join(artists)}** (unique song/songs: **{', '.join(sorted(songs))}**)")
else:
    max_artists, network_groups = (0, {})

if is_date_range_different:
    st.write(f"- Full dataset: average artists per track = **{average_artists_per_track:.2f}**, "
             f"Top 10 collaboration frequency = **{collaboration_frequency_by_rank.get('Top 10', 0):.2f}%**, "
             f"Top 11-50 = **{collaboration_frequency_by_rank.get('Top 11-50', 0):.2f}%**.")
    if full_highest_collab[0]:
        st.write(f"- Full dataset highest collaboration pair: **{full_highest_collab[0][0]}** & **{full_highest_collab[0][1]}** "
                 f"with **{full_highest_collab[1]}** collaborations.")
else:
    st.write("- Since the full date range is selected, these metrics represent the baseline UK market collaboration structures.")

st.markdown("**3. Content Explicitness:**")
st.write(f"- **{filtered_explicitness_percentage.get(True, 0):.2f}%** explicit, **{filtered_explicitness_percentage.get(False, 0):.2f}%** non-explicit.")
st.write(f"- Top 10 explicitness = **{filtered_explicit_percentage_by_rank.get('Top 10', 0):.2f}%**, Top 11-50 = **{filtered_explicit_percentage_by_rank.get('Top 11-50', 0):.2f}%**.")
if is_date_range_different:
    st.write(f"- Full dataset: **{explicitness_percentage.get(True, 0):.2f}%** explicit, **{explicitness_percentage.get(False, 0):.2f}%** non-explicit.")
    st.write(f"- Full dataset Top 10 explicitness = **{explicit_percentage_by_rank.get('Top 10', 0):.2f}%**, Top 11-50 = **{explicit_percentage_by_rank.get('Top 11-50', 0):.2f}%**.")

st.markdown("**4. Album Structure and Release Strategy:**")
st.write(f"- `single` = **{filtered_album_type_percentage.get('single', 0):.2f}%**, `album` = **{filtered_album_type_percentage.get('album', 0):.2f}%**.")
if is_date_range_different:
    st.write(f"- Full dataset: `single` = **{album_type_percentage.get('single', 0):.2f}%**, `album` = **{album_type_percentage.get('album', 0):.2f}%**.")

# Get most popular duration interval for filtered data
if not filtered_df.empty and 'duration_range' in filtered_df.columns:
    filtered_duration_range_counts = filtered_df['duration_range'].value_counts()
    filtered_most_popular_duration = filtered_duration_range_counts.index[0] if len(filtered_duration_range_counts) > 0 else 'Unknown'
else:
    filtered_most_popular_duration = 'Unknown'

# Get most popular duration interval for full dataset
if 'duration_range' in df_merged.columns:
    full_duration_range_counts = df_merged['duration_range'].value_counts()
    full_most_popular_duration = full_duration_range_counts.index[0] if len(full_duration_range_counts) > 0 else 'Unknown'
else:
    full_most_popular_duration = 'Unknown'

st.markdown("**5. Track Duration and Format:**")
st.write(f"- Short-form = **{duration_percentage_filtered.get('short-form', 0):.2f}%**, long-form = **{duration_percentage_filtered.get('long-form', 0):.2f}%**.")
st.write(f"- Most popular duration interval: **{filtered_most_popular_duration}**.")
if is_date_range_different:
    st.write(f"- Full dataset: short-form = **{duration_percentage.get('short-form', 0):.2f}%**, long-form = **{duration_percentage.get('long-form', 0):.2f}%**.")
    st.write(f"- Full dataset most popular duration interval: **{full_most_popular_duration}**.")

present_avg_duration = filtered_df['duration_min'].mean() if not filtered_df.empty else 0
overall_avg_duration = df_merged['duration_min'].mean()
if is_date_range_different:
    if present_avg_duration > overall_avg_duration:
        st.write(f"- The current selection is slightly longer on average (**{present_avg_duration:.2f} min**) than the full dataset (**{overall_avg_duration:.2f} min**). This suggests the selected view leans toward longer tracks.")
        st.markdown("- This trend toward longer durations may reflect a period favoring in-depth musical experiences.")
    elif present_avg_duration < overall_avg_duration:
        st.write(f"- The current selection is slightly shorter on average (**{present_avg_duration:.2f} min**) than the full dataset (**{overall_avg_duration:.2f} min**). This reinforces the short-form preference.")
        st.markdown("- This emphasis on shorter tracks could be driven by fast-paced digital consumption habits.")
    else:
        st.write(f"- The average track duration is equal for both the current selection and the full dataset (**{overall_avg_duration:.2f} min**), meaning the filter preserves the overall duration profile.")
        st.markdown("- This stability indicates consistent duration preferences in the UK market.")
else:
    st.write(f"- The average track duration is **{present_avg_duration:.2f} min**, representing the baseline UK market.")

st.markdown("#### Strategic Insights")
if is_date_range_different:
    st.write(f"- The current selection shows that { 'short-form content is even stronger' if duration_percentage_filtered.get('short-form',0) >= duration_percentage.get('short-form',0) else 'longer-form content is more prominent' } in this view.")
    st.markdown("- This variation in format preference may indicate shifting consumption patterns influenced by platform algorithms or listener demographics.")
    st.write(f"- Compared to the full dataset, the current selection { 'maintains broad collaboration strength' if filtered_collaboration_frequency_by_rank.get('Top 10',0) >= collaboration_frequency_by_rank.get('Top 10',0) else 'shows a slightly different collaboration profile' }.")
    st.markdown("- Collaboration trends can reflect industry networking or fan preferences for diverse artist combinations.")
    st.write("- The comparison between the current selection and the full dataset helps the team understand whether this view reflects a representative slice of the UK market or a more specialized subsegment.")
else:
    st.write("- Since the full date range is selected, these analytics reflect the complete UK market dynamics without need for comparison.")

st.markdown('---')

# --- 1. Initial Data Loading and Preprocessing (from Section IX) ---
print("--- Starting Data Preparation and Model Training ---")

# Re-initialize df from the original CSV and perform initial preprocessing
df = filtered_df.copy() # Use the already filtered data for consistency with the dashboard filters
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

# --- 3.1 Predictive Modeling - Random Forest (No Engineered Features) (from Section X) ---
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

# --- 3.2 Predictive Modeling - Random Forest (With Engineered Features) (from Section XIII) ---
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

# --- 3.3 Model Comparison DataFrames (from Section XIV) ---
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

# --- 4. Time Series Data (`unique_artists_per_day`) (from Section II, needed for dashboard) ---
unique_artists_per_day = filtered_df.groupby('date')['artist'].nunique()
print("Unique artists per day calculated for Time Series Analysis.")

# --- 5. Genre Prediction Function and Application (from Section XV) ---
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# Conceptual definition of major genres
major_genres = ['Pop', 'Rock', 'Hip-Hop/Rap', 'Jazz', 'Country',
                'Classical', 'Dance', 'R&B/soul', 'Electronic/EDM', 'Folk',
                'Metal', 'Blues', 'Reggae', 'Instrumental', 'Indie',
                'Gospel', 'Punk', 'Latin', 'Afrobeats', 'World Music']

# Load CLIP model from openai
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def predict_genre_from_image_ai_conceptual(image_url):
    """
    Predict genre from album cover URL using CLIP zero-shot classification.
    Always returns one of the major_genres.
    """
    if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.startswith('http'):
        return 'Unknown'
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Compare image against all genre prompts
        # inputs = processor(text=major_genres, images=image, return_tensors="pt", padding=True)
        text_inputs = processor.tokenizer(
            major_genres,
            padding=True,
            return_tensors="pt"
        )
        image_inputs = processor.image_processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            **text_inputs,
            **image_inputs
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        predicted_idx = probs.argmax(-1).item()
        return major_genres[predicted_idx]
    except Exception:
        return "Unknown"

@st.cache_data(show_spinner=False)
def build_genre_mapping(unique_urls):
    mapping = {}
    for url in tqdm(unique_urls, desc="Predicting genre from cover", unit="image"):
        mapping[url] = predict_genre_from_image_ai_conceptual(url)
    return mapping

print("Conceptual AI-driven genre prediction function defined.\nGenre Prediction is in progress and may take some time (2-3 minutes)")

unique_album_covers = df_merged['album_cover_url'].dropna().unique()
genre_mapping = build_genre_mapping(unique_album_covers)
df_merged['genre'] = df_merged['album_cover_url'].map(genre_mapping)
print("Genre prediction applied to df_merged.")

#print("Conceptual AI-driven genre prediction function defined.\nGenre Prediction is in progress and may take some time (2-3 minutes)")
#
## Ensure tqdm is applied correctly to pandas for progress tracking
#tqdm.pandas()
#
#unique_album_covers = df_merged['album_cover_url'].unique()
#unique_covers_df = pd.DataFrame({'album_cover_url': unique_album_covers})
#unique_covers_df['genre_predicted'] = unique_covers_df['album_cover_url'].progress_apply(predict_genre_from_image_ai_conceptual)
#genre_mapping = unique_covers_df.set_index('album_cover_url')['genre_predicted'].to_dict()
#df_merged['genre'] = df_merged['album_cover_url'].map(genre_mapping)
#print("Genre prediction applied to df_merged.")

# --- 5.1 Genre-Specific Analysis DataFrames (from Section XVI) ---
genre_popularity_stats = df_merged.groupby('genre')['popularity'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
genre_explicitness_percentage = df_merged.groupby('genre')['is_explicit'].mean() * 100
genre_duration_stats = df_merged.groupby('genre')['duration_min'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
print("Genre-specific statistics calculated.")
st.write("Genre-specific statistics calculated.")

print("--- All required dataframes and variables are now prepared. ---")

# --- Dashboard Title and Introduction ---
st.header("Recommendational Analysis Dashboard For UK Music Market Listeners")
st.markdown("""
This dashboard presents key insights and recommendations from the UK Music Market Analysis,
leveraging our data validation, descriptive analysis, and predictive modeling capabilities.
""")

# --- Recommendation 1: Predictive Modeling of Chart Success ---
st.subheader("**Predictive Modeling of Chart Success**")
st.markdown("""
Our Random Forest model, especially after careful feature engineering, demonstrated significant
predictive power for identifying tracks likely to achieve Top 10 chart success.
""")

# Display Classification Report (Table)
if 'rf_metrics_eng_df' in locals():
    st.write("Random Forest Model Performance (Without Engineered Features):")
    st.dataframe(rf_metrics_df.drop(columns=['support'], errors='ignore'))
    st.write("Random Forest Model Performance (With Engineered Features):")
    st.dataframe(rf_metrics_eng_df.drop(columns=['support'], errors='ignore'))
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
    fig_accuracy_comp, ax_accuracy_comp = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.barplot(x='Model', y='Accuracy', data=accuracy_summary_df, palette='viridis', hue='Model', legend=False, ax=ax_accuracy_comp)
    ax_accuracy_comp.set_title('Comparison of Model Accuracies')
    ax_accuracy_comp.set_xlabel('Model')
    ax_accuracy_comp.set_ylabel('Accuracy Score')
    ax_accuracy_comp.set_ylim(0, 1)  # Accuracy scores range from 0 to 1
    plt.xticks(rotation=45, ha='right')

    st.pyplot(fig_accuracy_comp)
    plt.close(fig_accuracy_comp)
else:
    st.warning("`accuracy_summary_df` not found. Please ensure the model accuracy summary section was run.")

# New 3D F1-score comparison plot
st.write("**3D Comparison of F1-scores: Logistic Regression vs. Random Forest (Without Engineered Features):**")

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

st.markdown("- Random Forest with engineered features outperforms Logistic Regression in predicting Top 10 success, emphasizing the value of feature interactions in UK chart modeling.")
st.markdown("- Model enhancements like duration-artist interactions capture nuanced factors driving UK music market dynamics.")

st.markdown('---')

# --- Section 2: Feature Engineering Visualizations (Relevant to Chart Success) ---
st.subheader("Feature Engineering Visualizations for Chart Success")
st.markdown("""
Visualizations of engineered features provide insights into their relationship with chart success.
""")

st.write("**Chart Success by Day of the Week:**")
fig_day_of_week, ax_day_of_week = plt.subplots(figsize=(10, 6), constrained_layout=True)
sns.countplot(x='day_of_week', hue='chart_success', data=df_merged, palette='viridis', ax=ax_day_of_week)
ax_day_of_week.set_title('Chart Success by Day of the Week')
ax_day_of_week.set_xlabel('Day of Week (0=Monday, 6=Sunday)')
ax_day_of_week.set_ylabel('Number of Tracks')
ax_day_of_week.legend(title='Chart Success (0=No, 1=Yes)')
ax_day_of_week.set_xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

st.pyplot(fig_day_of_week)
plt.close(fig_day_of_week)

st.markdown("- Chart success varies by day, with certain weekdays showing higher Top 10 entries, indicating optimal release timing in the UK market.")
st.markdown("- Weekly patterns may reflect audience engagement cycles and promotional strategies in UK music consumption.")

st.write("**Interaction of Duration and Number of Artists vs. Popularity:**")
fig_duration_x_artists, ax_duration_x_artists = plt.subplots(figsize=(12, 7), constrained_layout=True)
sns.scatterplot(x='duration_x_num_artists', y='popularity', hue='chart_success', data=df_merged, palette='coolwarm', alpha=0.6, ax=ax_duration_x_artists)
ax_duration_x_artists.set_title('Interaction of Duration and Number of Artists vs. Popularity by Chart Success')
ax_duration_x_artists.set_xlabel('Duration (min) * Number of Artists')
ax_duration_x_artists.set_ylabel('Popularity')
ax_duration_x_artists.legend(title='Chart Success (0=No, 1=Yes)')

st.pyplot(fig_duration_x_artists)
plt.close(fig_duration_x_artists)

st.markdown("- Longer durations with more artists correlate with higher popularity, suggesting collaborative extended tracks appeal to UK listeners.")
st.markdown("- Interaction effects highlight how production complexity influences chart performance in the UK market.")

st.write("**Distribution of Explicit Track Duration by Chart Success:**")
fig_explicit_duration, ax_explicit_duration = plt.subplots(figsize=(8, 6), constrained_layout=True)
sns.violinplot(x='chart_success', y='explicit_duration', data=df_merged, palette='Set2', hue='chart_success', legend=False, ax=ax_explicit_duration)
ax_explicit_duration.set_title('Distribution of Explicit Track Duration by Chart Success')
ax_explicit_duration.set_xlabel('Chart Success (0=No, 1=Yes)')
ax_explicit_duration.set_ylabel('Explicit Duration (minutes)')
ax_explicit_duration.set_xticks(ticks=[0, 1], labels=['Not Top 10', 'Top 10'])

st.pyplot(fig_explicit_duration)
plt.close(fig_explicit_duration)

st.markdown("- Explicit tracks in Top 10 tend to be shorter, balancing maturity with concise delivery for UK audience preferences.")
st.markdown("- Duration-explicitness interplay reveals content strategy nuances in achieving UK chart success.")

st.markdown('---')

# --- Recommendation 3: Time Series Analysis of Trends ---
st.subheader("**Time Series Analysis of Trends**")
st.markdown("""
Analyzing trends over time can reveal seasonality, shifts in artist dominance, or changes in content preferences.
Here's a look at the number of unique artists appearing in the Top 50 chart each day.
""")

if 'unique_artists_per_day' in locals():
    st.write("**Daily Unique Artists in Top 50:**")

    # Use constrained_layout=True to avoid tight_layout warnings
    fig_unique_artists, ax_unique_artists = plt.subplots(figsize=(12, 6), constrained_layout=True)
    unique_artists_per_day_df = unique_artists_per_day.reset_index()
    # Assuming the date format is 'DD-MM-YYYY' based on previous processing of df['date']
    unique_artists_per_day_df['date'] = pd.to_datetime(unique_artists_per_day_df['date'], dayfirst=True)
    unique_artists_per_day_df = unique_artists_per_day_df.sort_values('date')  # Sort by date for correct line plot
    sns.lineplot(x='date', y='artist', data=unique_artists_per_day_df, ax=ax_unique_artists)
    ax_unique_artists.set_title('Number of Unique Artists in Top 50 Per Day')
    ax_unique_artists.set_xlabel('Date')
    ax_unique_artists.set_ylabel('Number of Unique Artists')
    plt.xticks(rotation=45)
    st.pyplot(fig_unique_artists)
    plt.close(fig_unique_artists)
else:
    st.warning("`unique_artists_per_day` data not found. Please ensure the artist dominance analysis section was run.")

st.markdown("- Unique artist counts fluctuate over time, revealing periods of high diversity versus concentration in UK Top 50 charts.")
st.markdown("- Temporal trends may indicate market saturation, new entries, or seasonal influences on UK music landscape.")

st.markdown('---')

# --- Recommendation 4: Genre-Specific Analysis ---
st.subheader("Genre-Specific Analysis (Conceptual)")
st.markdown("""
While genre prediction is currently conceptual (using CLIPModel & CLIPProcessor from OpenAI) and sensitive in nature (for prediction of genre (based on album_cover_url column as a proxy for images)),
we can explore how different genres might relate to popularity, explicitness, and duration.
""")

if 'genre_popularity_stats' in locals() and not df_merged.empty:
    st.write("**Genre vs. Popularity:**")
    fig_genre_pop, ax_genre_pop = plt.subplots(figsize=(14, 7), constrained_layout=True)
    sns.boxplot(
        x='genre', y='popularity', data=df_merged,
        palette='coolwarm', hue='genre', legend=False,
        order=genre_popularity_stats.index, ax=ax_genre_pop
    )
    ax_genre_pop.set_title('Popularity Distribution by Genre')
    ax_genre_pop.set_xlabel('Genre')
    ax_genre_pop.set_ylabel('Popularity Score')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_genre_pop)
    plt.close(fig_genre_pop)

    if not genre_popularity_stats.empty:
        top_genre = genre_popularity_stats.index[0]
        bottom_genre = genre_popularity_stats.index[-1]
        top_pop = genre_popularity_stats['mean'].iloc[0]
        bottom_pop = genre_popularity_stats['mean'].iloc[-1]
        st.markdown(f"- **{top_genre}** leads in **popularity** with an average of **{top_pop:.1f}**, while **{bottom_genre}** ranks lowest at **{bottom_pop:.1f}**, showing clear **genre-based listener appeal** differences in the current filtered UK market.")
        st.markdown("- Popularity variations by genre may stem from cultural trends, marketing strategies, or demographic preferences in the UK.")
else:
    st.warning("`genre_popularity_stats` or `df_merged` not found. Please ensure the genre analysis section was run.")

if 'genre_explicitness_percentage' in locals() and not df_merged.empty:
    st.write("**Genre vs. Explicitness:**")
    fig_genre_exp, ax_genre_exp = plt.subplots(figsize=(14, 7), constrained_layout=True)
    sns.barplot(
        x=genre_explicitness_percentage.index,
        y=genre_explicitness_percentage.values,
        palette='viridis', hue=genre_explicitness_percentage.index,
        legend=False,
        order=genre_explicitness_percentage.sort_values(ascending=False).index,
        ax=ax_genre_exp
    )
    ax_genre_exp.set_title('Percentage of Explicit Content by Genre')
    ax_genre_exp.set_xlabel('Genre')
    ax_genre_exp.set_ylabel('Percentage Explicit (%)')
    ax_genre_exp.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_genre_exp)
    plt.close(fig_genre_exp)

    if not genre_explicitness_percentage.empty:
        most_explicit_genre = genre_explicitness_percentage.idxmax()
        least_explicit_genre = genre_explicitness_percentage.idxmin()
        most_explicit_pct = genre_explicitness_percentage.max()
        least_explicit_pct = genre_explicitness_percentage.min()
        st.markdown(f"- **{most_explicit_genre}** shows the highest **content maturity** at **{most_explicit_pct:.1f}%** explicit tracks, while **{least_explicit_genre}** is most **family-friendly** at **{least_explicit_pct:.1f}%**, reflecting distinct **audience expectations** across genres in the current filtered dataset.")
        st.markdown("- Explicitness differences highlight genre-specific cultural norms and target audience demographics in UK music.")
else:
    st.warning("`genre_explicitness_percentage` or `df_merged` not found. Please ensure the genre analysis section was run.")

if 'genre_duration_stats' in locals() and not df_merged.empty:
    st.write("**Genre vs. Duration:**")
    fig_genre_dur, ax_genre_dur = plt.subplots(figsize=(14, 7), constrained_layout=True)
    sns.boxplot(
        x='genre', y='duration_min', data=df_merged,
        palette='plasma', hue='genre', legend=False,
        order=genre_duration_stats.index, ax=ax_genre_dur
    )
    ax_genre_dur.set_title('Track Duration Distribution by Genre')
    ax_genre_dur.set_xlabel('Genre')
    ax_genre_dur.set_ylabel('Duration (minutes)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_genre_dur)
    plt.close(fig_genre_dur)

    if not genre_duration_stats.empty:
        longest_genre = genre_duration_stats.index[0]
        shortest_genre = genre_duration_stats.index[-1]
        longest_dur = genre_duration_stats['mean'].iloc[0]
        shortest_dur = genre_duration_stats['mean'].iloc[-1]
        st.markdown(f"- **{longest_genre}** dominates with longer **track lengths** (avg **{longest_dur:.2f}** min), while **{shortest_genre}** favors concise content (avg **{shortest_dur:.2f}** min), highlighting **genre-specific listening patterns** and production conventions in the UK market.")
        st.markdown("- Duration preferences by genre may reflect traditional formats, audience attention spans, or production styles in UK music culture.")
else:
    st.warning("`genre_duration_stats` or `df_merged` not found. Please ensure the genre analysis section was run.")

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

# --- Recommendation 5: Multivariate Analysis ---
st.subheader("Multivariate Analysis (3D Scatter Plot)")
st.markdown("""
This 3D scatter plot visualizes the interplay between track duration, number of artists, and popularity,
with points colored by chart success and distinguished by duration category (short-form vs. long-form).
""")

if not df_merged.empty and all(col in df_merged.columns for col in ['duration_min', 'num_artists', 'popularity', 'chart_success', 'duration_category']):
    # Use constrained_layout=True to avoid tight_layout warnings
    fig_3d_scatter = plt.figure(figsize=(14, 12), constrained_layout=True)
    ax_3d_scatter = fig_3d_scatter.add_subplot(111, projection='3d')

    # Plot the scatter points, using mapped markers
    for chart_success_val, chart_success_color in {0: 'blue', 1: 'red'}.items():
        for duration_cat_name, duration_cat_marker in {'short-form': 'o', 'long-form': '^'}.items():
            subset = df_merged[
                (df_merged['chart_success'] == chart_success_val) &
                (df_merged['duration_category'] == duration_cat_name)
            ]
            ax_3d_scatter.scatter(
                subset['duration_min'], subset['num_artists'], subset['popularity'], c=chart_success_color, 
                marker=duration_cat_marker, alpha=0.6, s=15,
                label=f'Chart Success: {chart_success_val}, Duration: {duration_cat_name}' if chart_success_val == 0 else "_nolegend_"  # Only add legend once for each type
            )

    ax_3d_scatter.set_xlabel('Duration (minutes)')
    ax_3d_scatter.set_ylabel('Number of Artists')
    ax_3d_scatter.set_zlabel('Popularity')
    ax_3d_scatter.set_title('3D Scatter Plot: Duration, Artists, Popularity by Chart Success & Duration Category')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Short-form', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Long-form', markerfacecolor='gray', markersize=10),
        mpatches.Patch(color='blue', label='Not Top 10'), mpatches.Patch(color='red', label='Top 10')
    ]
    ax_3d_scatter.legend(handles=legend_elements, title='Legend')

    # Render and close cleanly
    st.pyplot(fig_3d_scatter)
    plt.close(fig_3d_scatter)

    st.write('This 3D plot highlights whether higher popularity is associated with shorter or longer tracks and whether multi-artist collaborations are more likely to reach Top 10 success. Overall, it helps reveal the combined effect of duration, artist count, and chart success for the current UK market subset.')
    
    st.markdown("- Top 10 tracks cluster with moderate durations and collaborations, while non-Top 10 spread across longer solos, showing success patterns in UK charts.")
    st.markdown("- Multivariate clusters indicate optimal combinations of duration, artists, and popularity for UK market penetration.")
else:
    st.warning("Required data for 3D scatter plot not found. Please ensure the multivariate analysis section was run.")

st.markdown('---')

st.subheader('Conclusion')
if len(filtered_df) == 0:
    st.write('The current filter selection contains no records, so the conclusion is based on the full dataset baseline only. Please adjust the filters to see dynamic UK market insights for the selected subset.')
else:
    full_min_date = df_merged['date'].min().date()
    full_max_date = df_merged['date'].max().date()
    is_date_range_different = (start_date != full_min_date) or (end_date != full_max_date)
    
    st.write('This project provides structural and cultural intelligence into the UK music market by comparing the current filter view with the full dataset baseline. The dashboard now drives recommendations from both the selected subset and the overall UK market context.')
    
    if is_date_range_different:
        concentration_trend = 'more concentrated' if filtered_artist_concentration_index > artist_concentration_index else 'less concentrated' if filtered_artist_concentration_index < artist_concentration_index else 'similarly concentrated'
        explicit_trend = 'higher' if filtered_explicitness_percentage.get(True, 0) > explicitness_percentage.get(True, 0) else 'lower' if filtered_explicitness_percentage.get(True, 0) < explicitness_percentage.get(True, 0) else 'the same'
        duration_trend = 'shorter' if filtered_short_form_pct >= overall_short_form_pct else 'longer'
        st.write(f'- The present filtered view is **{concentration_trend}** than the entire dataset: **{filtered_artist_concentration_index:.2f}%** vs **{artist_concentration_index:.2f}%** for the Artist Concentration Index.')
        st.markdown("- This difference in concentration may reflect market saturation or breakout periods in the selected timeframe.")
        st.write(f'- Content explicitness in the selected subset is **{explicit_trend}** than the full dataset baseline: **{filtered_explicitness_percentage.get(True, 0):.2f}%** explicit currently vs **{explicitness_percentage.get(True, 0):.2f}%** overall.')
        st.markdown("- Variations in explicitness could be influenced by cultural shifts or platform policies in the chosen period.")
        st.write(f'- The selected subset remains **{duration_trend}** in average track length compared to the overall UK market, with an average of **{filtered_df["duration_min"].mean():.2f} minutes** vs **{df_merged["duration_min"].mean():.2f} minutes** overall.')
        st.markdown("- Duration shifts may indicate changing consumption patterns, such as preference for quick listens or immersive experiences.")
        
        overall_genre_counts = df_merged['genre'].fillna('Unknown').value_counts()
        filtered_genre_counts = filtered_df['genre'].fillna('Unknown').value_counts()
        overall_top3 = overall_genre_counts.head(3).index.tolist()
        filtered_top3 = filtered_genre_counts.head(3).index.tolist()
        if overall_top3 and filtered_top3:
            st.write(f'- The overall top 3 predicted genres in the full dataset are **{", ".join(overall_top3)}**, while the current filtered view top 3 are **{", ".join(filtered_top3)}**.')
            if overall_top3 == filtered_top3:
                st.write('- The selected subset closely mirrors the full UK dataset genre composition, indicating the chosen filters preserve major listener genre preferences.')
                st.markdown("- This consistency suggests stable genre preferences across the UK market.")
            else:
                st.write('- The selected subset shows a different top-genre mix than the full dataset, highlighting a more specific UK listener segment in the chosen date range or artist selection.')
                st.markdown("- Genre shifts may point to emerging trends or regional preferences in the selected period.")
        
        if 'rf_accuracy_eng' in locals() and 'rf_accuracy' in locals():
            better_model = 'with engineered features' if rf_accuracy_eng >= rf_accuracy else 'without engineered features'
            st.write(f'- Predictive modeling indicates that the Random Forest model **{better_model}** performs stronger for the current UK market slice, reinforcing the value of feature engineering in chart success forecasting.')
            st.markdown("- Model performance variations highlight the importance of adapting predictive approaches to specific market segments.")
        
        st.write('- Together, these dynamic conclusions support UK-focused strategy by highlighting whether the current filter view reflects a representative market slice or a distinctive subsegment with unique preferences.')
    else:
        st.write('- Since the full date range is selected, the insights reflect the complete UK market without need for comparison.')
        st.write(f'- Artist Concentration Index: **{filtered_artist_concentration_index:.2f}%**, indicating the level of market dominance by top artists.')
        st.write(f'- Content Explicitness: **{filtered_explicitness_percentage.get(True, 0):.2f}%** explicit, reflecting cultural norms in the UK market.')
        st.write(f'- Average Track Duration: **{filtered_df["duration_min"].mean():.2f} minutes**, showing baseline listener preferences.')
        overall_genre_counts = df_merged['genre'].fillna('Unknown').value_counts()
        overall_top3 = overall_genre_counts.head(3).index.tolist()
        if overall_top3:
            st.write(f'- Top 3 Genres: **{", ".join(overall_top3)}**, representing the most popular genres in the UK music market.')
        st.write('- These baseline metrics provide a comprehensive view of the UK music market dynamics.')
    
    st.write('- The dashboard is therefore useful for Atlantic Recording Corporation to identify UK listener preference indicators, collaboration strengths, and content composition trends in real time.')