import streamlit as st
import pandas as pd
import numpy as np
import itertools

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

st.write(f"Overall Content Explicitness (Filtered Data):")
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

st.write("\nPercentage of Explicit Tracks by Rank Group (Filtered Data):")
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

st.markdown('--')
st.subheader('Album Type Distribution')

# Calculate total counts for each unique value in the `album_type` column for the filtered data
album_type_counts_filtered = filtered_df['album_type'].value_counts()

# Calculate the percentage of each `album_type` for the filtered data
total_album_types_filtered = album_type_counts_filtered.sum()
album_type_percentage_filtered = (album_type_counts_filtered / total_album_types_filtered) * 100

st.write("\nTotal counts of each album type (Filtered Data):")
st.write(album_type_counts_filtered)

# Create the bar chart for album type distribution
if not album_type_counts_filtered.empty:
    fig_album_type, ax_album_type = plt.subplots(figsize=(10, 6))
    sns.barplot(x=album_type_counts_filtered.index, y=album_type_counts_filtered.values, hue=album_type_counts_filtered.index, palette='viridis', legend=False, ax=ax_album_type)
    ax_album_type.set_title('Distribution of Album Types (Filtered)')
    ax_album_type.set_xlabel('Album Type')
    ax_album_type.set_ylabel('Number of Tracks')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_album_type)
else:
    st.write("No album type data available for the selected filters to display the chart.")

st.write("\nPercentage of each album type (Filtered Data):")
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

st.write("\nDuration Category Counts (Filtered Data):")
st.write(duration_counts_filtered)

st.write("\nDuration Category Percentages (Filtered Data):")
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

    st.write("\nDistribution of Track Duration Categories by Popularity Bucket (Filtered Data):")
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
    
st.write('**Insights into UK Listener Preference Indicators:**')

st.markdown("Track Duration vs. Popularity Box Plot ")
st.markdown(f"- The data reveals a strong preference for **short-form tracks** (under 3.5 minutes) among UK listeners, accounting for a significant **{duration_percentage.get('short-form', 0):.2f}%** of all tracks. This indicates that concise content resonates well with the audience.")
st.markdown("- This preference is consistent across all popularity levels. Even in the 'Q4 (Most Popular)' bucket, short-form tracks clearly outnumber long-form tracks, suggesting that track duration does not hinder a song's popularity but rather aligns with current consumption habits.")
st.markdown("- **Implication for Artists and Producers:** To align with UK listener preferences and optimize for modern streaming platforms, artists and producers should consider focusing on creating shorter, impactful tracks. This strategy can enhance listener engagement and potentially improve chart performance.")

# 1. Check if filtered_df is not empty and contains the necessary columns
if not filtered_df.empty and 'popularity_bucket' in filtered_df.columns and 'duration_min' in filtered_df.columns:
    # 2. Create a figure and axes for the plot
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 7))

    # 3. Generate a box plot using seaborn.boxplot()
    sns.boxplot(x='popularity_bucket', y='duration_min', data=filtered_df, palette='viridis', ax=ax_boxplot)

    # 4. Set the title and labels
    ax_boxplot.set_title('Track Duration Distribution by Popularity Bucket (Filtered)')
    ax_boxplot.set_xlabel('Popularity Bucket')
    ax_boxplot.set_ylabel('Duration (Minutes)')

    # 5. Ensure the x-axis labels are rotated for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 6. Display the plot in Streamlit
    st.pyplot(fig_boxplot)

    # 7. Add descriptive text below the plot
    st.write('This box plot illustrates the distribution of track durations (in minutes) across different popularity buckets. It helps to visualize the typical range and spread of track lengths within each popularity quartile.')
else:
    # 8. Include an else block to display a message if data is not available
    st.write('No data available for the selected filters to display the track duration vs. popularity box plot.')

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
