# United-Kingdom-Top-50-Playlist-Market-Structure-Artist-Diversity-Content-Localization-Analysis
Unified Mentor Machine Learning Internship project which includes the Research Papers (EDA, insights and recommendations), Streamlit Dashboard of Analysis and requiremnts.

## 📌 Background and Context
The UK music market is globally influential but culturally distinct due to:
- Strong domestic artist representation  
- High prevalence of collaborations  
- Unique listener sensitivity to explicit content  
- Different album vs single consumption behavior compared to the US  

For **Atlantic Recording Corporation**, understanding the structural composition of the UK Top 50 playlist is critical for:
- Artist signing strategy  
- UK-specific marketing decisions  
- Release format optimization (single vs album)  
- Cross-border promotion planning  

---

## ❓ Problem Statement
Despite access to daily UK Top 50 playlist data, stakeholders lack clarity on:
- How artist dominance is distributed in the UK market  
- Whether UK charts favor domestic or international artists  
- How collaborations influence chart presence  
- Whether explicit content performs differently in the UK  
- How album structure (single vs album size) affects success  

Without this understanding, UK market strategies risk being copied from the US, ignoring cultural differences.

---

## 📊 Dataset Fields
| Column          | Description                          |
|-----------------|--------------------------------------|
| date            | Date of playlist snapshot            |
| position        | Playlist rank (1–50)                 |
| song            | Song title                           |
| artist          | Artist(s)                            |
| popularity      | Popularity score from Atlantic API   |
| duration_ms     | Song duration (milliseconds)         |
| album_type      | Single / Album                       |
| total_tracks    | Number of tracks in album            |
| is_explicit     | Explicit content flag                |
| album_cover_url | Album artwork URL                    |

---

## 🛠 Analytical Methodology (Step-by-Step)

1. **Data Validation & Standardization**
   - Validate daily Top 50 entries  
   - Normalize artist names  
   - Split multi-artist collaborations using delimiter (&)  

2. **Artist Dominance & Diversity Analysis**
   - Count unique artists per day  
   - Measure total appearances per artist  
   - Identify top-dominating artists  
   - Compute Artist Concentration Index  

3. **Collaboration Structure Analysis**
   - Identify solo vs collaborative tracks  
   - Average collaborators per song  
   - Collaboration frequency by rank group (Top 10, Top 50)  
   - Artist collaboration network graph  

4. **Content Explicitness Analysis**
   - Explicit vs non-explicit share  
   - Explicit content distribution by rank  
   - Cultural sensitivity insights for UK listeners  

5. **Album Structure & Release Strategy Analysis**
   - Single vs album track presence  
   - Album size (total_tracks) vs playlist inclusion  
   - Release format dominance in the UK market  

6. **Track Duration & Format Analysis**
   - Duration distribution (short-form vs long-form)  
   - Duration vs popularity bucket analysis  
   - UK listener preference indicators  

7. **Market Structure Metrics**
   - Playlist concentration ratio (Top 5 artists share)  
   - Diversity score (unique artists / total entries)  
   - Content variety index  

---

## 📈 Key Performance Indicators (KPIs)

| KPI Name                  | Description              |
|----------------------------|--------------------------|
| Artist Concentration Index | Market dominance         |
| Unique Artist Count        | Diversity measure        |
| Collaboration Ratio        | Partnership prevalence   |
| Explicit Content Share     | Cultural preference      |
| Single vs Album Ratio      | Release strategy         |
| Content Variety Index      | Market balance           |

---

## 💻 Streamlit Web Application Requirements

### Core Modules
- Artist dominance leaderboard  
- Collaboration network visualization  
- Explicit vs clean content analysis  
- Album type distribution charts  
- Track duration insights  

### User Capabilities
- Date range selector  
- Artist filter  
- Solo vs collaboration toggle  
- Album type filter  

---

## 📑 Deliverables
- Research paper (EDA, insights, recommendations)  
- Streamlit dashboard (live analytics)  
- Executive summary for government stakeholders  

---

## ✅ Conclusion
This project provides **structural and cultural intelligence** into the UK music market.  
By shifting focus away from popularity trends (US project) toward **artist diversity, collaboration dynamics, and content composition**, Atlantic Recording Corporation gains **region-specific insights** essential for designing effective UK-focused music strategies in a competitive global industry.

---

## 📂 Access Dataset
Dataset available via **Unified Mentor Project Portal**.
