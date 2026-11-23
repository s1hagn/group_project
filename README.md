# YouTube Video Virality: Exploratory Analysis and Machine Learning
**LINLIANG CHEN, LIJIA QIAN, SIYA ZHANG, MINXI LI**

## 1. Project Overview
Why do some YouTube videos receive substantially more attention than others? This project explores how video characteristics are associated with virality and examines whether a machine learning model can classify a video as viral using features available around the time of upload.

The project follows a practical data science workflow: data cleaning, exploratory data analysis (EDA), supervised classification, model evaluation, and presentation of results in an interactive dashboard.

## 2. Research Questions
- RQ1: What factors are associated with YouTube video virality? We compare viral rates across video categories, duration groups, and their combinations, and examine subscriber counts by category.
- RQ2: Can machine learning predict whether a YouTube video will go viral? We compare Logistic Regression and Decision Tree classifiers using video and upload characteristics.

## 3. Dataset
**Source:** *What Makes a YT Video Go Viral?: 12,000 Vids* from kaggle  https://www.kaggle.com/datasets/vedauv/what-makes-a-yt-video-go-viral-12000-vids

The project uses a synthetic YouTube video dataset. Each row represents a video, with information about its characteristics, upload context, channel size, engagement, and a binary viral label. In the project, viral = 1 represents a video in the top 20% by views, while viral = 0 represents a non-viral video.

**Key fields include:**
| Group | Variables | Purpose |
| --- | --- | --- |
| Identifier | `video_id` | Identify videos and remove duplicates |
| Video characteristics | `category`, `duration_sec`, `title_length`, `thumbnail_face`, `tags_count`, `description_length` | Describe content and presentation |
| Upload information | `upload_hour`, `upload_day` | Describe timing of upload |
| Channel characteristic | `subscriber_count` | Explore differences in channel size |
| Performance indicators | `views`, `likes`, `comments`, `shares` | Describe outcomes after upload |
| Target | `viral` | Binary classification label (0 or 1) |

**Important data consideration:** The dataset is simulated, and some engagement indicators are generated in relation to views. These indicators should not be treated as independent explanations of virality or used to predict a view-based target before those outcomes are observed. Accordingly, the classification models exclude views, likes, comments, and shares from their input features.

## 4. Repository Structure

```text
group_project/
├── README.md
├── viral_videos.csv
├── cleaned_viral_videos.csv
├── data_cleaning.py
├── EDA_indicators.py
├── Machine_Learning_Modelling.py
├── dashboard.py
├── viral_rate_by_category.png
├── viral_rate_by_duration.png
├── viral_rate_heatmap.png
├── median_subscribers_by_category.png
├── ml_confusion_matrices.png
└── ml_roc_comparison.png
