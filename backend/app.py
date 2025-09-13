from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy
import joblib
import os
import json
import requests
from datetime import datetime, timedelta, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from linkedinparser import LinkedInJobFetcher
import re

# -------------------- Flask App Setup --------------------
app = Flask(__name__)
CORS(app)

# -------------------- Model + Vectorizer --------------------
MODEL_PATH = r"D:\fakejob\backend\RFCmodel"
VECTORIZER_PATH = r"D:\fakejob\backend\TFIDF_vectorizer.joblib"
RESULTS_PATH = r"D:\fakejob\backend\results.json"

print("[LOG] Loading model and vectorizer...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
nlp = spacy.load("en_core_web_sm")
print("[LOG] Model and vectorizer loaded successfully")

# -------------------- Adzuna API Keys --------------------
APP_ID = "YOUR_APP_ID"
APP_KEY = "YOUR_API_KEY"

# -------------------- Utils --------------------
def preprocess_text(text):
    print("[LOG] Preprocessing text...")
    try:
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc]
        cleaned_text = " ".join(tokens)
        X = vectorizer.transform([cleaned_text])
        df_features = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        print("[LOG] Text preprocessing successful")
        return df_features
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise

def clean_job_title(title):
    return re.sub(r"\(.*?\)|\[.*?\]", "", title).strip()

def fetch_jobs(job_title, days=30, results_per_page=20, country="in"):
    print(f"[LOG] Fetching jobs from Adzuna for title: '{job_title}'")
    if not job_title or job_title.strip() == "":
        print("[LOG] No job title provided to fetch jobs.")
        return []

    query = job_title.strip()
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": results_per_page,
        "what": query,
        "max_days_old": days,
        "content-type": "application/json"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"[LOG] Adzuna returned {len(data.get('results', []))} jobs")
    except Exception as e:
        print(f"[ERROR] Adzuna fetch failed: {e}")
        return []

    jobs = data.get("results", [])
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_jobs = []

    for job in jobs:
        created = job.get("created")
        if created:
            try:
                job_date = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if job_date >= cutoff_date:
                    job["parsed_date"] = job_date
                    recent_jobs.append(job)
            except Exception as e:
                print(f"[ERROR] Parsing job date failed: {e}")
    print(f"[LOG] {len(recent_jobs)} recent jobs after filtering by date")
    return recent_jobs

def find_similar_jobs(user_description, jobs, threshold=0.10):
    print("[LOG] Calculating similarity with fetched jobs...")
    job_texts = [job.get("description", "").strip() for job in jobs if job.get("description")]
    if not job_texts:
        print("[LOG] No job descriptions available to compare")
        return []

    try:
        documents = [user_description] + job_texts
        vectorizer_local = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer_local.fit_transform(documents)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        similar_jobs = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                similar_jobs.append({
                    "title": jobs[idx].get("title"),
                    "company": jobs[idx].get("company", {}).get("display_name"),
                    "location": jobs[idx].get("location", {}).get("display_name"),
                    "date": jobs[idx].get("parsed_date"),
                    "similarity": round(float(score), 2),
                    "description": jobs[idx].get("description")[:200] + "..."
                })

        # ✅ Sort by similarity descending
        similar_jobs = sorted(similar_jobs, key=lambda x: x["similarity"], reverse=True)

        print(f"[LOG] Found {len(similar_jobs)} similar jobs (sorted)")
        return similar_jobs
    except Exception as e:
        print(f"[ERROR] Similarity calculation failed: {e}")
        return []


def temporal_analysis_core(title, description):
    print("[LOG] Starting temporal analysis...")
    job_title = clean_job_title(title)
    if not job_title:
        print("[ERROR] No job title provided for temporal analysis.")
        return {"error": "No job title provided for temporal analysis."}

    jobs = fetch_jobs(job_title=job_title, days=7)
    similar_jobs = find_similar_jobs(description, jobs, threshold=0.10)

    from collections import Counter
    date_counts = Counter([job["date"].strftime("%Y-%m-%d") for job in similar_jobs])
    today = datetime.now().date()
    chart_data = []
    for i in range(7):
        day = today - timedelta(days=i)
        chart_data.append({
            "date": day.strftime("%Y-%m-%d"),
            "count": date_counts.get(day.strftime("%Y-%m-%d"), 0)
        })
    chart_data.reverse()
    print("[LOG] Temporal analysis completed")
    return {
        "total_jobs": len(jobs),
        "similar_jobs": similar_jobs,
        "chart_data": chart_data
    }

# @app.route("/api/temporal-analysis", methods=["POST"])
# def temporal_analysis():
#     try:
#         data = request.json or {}
#         title = data.get("title", "")
#         description = data.get("description", "")
#         print("[LOG] Temporal analysis called with title:", title)
#         result = temporal_analysis_core(title, description)
#         return jsonify(result)
#     except Exception as e:
#         print(f"[ERROR] /api/temporal-analysis endpoint failed: {e}")
#         return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_text():
    try:
        data = request.json
        title = data.get("title", "").strip()
        description = data.get("description", "").strip()

        if not title and not description:
            return jsonify({"error": "No title or description provided"}), 400

        # Combine title + description for prediction
        text = f"{title}+{description}".strip()

        # --- Prediction ---
        X_input = preprocess_text(text)
        prediction = model.predict(X_input)[0]
        if hasattr(prediction, "item"):
            prediction = prediction.item()
        class_index = list(model.classes_).index(prediction)
        probability = float(model.predict_proba(X_input)[0][class_index])

        # --- Temporal Analysis ---
        temporal_result = temporal_analysis_core(title, description)

        return jsonify({
            "description": description if description else title,
            "title": title if title else "Job Posting",
            "prediction": prediction,
            "probability": probability,
            "temporal": temporal_result
        })

    except Exception as e:
        print(f"[ERROR] /api/predict failed: {e}")
        return jsonify({"error": str(e)}), 500



def fetch_and_predict(url):
    print(f"[LOG] Fetching LinkedIn job from URL: {url}")
    scraper = LinkedInJobFetcher(headless=True)
    job_data = scraper.fetch(url)
    scraper.close()

    if not job_data:
        print("[ERROR] Failed to fetch job from LinkedIn")
        return {"error": "Failed to fetch job from LinkedIn"}
    print(job_data)

    title = job_data.get("title", "").strip() or "Job Posting"
    description = job_data.get("description", "").strip()

    if not description:
        print("[ERROR] No job description found")
        return {"error": "No job description found"}

    try:
        # --- Prediction ---
        print("[LOG] Predicting fake/real job...")
        X_input = preprocess_text(description + " " + title)
        prediction = model.predict(X_input)[0]
        if hasattr(prediction, "item"):
            prediction = prediction.item()
        class_index = list(model.classes_).index(prediction)
        probability = float(model.predict_proba(X_input)[0][class_index])
        print(f"[LOG] Prediction: {prediction}, Probability: {probability}")

        # --- Temporal analysis ---
        temporal_result = temporal_analysis_core(title, description)

        return {
            "title": title,
            "description": description,
            "prediction": prediction,
            "probability": probability,
            "temporal": temporal_result
        }

    except Exception as e:
        print(f"[ERROR] fetch_and_predict failed: {e}")
        return {"error": str(e)}

# -------------------- Endpoint --------------------
@app.route("/api/fetch-and-predict", methods=["POST"])
def fetch_and_predict_endpoint():
    try:
        data = request.json
        url = data.get("url", "").strip()
        if not url:
            print("[ERROR] No URL provided in request")
            return jsonify({"error": "No URL provided"}), 400

        result = fetch_and_predict(url)
        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] /api/fetch-and-predict endpoint failed: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/metrics', methods=['GET']) 
def get_metrics(): 
    if os.path.exists(RESULTS_PATH): 
        with open(RESULTS_PATH, 'r') as f: 
            data = json.load(f) 
            return jsonify(data) 
    else: 
        return jsonify({"error": "Results file not found"}), 404

# -------------------- Run App --------------------
if __name__ == '__main__':
    print("[LOG] Starting Flask app on port 5000")
    app.run(debug=True, port=5000)

