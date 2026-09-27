import os, csv, shutil, requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

G_Api = os.environ["GUARDIAN_API_KEY"]
TC = 250
PH_TC = 70
PHF = [
    "https://www.philstar.com/rss/pilipino-star-ngayon/bansa",
    "https://www.philstar.com/rss/pilipino-star-ngayon/probinsiya",
    "https://www.philstar.com/rss/pilipino-star-ngayon/metro",
    "https://www.philstar.com/rss/pilipino-star-ngayon/opinyon",
    "https://www.philstar.com/rss/pilipino-star-ngayon/palaro",
    "https://www.philstar.com/rss/pilipino-star-ngayon/showbiz",
    "https://www.philstar.com/rss/pang-masa",
]

def clear_folder(folder):
    folder = Path(folder)
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

def fetchnews(n, out_folder="Data/true_global"):
    clear_folder(out_folder)
    saved, page = 0, 1
    while saved < n:
        resp = requests.get("https://content.guardianapis.com/search",
            params={"api-key": G_Api, "show-fields": "bodyText",
                    "page-size": 50, "page": page, "order-by": "newest"}, timeout=15)
        resp.raise_for_status()
        results = resp.json()["response"]["results"]
        if not results: break
        for a in results:
            if saved >= n: break
            body = a.get("fields", {}).get("bodyText", "").strip()
            if not body: continue
            Path(f"{out_folder}/real_{saved:04d}.txt").write_text(body, encoding="utf-8")
            saved += 1
        page += 1
    print(f"Saved {saved} real articles")

def getphilnews(n, out_folder="Data/true_ph"):
    clear_folder(out_folder)
    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
    headers = {"User-Agent": "Mozilla/5.0"}
    saved = 0
    for feed_url in PHF:
        if saved >= n: break
        try:
            resp = requests.get(feed_url, headers=headers, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            print(f"Skipping {feed_url}: {e}"); continue
        for item in root.findall(".//item"):
            if saved >= n: break
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            content_el = item.find("content:encoded", ns)
            body = content_el.text.strip() if content_el is not None and content_el.text else desc
            text = f"{title}\n\n{body}".strip()
            if not text: continue
            Path(f"{out_folder}/ph_{saved:04d}.txt").write_text(text, encoding="utf-8")
            saved += 1
    print(f"Saved {saved} Philippine articles")
    return saved

def getfake(n, csv_path="Fake.csv", out_folder="Data/false"):
    clear_folder(out_folder)
    saved = 0
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            if saved >= n: break
            text = row.get("text", "").strip()
            if not text: continue
            Path(f"{out_folder}/fake_{saved:04d}.txt").write_text(text, encoding="utf-8")
            saved += 1
    print(f"Saved {saved} fake articles")

def load_fakknews_PH_false(n, csv_path="Fakknews_Ph.csv", fake_label="0", out_folder="Data/false_ph"):
    clear_folder(out_folder)
    saved = 0
    with open(csv_path, encoding="utf-8-sig", errors="ignore") as f:
        for row in csv.DictReader(f):
            if saved >= n: break
            if row.get("label", "").strip() != fake_label: continue
            text = row.get("article", "").strip()
            if not text: continue
            Path(f"{out_folder}/ph_fake_{saved:04d}.txt").write_text(text, encoding="utf-8")
            saved += 1
    print(f"Saved {saved} fake Philippine articles")

def load_data(true_folder, false_folder):
    Text, Bool = [], []
    for folder, label in [(true_folder, 1), (false_folder, 0)]:
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                Text.append(Path(path).read_text(encoding="utf-8", errors="ignore"))
                Bool.append(label)
    return Text, Bool

def train_and_save(true_folder, false_folder, model_name, vectorizer_name):
    Articles, labels = load_data(true_folder, false_folder)
    XTr, XTe, YTr, YTe = train_test_split(Articles, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    XTr_Vec = vectorizer.fit_transform(XTr)
    model = MLPClassifier(hidden_layer_sizes=(50, 100), max_iter=500, random_state=42)
    model.fit(XTr_Vec, YTr)
    joblib.dump(model, model_name)
    joblib.dump(vectorizer, vectorizer_name)
    print(f"Saved {model_name} and {vectorizer_name}")

if __name__ == "__main__":
    fetchnews(TC)
    ph_count = getphilnews(PH_TC)
    getfake(TC, csv_path="Fake.csv")
    load_fakknews_PH_false(ph_count, csv_path="Fakknews_Ph.csv", fake_label="0")
    train_and_save("Data/true_global", "Data/false", "model.pkl", "vectorizer.pkl")
    train_and_save("Data/true_ph", "Data/false_ph", "model_ph.pkl", "vectorizer_ph.pkl")
    Path("version.txt").write_text(datetime.now().strftime("%Y%m%d%H%M%S"))