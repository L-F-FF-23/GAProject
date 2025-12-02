# Requirements:
# pip install requests tqdm

import requests
import json
from tqdm import tqdm

API_KEY = "YOUR_PANDASCORE_KEY"  # <-- Replace this
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

OUTPUT_FILE = "lol_esports_2025.json"
PER_PAGE = 50
START_YEAR = "2025"

# Step 1: Get a sample of videogames to find the correct slug
print("Fetching videogame slugs...")
vg_resp = requests.get("https://api.pandascore.co/v2/videogames", headers=HEADERS)
if vg_resp.status_code != 200:
    raise Exception(f"Failed to fetch videogames: {vg_resp.status_code}")

vg_data = vg_resp.json()

# Find LoL slug
lol_slug = None
for vg in vg_data:
    if "league of legends" in vg["name"].lower():
        lol_slug = vg["slug"]
        break

if not lol_slug:
    raise Exception("Could not find League of Legends slug in PandaScore.")

print(f"Using videogame slug: {lol_slug}")

# Step 2: Fetch matches
all_matches = []
page = 1

print("Fetching matches...")
while True:
    url = "https://api.pandascore.co/v2/matches"
    params = {
        "filter[videogame]": lol_slug,
        "page": page,
        "per_page": PER_PAGE,
        "sort": "begin_at"
    }

    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print("Error:", resp.status_code, resp.text)
        break

    data = resp.json()
    if not data:
        break

    for match in data:
        try:
            begin_at = match.get("begin_at", "")
            if not begin_at.startswith(START_YEAR):
                continue
            if match.get("status") != "finished":
                continue

            team1 = match["opponents"][0]["opponent"]["name"]
            team2 = match["opponents"][1]["opponent"]["name"]
            winner = match.get("winner", {}).get("name")
            all_matches.append({"team1": team1, "team2": team2, "winner": winner})
        except Exception:
            continue

    tqdm.write(f"Fetched page {page}, total matches so far: {len(all_matches)}")
    page += 1

# Step 3: Save to JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_matches, f, ensure_ascii=False, indent=2)

print(f"Done! Total matches fetched: {len(all_matches)}")
print(f"Saved JSON to {OUTPUT_FILE}")
