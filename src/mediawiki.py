import requests
import json
import re
from datetime import datetime

API_URL = "https://lol.fandom.com/api.php"

def get_all_match_pages():
    matches = []
    apcontinue = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",
            "aplimit": "500",
            "apnamespace": "0"
        }

        if apcontinue:
            params["apcontinue"] = apcontinue

        res = requests.get(API_URL, params=params).json()
        pages = res["query"]["allpages"]

        for page in pages:
            title = page["title"]
            if " vs " in title:
                matches.append(title)

        if "continue" in res:
            apcontinue = res["continue"]["apcontinue"]
        else:
            break

    return matches

def get_page_content(title):
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvslots": "main",
        "rvprop": "content",
        "titles": title,
    }
    res = requests.get(API_URL, params=params).json()

    pages = res["query"]["pages"]
    for pageid in pages:
        try:
            return pages[pageid]["revisions"][0]["slots"]["main"]["*"]
        except KeyError:
            return ""
    return ""

def extract_match_info(content):
    """Extract team1, team2, winner only."""
    team1 = re.search(r"\|\s*team1\s*=\s*(.*)", content)
    team2 = re.search(r"\|\s*team2\s*=\s*(.*)", content)
    winner = re.search(r"\|\s*winner\s*=\s*(.*)", content)
    date = re.search(r"\|\s*date\s*=\s*(.*)", content)

    def clean(x):
        return x.group(1).strip() if x else None

    # We still extract date internally for filtering, but do not output it.
    return {
        "team1": clean(team1),
        "team2": clean(team2),
        "winner": clean(winner),
        "_date": clean(date)  # internal use only
    }

def filter_matches_2025(match):
    try:
        if match["_date"] is None:
            return False
        year = datetime.strptime(match["_date"], "%Y-%m-%d").year
        return year == 2025
    except:
        return False

def main():
    print("Fetching match pages...")
    pages = get_all_match_pages()

    results = []

    for title in pages:
        content = get_page_content(title)
        info = extract_match_info(content)

        if filter_matches_2025(info):
            # Remove internal date before saving
            del info["_date"]
            results.append(info)

    with open("matches_2025.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(results)} matches to matches_2025.json")

if __name__ == "__main__":
    main()
