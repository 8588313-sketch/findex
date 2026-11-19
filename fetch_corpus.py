import json
import urllib.parse
import urllib.request
from pathlib import Path

TITLES = {
    "uk": ["Київ", "Україна", "Тарас_Шевченко", "Дніпро", "Карпати",
           "Львів", "Python", "Математика", "Фізика", "Музика"],
    "en": ["Python_(programming_language)", "Kyiv", "Alan_Turing",
           "Search_engine", "Unicode", "Linux", "Ada_Lovelace",
           "Moon", "Chess", "Coffee"],
}

out = Path("data")
out.mkdir(exist_ok=True)

for lang, titles in TITLES.items():
    for title in titles:
        params = urllib.parse.urlencode({
            "action": "query", "prop": "extracts", "explaintext": 1,
            "format": "json", "redirects": 1, "titles": title,
        })
        url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "findex-lab/0.1 (student project)"})
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        page = next(iter(data["query"]["pages"].values()))
        text = page.get("extract", "")
        (out / f"{lang}_{title}.txt").write_text(text, encoding="utf-8")
        print(lang, title, len(text))
