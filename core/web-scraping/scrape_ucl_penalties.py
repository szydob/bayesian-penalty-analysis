"""
UEFA Champions League Penalty Shootout Scraper
================================================
Scrapes penalty shootout data from Transfermarkt for Bayesian modeling.

Three-phase pipeline:
  Phase 1: Extract match report URLs from the UCL penalty shootouts listing.
  Phase 2: Parse individual penalty kicks from each match report.
  Phase 3: Scrape player profiles for foot & position data.

Output: UCL-Shootout.csv in project/data/

Anti-scraping measures:
  - Custom User-Agent header (no default Selenium fingerprint)
  - Randomised delays (2-5s) between page loads
  - WebDriverWait for cookie banners / pop-ups
  - WebDriver flag suppression
"""

import csv
import os
import random
import re
import sys
import time
import json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # project/
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "UCL-Shootout.csv"

# Checkpoint files
PHASE1_CACHE = DATA_DIR / "checkpoint_phase1_urls.json"
KICKS_CACHE = DATA_DIR / "checkpoint_phase2_kicks.json"
PLAYERS_CACHE = DATA_DIR / "checkpoint_phase3_players.json"

BASE_URL = "https://www.transfermarkt.com"
LISTING_URL = (
    f"{BASE_URL}/uefa-champions-league/elfmeterschiessen/"
    "pokalwettbewerb/CL/saison_id//runde//land_id/0/plus/1"
)

# ─── Position mapping (Categorical ID for Stan: 1=GK, 2=DEF, 3=MID, 4=FWD) ───
POSITION_MAP = {
    # Goalkeeper
    "Goalkeeper": 1,
    # Defenders
    "Centre-Back": 2,
    "Left-Back": 2,
    "Right-Back": 2,
    "Left Wing-Back": 2,
    "Right Wing-Back": 2,
    "Sweeper": 2,
    # Midfielders
    "Defensive Midfield": 3,
    "Central Midfield": 3,
    "Attacking Midfield": 3,
    "Left Midfield": 3,
    "Right Midfield": 3,
    # Forwards
    "Centre-Forward": 4,
    "Left Winger": 4,
    "Right Winger": 4,
    "Second Striker": 4,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Driver helpers
# ═══════════════════════════════════════════════════════════════════════════════

def create_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver with anti-detection flags."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    # Generic desktop User-Agent
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    # Suppress WebDriver flags so navigator.webdriver === false
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opts)
    # Override navigator.webdriver via CDP
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def random_sleep(lo: float = 2.0, hi: float = 5.0) -> None:
    """Sleep for a random duration to avoid rate limiting."""
    time.sleep(random.uniform(lo, hi))


def safe_get(driver: webdriver.Chrome, url: str, retries: int = 3) -> None:
    """
    Navigate to *url* with retry logic.  Also attempts to dismiss any
    cookie-consent overlay that Transfermarkt shows.
    """
    for attempt in range(retries):
        try:
            driver.get(url)
            random_sleep()
            _dismiss_cookie_banner(driver)
            return
        except Exception as exc:
            print(f"  ⚠ Attempt {attempt + 1}/{retries} failed for {url}: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to load {url} after {retries} retries")


def _dismiss_cookie_banner(driver: webdriver.Chrome) -> None:
    """Try to click a cookie-accept button if present (Sourcepoint CMP)."""
    try:
        # Transfermarkt uses a Sourcepoint iframe for consent
        iframe = WebDriverWait(driver, 4).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "iframe[id^='sp_message_iframe']")
            )
        )
        driver.switch_to.frame(iframe)
        accept_btn = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "button[title='Accept All'], button[title='ACCEPT ALL'], "
                                  "button.message-component.message-button.no-children.focusable.accept-all, "
                                  "button[aria-label='Accept All']")
            )
        )
        accept_btn.click()
        driver.switch_to.default_content()
        time.sleep(1)
    except Exception:
        # No banner or already accepted – continue silently
        driver.switch_to.default_content()


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 – Extract match report URLs
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_get_match_report_urls(driver: webdriver.Chrome) -> list[dict]:
    """
    Visit the UCL penalty-shootout listing page and extract all match-report
    URLs together with a numeric Match_ID.

    Returns a list of dicts: [{"match_id": int, "url": str}, ...]
    """
    if PHASE1_CACHE.exists():
        print("  → Loading Match Report URLs from cache...")
        with open(PHASE1_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    print("═" * 70)
    print("Phase 1 · Extracting match-report URLs from the listing page")
    print("═" * 70)

    safe_get(driver, LISTING_URL)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    matches: list[dict] = []
    # Match-report links sit in the result column: <a href="/spielbericht/index/spielbericht/XXXXXX">
    for a_tag in soup.select("table.items a[href*='/spielbericht/index/spielbericht/']"):
        href = a_tag.get("href", "")
        # Extract the numeric ID at the end of the path
        m = re.search(r"/spielbericht/index/spielbericht/(\d+)", href)
        if m:
            match_id = int(m.group(1))
            # Avoid duplicates (some links appear in different columns)
            if not any(d["match_id"] == match_id for d in matches):
                matches.append({
                    "match_id": match_id,
                    "url": f"{BASE_URL}{href}",
                })

    with open(PHASE1_CACHE, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=4)

    print(f"  ✓ Found {len(matches)} unique penalty-shootout match reports.\n")
    return matches


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 – Parse penalty shootout kicks
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_elimination(kicks: list[dict]) -> list[dict]:
    """
    Walk through the penalty sequence and mark *Elimination = 1* for any kick
    whose outcome can immediately decide the match.

    Standard shootout rules:
      • Best-of-5 format first (rounds 1-5).
      • After round 5, sudden death (ABAB pattern).
      • A team is eliminated if the deficit is greater than the remaining
        kicks the trailing team has in the current best-of-5 phase, OR
        if in sudden death a round is completed with one team ahead.
    """
    home_score = 0
    away_score = 0

    # Identify the two teams from the sides (heim / gast)
    for kick in kicks:
        kick["Elimination"] = 0  # default

    # We iterate in order and after each kick check whether the match
    # can end right now.
    for i, kick in enumerate(kicks):
        side = kick["_side"]  # "heim" or "gast"
        scored = kick["Goal"]

        if side == "heim":
            home_score += scored
        else:
            away_score += scored

        # Count how many kicks each side has taken so far
        home_taken = sum(1 for k in kicks[:i + 1] if k["_side"] == "heim")
        away_taken = sum(1 for k in kicks[:i + 1] if k["_side"] == "gast")

        # ── Best-of-5 phase (each team takes up to 5) ──
        if home_taken <= 5 and away_taken <= 5:
            home_remaining = 5 - home_taken
            away_remaining = 5 - away_taken
            # Team trailing can't catch up even if they score every remaining
            if home_score - away_score > away_remaining:
                kick["Elimination"] = 1  # away eliminated
            elif away_score - home_score > home_remaining:
                kick["Elimination"] = 1  # home eliminated
        else:
            # ── Sudden death ──
            # In sudden death, teams alternate one kick each per round.
            # After both sides have taken the same number, if scores differ → done
            if home_taken == away_taken and home_score != away_score:
                kick["Elimination"] = 1
            # If one side just shot and the other already has the same count,
            # and the trailing team can't equalise (the deficit is > 1)
            elif home_taken != away_taken:
                # The side with fewer kicks still has one to take this round
                if home_taken > away_taken:
                    # Home just shot; away hasn't yet this round
                    if home_score - away_score > 1:
                        kick["Elimination"] = 1
                else:
                    if away_score - home_score > 1:
                        kick["Elimination"] = 1

    return kicks


def phase2_parse_match_report(
    driver: webdriver.Chrome,
    match_id: int,
    url: str,
) -> list[dict]:
    """
    Navigate to a match-report page and extract every penalty kick from the
    shootout section.

    Returns a list of dicts with keys:
        Match_ID, Shooter_Name, Shooter_Profile_URL, Penalty_Number, Goal, _side
    """
    safe_get(driver, url)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Locate the penalty shootout section
    pso_div = soup.find("div", id="sb-elfmeterscheissen")
    if pso_div is None:
        print(f"  ⚠ No penalty-shootout section found for match {match_id}")
        return []

    kicks: list[dict] = []
    penalty_number = 0

    for li in pso_div.find_all("li", class_=re.compile(r"sb-aktion-(heim|gast)")):
        penalty_number += 1

        # Determine side (home / away)
        if "sb-aktion-heim" in li.get("class", []):
            side = "heim"
        else:
            side = "gast"

        # Goal or miss?  Indicated by the <span> class inside sb-aktion-uhr
        icon_span = li.select_one(".sb-aktion-uhr span")
        if icon_span:
            classes = " ".join(icon_span.get("class", []))
            scored = 1 if "sb-11m-tor" in classes else 0
        else:
            scored = 0

        # Shooter name & profile URL – inside sb-aktion-aktion
        action_div = li.select_one(".sb-aktion-aktion")
        name = ""
        profile_url = ""
        if action_div:
            a_tag = action_div.find("a", class_="wichtig")
            if a_tag:
                name = a_tag.get_text(strip=True)
                href = a_tag.get("href", "")
                # Convert to a /profil/ URL for Phase 3
                # The href points to leistungsdatendetails; we need profil
                spieler_match = re.search(r"/spieler/(\d+)", href)
                if spieler_match:
                    player_id = spieler_match.group(1)
                    # Build canonical profile URL from the image link instead
                    img_link = li.select_one(".sb-aktion-spielerbild a")
                    if img_link and img_link.get("href"):
                        profile_url = f"{BASE_URL}{img_link['href']}"
                    else:
                        # Fallback: construct from player ID
                        slug = name.lower().replace(" ", "-")
                        profile_url = f"{BASE_URL}/{slug}/profil/spieler/{player_id}"

        kicks.append({
            "Match_ID": match_id,
            "Shooter_Name": name,
            "Shooter_Profile_URL": profile_url,
            "Penalty_Number": penalty_number,
            "Goal": scored,
            "_side": side,
        })

    # Compute Elimination column
    kicks = _compute_elimination(kicks)
    return kicks


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3 – Scrape player profiles for foot & position
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_position_to_id(raw_position: str) -> int:
    """
    Given a raw position string like "Defender - Right-Back", return the
    categorical Stan index (1=GK, 2=DEF, 3=MID, 4=FWD). Defaults to 4 (FWD).
    We extract the specific position after the dash.
    """
    if not raw_position:
        return 4

    # The info-table shows e.g. "Defender - Right-Back" or just "Goalkeeper"
    # We try to match the specific part after the dash first, then the whole string.
    parts = [p.strip() for p in raw_position.split("-", maxsplit=1)]

    # Try each known position against the raw string
    for known_pos, pos_id in POSITION_MAP.items():
        if known_pos.lower() in raw_position.lower():
            return pos_id

    # Fallback: if nothing matched, default to 4
    return 4


def phase3_scrape_player_profiles(
    driver: webdriver.Chrome,
    profile_urls: list[str],
) -> dict[str, dict]:
    """
    For each unique player profile URL, extract foot and position.

    Returns a dict keyed by profile URL:
        { url: {"is_left": int, "Position_ID": int}, ... }
    """
    print("\n" + "═" * 70)
    print("Phase 3 · Scraping player profiles for foot & position")
    print("═" * 70)

    player_data: dict[str, dict] = {}
    if PLAYERS_CACHE.exists():
        with open(PLAYERS_CACHE, "r", encoding="utf-8") as f:
            player_data = json.load(f)
        print(f"  → Loaded {len(player_data)} players from cache.")

    total = len(profile_urls)

    for idx, url in enumerate(profile_urls, 1):
        if url in player_data:
            continue
            
        print(f"  [{idx}/{total}] {url}")

        try:
            safe_get(driver, url)
        except RuntimeError:
            print(f"    ⚠ Skipping (could not load)")
            player_data[url] = {
                "is_left": 0,
                "Position_ID": 4,
            }
            continue

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # ── Foot ──
        is_left = 0
        foot_label = soup.find("span", class_="info-table__content--regular", string=re.compile(r"Foot"))
        if foot_label:
            foot_value = foot_label.find_next_sibling(
                "span", class_="info-table__content--bold"
            )
            if foot_value:
                foot_text = foot_value.get_text(strip=True).lower()
                is_left = 1 if foot_text == "left" else 0
        
        # Also try the data-header if info-table didn't work
        if not foot_label:
            for span in soup.select("span.info-table__content"):
                if "Foot:" in span.get_text():
                    next_span = span.find_next_sibling("span")
                    if next_span:
                        foot_text = next_span.get_text(strip=True).lower()
                        is_left = 1 if foot_text == "left" else 0
                    break

        # ── Position ──
        # Try the info-table first: "Position:" label → bold value
        raw_position = ""
        pos_label = soup.find("span", class_="info-table__content--regular", string=re.compile(r"Position"))
        if pos_label:
            pos_value = pos_label.find_next_sibling(
                "span", class_="info-table__content--bold"
            )
            if pos_value:
                raw_position = pos_value.get_text(strip=True)
        
        # Fallback: try the data-header
        if not raw_position:
            header_pos = soup.select_one("li.data-header__label span.data-header__content")
            for li in soup.select("li.data-header__label"):
                if "Position:" in li.get_text():
                    content = li.find("span", class_="data-header__content")
                    if content:
                        raw_position = content.get_text(strip=True)
                    break

        # Fallback: try the detail-position box
        if not raw_position:
            detail_pos = soup.select_one("dd.detail-position__position")
            if detail_pos:
                raw_position = detail_pos.get_text(strip=True)

        position_id = _parse_position_to_id(raw_position)

        player_data[url] = {
            "is_left": is_left,
            "Position_ID": position_id,
        }

        # ── Checkpoint Save ──
        if idx % 10 == 0 or idx == total:
            print(f"  💾 Saving Phase 3 checkpoint at step {idx}...")
            with open(PLAYERS_CACHE, "w", encoding="utf-8") as f:
                json.dump(player_data, f, ensure_ascii=False, indent=4)

    print(f"  ✓ Scraped {len(player_data)} unique player profiles.\n")
    return player_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    driver = create_driver()
    try:
        # ── Phase 1 ──────────────────────────────────────────────────────
        match_reports = phase1_get_match_report_urls(driver)
        if not match_reports:
            print("No match reports found. Exiting.")
            return

        # ── Phase 2 ──────────────────────────────────────────────────────
        print("═" * 70)
        print("Phase 2 · Parsing penalty shootouts from match reports")
        print("═" * 70)

        all_kicks: list[dict] = []
        processed_match_ids = set()
        
        if KICKS_CACHE.exists():
            with open(KICKS_CACHE, "r", encoding="utf-8") as f:
                all_kicks = json.load(f)
            processed_match_ids = {k["Match_ID"] for k in all_kicks}
            print(f"  → Loaded {len(processed_match_ids)} already processed matches from cache.")

        for idx, mr in enumerate(match_reports, 1):
            if mr['match_id'] in processed_match_ids:
                continue
                
            print(f"  [{idx}/{len(match_reports)}] Match {mr['match_id']}: {mr['url']}")
            kicks = phase2_parse_match_report(driver, mr["match_id"], mr["url"])
            print(f"    → {len(kicks)} penalties extracted")
            all_kicks.extend(kicks)
            
            # ── Checkpoint Save ──
            if idx % 10 == 0 or idx == len(match_reports):
                print(f"  💾 Saving Phase 2 checkpoint at match step {idx}...")
                with open(KICKS_CACHE, "w", encoding="utf-8") as f:
                    json.dump(all_kicks, f, ensure_ascii=False, indent=4)

        print(f"\n  ✓ Total penalties extracted: {len(all_kicks)}")

        if not all_kicks:
            print("No penalty data extracted. Exiting.")
            return

        # ── Phase 3 ──────────────────────────────────────────────────────
        unique_urls = list({k["Shooter_Profile_URL"] for k in all_kicks if k["Shooter_Profile_URL"]})
        player_profiles = phase3_scrape_player_profiles(driver, unique_urls)

    finally:
        driver.quit()

    # ── Merge & export ───────────────────────────────────────────────────
    print("═" * 70)
    print("Merging data and exporting CSV")
    print("═" * 70)

    rows: list[dict] = []
    for kick in all_kicks:
        profile = player_profiles.get(kick["Shooter_Profile_URL"], {})
        rows.append({
            "Match_ID": kick["Match_ID"],
            "Shooter_Name": kick["Shooter_Name"],
            "Penalty_Number": kick["Penalty_Number"],
            "Goal": kick["Goal"],
            "Elimination": kick["Elimination"],
            "is_left": profile.get("is_left", 0),
            "Position_ID": profile.get("Position_ID", 4), # Domyślnie FWD
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✓ Saved {len(df)} rows → {OUTPUT_FILE}")
    print(f"\n  Preview:\n{df.head(20).to_string(index=False)}")
    print(f"\n  Column summary:\n{df.describe().to_string()}")
    
    # ── Cleanup Checkpoints ──
    for cache_file in [PHASE1_CACHE, KICKS_CACHE, PLAYERS_CACHE]:
        if cache_file.exists():
            cache_file.unlink()


if __name__ == "__main__":
    main()