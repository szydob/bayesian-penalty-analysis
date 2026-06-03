# ⚽ Bayesian Analysis of Penalty Shootouts

A research project utilizing Bayesian modeling to analyze the success rate of penalty kicks in professional football. The primary goal is to verify the impact of psychological pressure (e.g., sudden-death elimination kicks), fatigue (shootout sequence number), and tactical/physical constraints on the probability of scoring a goal.

## Data Sources

The project relies on two main datasets:

### 1. World Cup Dataset (Baseline)
The baseline data originates from a public dataset available on Kaggle: **[World Cup Penalty Shootouts](https://www.kaggle.com/datasets/pablollanderos33/world-cup-penalty-shootouts)**. 
This dataset contains a clean, detailed historical record of penalty kicks from the FIFA World Cup. A full description of the raw data structure and dictionaries can be found directly on the source page.

### 2. UEFA Champions League (UCL) Dataset - Custom Scraper
To test our models on more elite and recent club data, we built a custom three-phase scraper (`src/ucl_scraper.py`) using `Selenium` and `BeautifulSoup`. The script automatically extracts historical data from **Transfermarkt**.

**Scraper Workflow Outline:**
1. **Match Report Extraction:** Scrapes the complete list of URLs for match reports containing penalty shootouts in UEFA Champions League history.
2. **Shot Parsing:** Iterates through each match to extract the shooter, the outcome of the kick, and dynamically calculates the pressure variable (`Elimination` — whether the current kick could immediately decide the match outcome via sudden-death or mathematical elimination).
3. **Profile Enrichment:** Visits each shooter's profile to extract their preferred foot and tactical position. Positions are mapped on the fly into a 4-level categorical variable (`Position_ID`: 1=GK, 2=DEF, 3=MID, 4=FWD), optimized for vectorized indexing in Stan.

*The script is equipped with advanced anti-scraping bypass mechanisms and a robust JSON checkpointing system that saves the state every 10 iterations (ensuring resilience against network drops and memory leaks).*

## How to Run
This project uses uv, an extremely fast Python package manager.
1. Clone the repository:

```bash
git clone git@github.com:szydob/bayesian-penalty-analysis.git
cd bayesian-penalty-analysis
```

2. Create a virtual environment and sync dependencies:
```bash
uv sync
```

3. Scrape the UCL dataset:
```bash
uv run core/web-scraping/scrape_ucl_penalties.py
```

## Modeling Methodology
> TODO