# UEFA Champions League Penalty Shootouts Dataset Description

## 1. Data Source & Origin
The dataset is programmatically scraped from **Transfermarkt** (historical data spanning up to the 2025/2026 UEFA Champions League seasons). It compiles a comprehensive history of every penalty shootout that took place in the tournament's knockout stages. The ingestion pipeline works in three distinct phases: gathering match reports, tracing execution structures of each penalty kick, and visiting individual player profiles to extract tactical/physical constraints.

## 2. Preprocessing & Derived Columns Logics
During the extraction pipeline, raw football statistics are converted into analytical predictors using the following deterministic engineering steps:
- **`Goal` Inference**: Calculated from the structural class elements inside the Match Report timelines (`sb-11m-tor` vs alternate miss classes).
- **`Elimination` Calculation**: Built dynamically using standard IFAB shootout logic. The loop tracks the cumulative score for home and away sides. A kick is assigned `Elimination = 1` if it is a "sudden-death" turn or if a miss/score mathematically ends the match instantly.
- **`Position_ID` Categorization**: Raw textual positions (e.g., Centre-Back, Attacking Midfield, Left Winger) are grouped into broad tactical domains and mapped directly to a single categorical index matching Stan's vector syntax constraints.

## 3. Dataset Schema & Data Dictionary

The engineered data is structured as an $N \times 7$ matrix (`UCL-Shootout.csv`) containing the following features:

| Column Name | Data Type | Value Constraints | Description |
| :--- | :--- | :--- | :--- |
| **`Match_ID`** | Integer | `Numeric ID` | Unique Transfermarkt index for the match report. Used for multi-level hierarchical grouping. |
| **`Shooter_Name`** | String | `Text` | Full registered name of the player taking the penalty kick. |
| **`Penalty_Number`** | Integer | `[1, 2, 3, ... 16+]` | Order of the kick inside the current shootout sequence. Simulates linear exhaustion/fatigue accumulation. |
| **`Goal`** | Binary | `0` or `1` | **Target Variable ($y$)**. `1` represents a successful goal scored; `0` represents a miss, post-hit, or goalkeeper save. |
| **`Elimination`** | Binary | `0` or `1` | Variable representing direct scoreline pressure. `1` if the current kick can immediately resolve the match outcome. |
| **`is_left`** | Binary | `0` or `1` | Physical bias parameter. `1` if the shooter's dominant kicking foot is left; `0` if right or ambidextrous. |
| **`Position_ID`** | Integer | `[1, 2, 3, 4]` | **Categorical Stan Factor**. Maps the tactical position of the player: <br>• **`1`** = Goalkeepers (GK) <br>• **`2`** = Defenders (CB, LB, RB, LWB, RWB, SW) <br>• **`3`** = Midfielders (DM, CM, AM, LM, RM) <br>• **`4`** = Forwards (CF, LW, RW, SS) |

## 4. Modeling Advantage of `Position_ID`
By flattening player roles into a singular, 1-indexed category (`Position_ID`), the data can be injected into the Stan file directly as an integer array:
```stan
array[N] int<lower=1, upper=4> position_id;