# Data

This folder holds the primary dataset used by the MeloMatch-AI notebooks.

| File | Samples | Features | Description |
|---|---|---|---|
| `mxmh_survey_results.csv` | 736 | 33 | Responses to the Music & Mental Health (MxMH) survey |

## Source

**Music & Mental Health Survey Results**: published on Kaggle by Catherine Rasgaitis. Responses were collected through a voluntary online form; timestamps in this file run from **27 Aug 2022 to 9 Nov 2022**. Please cite the original dataset if you reuse it.

## Data dictionary

### Respondent background

| Column | Type | Description |
|---|---|---|
| `Timestamp` | datetime | When the form was submitted |
| `Age` | numeric | Respondent's age (10–89; 1 missing) |
| `Primary streaming service` | categorical | Spotify, YouTube Music, Apple Music, Pandora, Other, or none |

### Listening habits

| Column | Type | Description |
|---|---|---|
| `Hours per day` | numeric | Hours spent listening to music per day (0–24) |
| `While working` | Yes/No | Listens to music while studying or working |
| `Instrumentalist` | Yes/No | Plays an instrument regularly |
| `Composer` | Yes/No | Composes music |
| `Fav genre` | categorical | Favorite genre (16 options) — **source of the model target** |
| `Exploratory` | Yes/No | Actively explores new artists/genres |
| `Foreign languages` | Yes/No | Regularly listens to music with lyrics in a language they aren't fluent in |
| `BPM` | numeric | Beats per minute of their favorite genre (107 missing; contains outliers) |
| `Frequency [<Genre>]` ×16 | ordinal | How often they listen to each genre: *Never, Rarely, Sometimes, Very frequently*. Genres: Classical, Country, EDM, Folk, Gospel, Hip hop, Jazz, K pop, Latin, Lofi, Metal, Pop, R&B, Rap, Rock, Video game music |

### Mental health (self-reported, 0–10 scale)

| Column | Description |
|---|---|
| `Anxiety` | Self-rated anxiety |
| `Depression` | Self-rated depression |
| `Insomnia` | Self-rated insomnia |
| `OCD` | Self-rated obsessive-compulsive symptoms |
| `Music effects` | Does music improve or worsen these conditions? *Improve* (542), *No effect* (169), *Worsen* (17), missing (8) |
| `Permissions` | Consent acknowledgment (always "I understand.") |

## Quick stats

**Favorite genre counts:** Rock 188 · Pop 114 · Metal 88 · Classical 53 · Video game music 44 · EDM 37 · R&B 35 · Hip hop 35 · Folk 30 · K pop 26 · Country 25 · Rap 22 · Jazz 20 · Lofi 10 · Gospel 6 · Latin 3

**Mental-health score means:** Anxiety 5.8 · Depression 4.8 · Insomnia 3.7 · OCD 2.6

**Primary streaming service:** Spotify 458 · YouTube Music 94 · none 71 · Apple Music 51 · Other 50 · Pandora 11

## Known data-quality issues

- **Missing values** in `Age`, `Primary streaming service`, `While working`, `Instrumentalist`, `Composer`, `Foreign languages`, `BPM` and `Music effects`. The notebooks impute numeric columns with the median and categorical columns with the most frequent value.
- **BPM outliers.** IQR bounds are 34–210 BPM; 11 rows fall outside, including 0, 4, 8, 20, 624 and **999,999,999**. These are flagged in the notebooks but not removed.
- **Rare genres.** Latin (3), Gospel (6) and Lofi (10) are too small to model alone, which is part of why genres are grouped into four clusters.
- **Self-selection.** Respondents skew young (median age 21) and Spotify-heavy, and opted in online, so the sample isn't representative of the general population.

## How the notebooks use this data

- **Target:** `Fav genre` → 4 clusters (High-Intensity, Melodic-Vocal, Calm-Reflective, Other/Residual).
- **Engineered feature:** `Genre_Diversity` = number of `Frequency [...]` columns answered "Sometimes" or "Very frequently".
- **Not used as features:** `Timestamp`, `Primary streaming service`, the individual `Frequency [...]` columns, `Music effects`, `Permissions`.

## Loading

```python
import pandas as pd
df = pd.read_csv("Data/mxmh_survey_results.csv")   # from the repo root
```
