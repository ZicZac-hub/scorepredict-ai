# Opening market odds and exact-score forecasts

A fixed market adjustment improved exact-score log loss on the same fixtures in both historical seasons. The two-match Double 5 results were similar to V8. In 2024–25 the market-adjusted set hit **19 of 165** dates on the V8-selected pairs, compared with **17 of 165** for V8. Selecting fixture pairs anew with market-adjusted probabilities yielded **16 of 165**. The same-pair difference was inconclusive, with a paired exact-test p-value of **0.79**. [Replay code](../market_odds_replay.py)

## Matched single-fixture results

| Season | Fixtures | V8 top 1 | Market top 1 | V8 top 5 | Market top 5 | V8 log loss | Market log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023–24 | 1,669 | 218 (13.06%) | 234 (14.02%) | 813 (48.71%) | 850 (50.93%) | 2.94568 | 2.90234 |
| 2024–25 | 1,674 | 204 (12.19%) | 208 (12.43%) | 813 (48.57%) | 835 (49.88%) | 2.92166 | 2.88664 |
| Combined hits | 3,343 | 422 (12.62%) | 442 (13.22%) | 1,626 (48.64%) | 1,685 (50.40%) | | |

The mean paired log-loss reduction was **0.04334** in 2023–24 and **0.03502** in 2024–25. The 95% source-date bootstrap intervals were **0.03542–0.05153** and **0.02586–0.04499**. Each of the 5,000 bootstrap replicates resampled whole source dates, keeping the fixtures and both models’ scores paired within each date. The intervals describe this historical fixture pool and assume independence between resampled date clusters.

## Five exact-score doubles

Each policy had one pair and five exact-score combinations per source date. A win required the regulation-time scores of both matches on the same combination. Both arms had the same eligible fixture pool. The matched comparison also applied the market-adjusted scores to the pair selected by V8.

| Season | Issued / source dates | Eligible pairs compared | V8 wins | Market wins on V8 pairs | Market wins on its own pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023–24 | 165 / 185 | 12,131 | 15 | 15 | 15 |
| 2024–25 | 165 / 183 | 12,497 | 17 | 19 | 16 |

On V8’s pairs, the models differed on eight winning dates in 2023–24, with four wins unique to each. In 2024–25 the market arm won eight dates that V8 missed and V8 won six dates that the market arm missed. The market-on-V8 set’s 2024–25 hit rate was **11.52%**, with a descriptive 95% Wilson interval of **7.50%–17.28%**. The calculation treats source dates as independent. Repeated teams and common model errors can introduce dependence.

In 2024–25, ranking pairs with the market grid increased average modeled five-row coverage from **7.94%** to **9.07%** while producing one fewer hit. The observed Double 5 results leave the 70% target unsupported.

## Sources and method

The test used English, German, Spanish, French, and Italian top-flight results from the repository’s pinned OpenFootball source manifest. V8 grids were reconstructed from up to 540 days of earlier dated fixtures. Every match on a source date stayed out of that date’s model inputs. Each retained forecast had at least two V8 components. Date-only match records support historical ordering and source-date slates. Live kickoff timestamps and Sydney matchdays were outside the source scope. [OpenFootball manifest](../data/target70_research/source_manifest.json)

The market arm used the public Football-Data.co.uk non-closing Pinnacle 1X2 prices (`PSH`, `PSD`, `PSA`) and over/under 2.5 prices (`P>2.5`, `P<2.5`). Decimal prices were inverted and normalized within each market. One fixed V22 adjustment applied strength `0.35` and a total-variation cap of `0.15` to every qualified V8 grid. The method and its constants were specified before scoring the replay. The CSV fixture joined by competition, source date, explicit home and away club aliases, and matching regulation-time score. Identity and score joined for all V8 grids in scope. Missing opening prices excluded 23 fixtures in 2023–24 and 18 in 2024–25. CSV prices lack individual observation timestamps, so this replay cannot certify availability at a particular live decision cutoff. Live V21 and V24 source packets may use different bookmakers and lead times. [Public odds archive](https://www.football-data.co.uk/data.php) · [Column definitions](https://www.football-data.co.uk/notes.txt) · [V22 adjustment](../v22_evidence_models.py)

## Live scope

Home and Daily Predictions use the adjustment by default for fixtures in the five replayed leagues when API-Football supplies the fixture, V8 has at least two components, and the repository holds qualifying pre-match 1X2 and over/under 2.5 prices. The forecast uses the original V8 grid whenever qualifying prices are unavailable. Serving the adjustment uses zero extra odds requests. Double 5 retains its separate prospective policy and verification. [Live source qualification](../archived_market_forecast.py)

## Reproduce

Use the repository’s Python requirements. These commands write source data and the JSON report to a local cache outside the Git checkout. The OpenFootball manifest and market script check the SHA-256 of every source file. Cached files support offline replay when `--download-odds` is omitted. [Replay code and pinned odds checksums](../market_odds_replay.py)

```bash
python target70_data.py \
  --revision e6744429ee395bc86f247348c6184bb08d4eb361 \
  --cache-dir /tmp/scorepredict-openfootball \
  --manifest /tmp/scorepredict-openfootball/source-manifest.json

python market_odds_replay.py \
  --openfootball-cache /tmp/scorepredict-openfootball \
  --odds-directory /tmp/scorepredict-odds \
  --download-odds > /tmp/scorepredict-market-replay.json
```
