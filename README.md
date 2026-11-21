# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arsheearanjbar-sudo/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arsheearanjbar-sudo/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A counts-based forecaster for Grand Theft Auto VI's 55 PlayStation 5 trophy earn rates on Exophase in week one. An earn rate is earners over players tracked, and both sides move in launch week — so the program models the two counts separately and divides at the target: the tracked population as a power law in time since release (with a stale-page surge prior when the history shows no growth), and each trophy's earners through a per-trophy earn hazard fitted to the case's reads. It writes five honest quantiles per trophy, with a spread that covers the fit's residuals, the population forecast's uncertainty, and the page-refresh lag of the answering read.

The second version calibrates the parts the first version left as worst-case bounds: the refresh lag enters the spread as the standard deviation of a uniform lag instead of its maximum, the flat-history population uncertainty is calibrated instead of round, and the stale-page surge prior is measured on the practice cases (0.27 per hour) instead of guessed. The hazard is fitted on the last 24 hours of reads only, so stale early reads cannot drag it. A tier hazard prior was tested and rejected: even a tiny one moves the medians far above every scenario's answer. Measured on practice scenarios built from the refresh the rules record (population 638 → 3,868, every rate but the 100% one lower), the mean pinball loss falls from 0.33 in the first version to 0.23 here.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` (Node 22, standard library only) reads every `<id>.in` in the input directory and writes `<id>.out` — one row per trophy, five quantiles in percent — to the output directory. It is deterministic: two runs write the same bytes, and it runs in well under a second per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=sh01ee1y&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
