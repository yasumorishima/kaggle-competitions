# Run the Hoop, Win the Snap: a Combine drill that carries into NFL get-off

*Working draft (outline + numbers). Limit: 2,000 words, fewer than 10 figures. Each section names the judging criterion it serves.*

## 1. The question (Football 30%)
- Teams time the DL "run the hoop" drill but rarely ask whether the tracked movement carries into games.
- Claim: a rookie DL's hoop time (from Combine tracking, not a stopwatch) predicts his in-game get-off, and the link survives the situations he is used in.
- Weekly use: a scouting flag before the draft, and a get-off expectation to compare against in-season tracking.

## 2. Data and definitions (Data Science 30%)
- 106 DL rookies (2023-2025 drafts) with both a tracked hoop rep and game get-off (`player_get_off`); 118 DL rookies in all, 115 with get-off.
- Hoop duration: time from the first to the last tracked frame of the circuit at 10 Hz (`analysis/dl_drills.py`).
- Get-off: per-play `player_get_off`; player level = shrunken mean from a MixedLM (position fixed effect, player random intercept; ICC 0.060).
- Only one drill feature (hoop duration) was fixed in advance; no feature search.

## 3. Result 1: the drill predicts game get-off (Figure 1: `figs/hoop_getoff.png`)
- Spearman by draft year: 2023 +0.10 (30), 2024 +0.49 (34), 2025 +0.33 (39).
- Nested CV R²: base (stopwatch + draft slot) -0.006, hoop alone +0.082 (raw target) / +0.096 (shrunken target); gain over base +0.036 [+0.007, +0.062].
- Play-level MixedLM, per 1 SD of hoop (0.446 s): +0.021 s [+0.005, +0.037] (p=0.01) after the 9 base covariates.

## 4. Result 2: it is not just usage (Figure 2: `figs/within_forest.png`; Figure 3: `figs/within_multiples.png`)
- Faster hoop rookies are used more in pass situations (EDGE -0.37, 3rd-4th down -0.32, shotgun -0.25), so adjusting for situation absorbs part of the effect: +0.013 s [-0.002, +0.027] (p=0.09; 96% of cluster bootstraps positive).
- Inside each situation the rank correlation remains: overall +0.37 [+0.20, +0.51]; pass +0.34, run +0.25, shotgun pass +0.27, downs 1-2 +0.32, downs 3-4 +0.36; all five 95% intervals above 0.

## 5. Result 3: a second drill, a second outcome — DT 40-yd rise and pressure (Figure 7: `figs/dt_pressure_tercile.png`)
- Drill feature fixed in advance: tracked speed 1.0 s after the 40-yd start (v10), not the stopwatch split.
- 41 DT/NT rookies, 10,600 pass-rush snaps; outcome = pressure on the snap (time to pressure, hurry or sack).
- Pre-registered model: logistic GEE clustered by player, position + stopwatch/draft-slot base + z(v10): β +0.146 per SD (odds ratio 1.16) [+0.026, +0.265], p=0.017. Holds with the full base (+0.154), with situation controls (+0.157) and as a GLMM with a player intercept (+0.145).
- Tercile view (player cluster bootstrap): low v10 0.060 [0.045, 0.069], mid 0.067 [0.041, 0.083], high 0.081 [0.065, 0.091] pressures per rush — 1.35x from low to high, monotone.
- Weaker side, reported as is: resampling whole players gives 92% positive ([-0.04, +0.30]); DL as a whole is +0.075 (p=0.13), so this is a DT result.
- 2025 runs the other way (-0.21 [-0.51, +0.10], 14 players): no NTs, half the rush snaps, a narrow v10 range, and one player (removing Warren Brinson moves it to +0.00). A random 14-player group is that low 3.2% of the time; the year difference is not significant (Wald p=0.14). We report the pooled estimate, as for 2023 in Result 1.

## 6. Robustness and limits (Data Science 30%)
- 2023 is weak: +0.07 [-0.27, +0.42] (31). Explained by position mix in a small year (OLB n=9 at -0.43; removing 3 fast-hoop/slow-get-off OLBs gives +0.23), not by play counts, range, drill setup or usage. Heterogeneity across years is not significant (Cochran Q p=0.16); we report the pooled estimate (Fisher +0.34).
- n is capped by design (DL rookies with a tracked hoop); intervals are shown everywhere.
- Opponents are not tracked, so get-off is the athlete's own first movement, not a win against a blocker.

## 7. How a team would use it (Football 30%)
- Draft board: hoop time as a get-off prior for DL with no college tracking.
- In season: compare a rookie's observed get-off with the drill-based expectation to separate athleticism from assignment.
- Interior rushers: the tracked 40-yd rise (not the split) as a pressure prior for DTs.

## 8. Figures (Visualization 20%) — planned set (max 9)
1. `figs/hoop_getoff.png` — hoop vs get-off scatter by year + CV R² bars.
2. `figs/within_forest.png` — within-situation correlations with intervals.
3. `figs/within_multiples.png` — small multiples by situation.
4. (todo) hoop trajectory examples: fast vs slow rookie, tracked path around the circle.
5. (todo) shrinkage: raw vs MixedLM player get-off.
6. `figs/forty_curve.png` — CB / DT 40-yd tracked speed curves: two players with the same stopwatch 10-yd split but a different rise, and drill v@1.0 s vs game burst (CB ρ +0.41, DT +0.44; beyond the split +0.37 / +0.32). Replaces the cut-fingerprint slot (pillar 1 reframed to trait translation on 10-31).
7. `figs/dt_pressure_tercile.png` — DT pressure rate by v10 tercile with player-bootstrap intervals; each player's shrunken rate vs v10 by draft year with the GEE line (Result 3).

## 9. Code
- Public notebook (December): reproduces every number above from the competition files (`analysis/*.py`).
