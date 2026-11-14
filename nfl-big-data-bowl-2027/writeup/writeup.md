# Run the Hoop, Win the Snap: a Combine drill that carries into NFL get-off

## The question

Every spring, defensive linemen run the "hoop" at the Combine: a figure-eight around two rings, bent low, as fast as they can. Scouts time it with a stopwatch and move on. Nobody asks the obvious follow-up: does the movement the drill measures show up on Sundays?

Tracking data lets us ask it for the first time. The Combine drills are now recorded at 10 Hz, and so is every snap of the rookies' first seasons. Our claim is simple. A rookie defensive lineman's tracked hoop time predicts how quickly he gets off the ball in games, and that link survives the situations he is used in. A second drill, the first second of a defensive tackle's 40-yard dash, predicts how often he pressures the quarterback.

For a team this means two things. Before the draft, the hoop gives a get-off expectation for players with no college tracking. During the season, a rookie's observed get-off can be compared with what his drill predicted, which separates athleticism from assignment.

## Data and definitions

We use 106 defensive-line rookies from the 2023–2025 drafts who have both a tracked hoop rep and in-game get-off; 118 DL rookies were tracked in all, 115 of them with get-off.

**Hoop duration** is the time from the first to the last tracked frame of the circuit. **Figure 4** shows what this number contains. It overlays the tracked paths and speed traces of the three fastest rookies (6.3–6.6 s, all edge rushers) and the three slowest (8.4–9.2 s, all interior tackles) among 109 DL with a hoop rep. The fast group holds 4–5 yd/s through the second loop. The slow group drops to about 2 yd/s there. The drill rewards keeping speed while bending, which is what a rusher does when he turns the corner. The two groups also differ by position, so the figure shows what the drill measures, not the effect size; every estimate below adjusts for position.

**Get-off** is the per-play `player_get_off`. At the player level we use a shrunken mean from a mixed model with a position fixed effect and a player random intercept. The intraclass correlation is small (0.060), so shrinkage matters. **Figure 5** shows it: most players keep 94% of their raw deviation (median shrink factor 0.94), and only those with fewer than about 30 plays are pulled strongly toward the mean (factor 0.63). Low-play players therefore cannot drive the result.

We fixed one drill feature in advance, hoop duration, and searched over no others. The same rule applies to the second drill below.

## Result 1: the drill predicts game get-off

**Figure 1** plots hoop time against shrunken get-off by draft year, with cross-validated R² beside it. The rank correlation is positive in all three years: +0.10 in 2023 (n=30), +0.49 in 2024 (34) and +0.33 in 2025 (39).

To test whether the hoop adds anything to what teams already have, we compare models in nested cross-validation. A base model with the stopwatch drills and draft slot explains nothing out of sample (R² −0.006). Hoop time alone reaches +0.082 on the raw target and +0.096 on the shrunken one. Added to the base, it gains +0.036 R², with a 95% bootstrap interval of [+0.007, +0.062].

At the play level, a mixed model with the nine base covariates gives +0.021 s of faster get-off per standard deviation of hoop time (0.446 s), with an interval of [+0.005, +0.037] and p=0.01.

## Result 2: it is not just usage

An obvious objection is that fast-hoop players are simply used differently. They are. Faster rookies line up more often as edge rushers (correlation −0.37 between hoop time and edge share), on third and fourth down (−0.32) and against shotgun (−0.25). These are all situations in which get-off is quicker anyway.

Adjusting for situation does absorb part of the effect. The play-level estimate falls to +0.013 s [−0.002, +0.027] (p=0.09), and 96% of player-cluster bootstraps stay positive.

The cleaner test is to look inside each situation. **Figure 2** shows the within-situation rank correlations with their intervals, and **Figure 3** shows the scatter for each. Overall the correlation is +0.37 [+0.20, +0.51]. It is +0.34 on pass plays, +0.25 on runs, +0.27 on shotgun passes, +0.32 on first and second down and +0.36 on third and fourth down. All five 95% intervals sit above zero. Usage explains some of the link but not the link itself.

## Result 3: a second drill, a second outcome

If drills carry into games, the pattern should repeat elsewhere. We fixed a second feature in advance: the tracked speed 1.0 s after the start of the 40-yard dash (v10). This is not the stopwatch 10-yard split. **Figure 6** shows why the difference matters. Two players with the same split can reach very different speeds in the first second. Across positions, v10 tracks game burst (cornerbacks ρ +0.41, defensive tackles +0.44) and adds to the split (+0.37 and +0.32 beyond it).

For interior rushers we link v10 to pressure. The sample is 41 DT/NT rookies on 10,600 pass-rush snaps, and the outcome is a pressure on the snap (a hurry, a sack or a recorded time to pressure). In the pre-registered model, a logistic GEE clustered by player with position, the stopwatch and draft-slot base and z(v10), the coefficient is +0.146 per SD, an odds ratio of 1.16 [+0.026, +0.265], p=0.017. It holds with the full base (+0.154), with situation controls (+0.157) and as a mixed model with a player intercept (+0.145).

**Figure 7** gives the plain view. By tercile of v10, pressure rates are 0.060 [0.045, 0.069] for the low group, 0.067 [0.041, 0.083] for the middle and 0.081 [0.065, 0.091] for the high. That is 1.35 times as many pressures per rush from low to high, and the rise is monotone.

We report the weaker side as it is. Resampling whole players gives 92% positive, with an interval of [−0.04, +0.30]. For all defensive linemen the effect is +0.075 (p=0.13), so this is a defensive-tackle result. The 2025 class runs the other way (−0.21 [−0.51, +0.10], 14 players). That class has no nose tackles, half the rush snaps and a narrow v10 range, and removing one player (Warren Brinson) moves it to +0.00. A random group of 14 players is that low 3.2% of the time, and the difference between years is not significant (Wald p=0.14). We therefore report the pooled estimate.

## Robustness and limits

The 2023 class is the weak year for the hoop: +0.07 [−0.27, +0.42] with 31 players. The cause is the position mix in a small year. Nine outside linebackers correlate at −0.43, and removing three of them with fast hoops and slow get-off gives +0.23. Play counts, the range of hoop times, the drill setup and usage do not explain it. Heterogeneity across years is not significant (Cochran's Q p=0.16), and the pooled correlation is +0.34.

The sample size is capped by design: only DL rookies with a tracked hoop qualify. We show an interval for every estimate.

Opponents are not tracked, so get-off measures the player's own first movement, not a win against a blocker. Pressure in Result 3 is the outcome that includes the blocker.

## How a team would use it

- **Draft board.** Use tracked hoop time as a get-off prior for defensive linemen with no college tracking. It adds out-of-sample R² that the stopwatch drills and draft slot do not have.
- **In season.** Compare a rookie's observed get-off with the expectation from his drill. A gap that persists inside the same situations points to technique or assignment, not athleticism.
- **Interior rushers.** For defensive tackles, the tracked first second of the 40, not the split, is a pressure prior: the top third pressures 1.35 times as often as the bottom third.

## Figures

1. Hoop time against shrunken get-off by draft year, with cross-validated R² for the base, hoop-only and combined models.
2. Within-situation rank correlations with 95% intervals.
3. Small multiples of hoop against get-off for each situation.
4. The drill itself: tracked figure-eight paths and speed traces of the three fastest (6.3–6.6 s, all edge) and three slowest (8.4–9.2 s, all DT) of 109 DL rookies. The fast group holds 4–5 yd/s through the second loop, where the slow group falls to about 2 yd/s. Fast and slow also split by position (edge against interior), which Results 1–3 adjust for.
5. The target: raw and shrunken get-off against plays for 115 players, showing that shrinkage is strong only below about 30 plays.
6. 40-yard tracked speed curves: two players with the same split and a different rise, and v10 against game burst for CB and DT.
7. DT pressure rate by v10 tercile with player-bootstrap intervals, and each player's shrunken rate against v10 by draft year with the GEE line.

## Code

A public notebook reproduces every number above from the competition files.
