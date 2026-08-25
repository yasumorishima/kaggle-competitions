# Kaggle Competitions

Kaggle Notebooks Expert. 14 Bronze Notebook Medals + active competition participation.

**Note:** Notebook Medals are earned through community votes on shared notebooks - NOT competition ranking medals.

---

## ☁️ GitHub-Driven Workflow (GitHub Actions + Kaggle API)

Manage all notebook code in GitHub — version control, diff, CI, and auto-deploy to Kaggle via `git push`.

```
Edit in VSCode / any editor → git push → GitHub Actions / RPi5 → kaggle kernels push → Auto-submit via API
```

### Why GitHub Instead of Kaggle's Browser Editor?

- **Version control**: Full git history, branching, diff for every change
- **Editor freedom**: Use VSCode, Vim, or any editor instead of Kaggle's browser UI
- **CI/CD**: GitHub Actions automates deployment to Kaggle — no manual upload
- **Secrets management**: API keys stored in GitHub Secrets, not local files
- **Multi-device**: Push from any machine with git

### How It Works

1. Edit `.ipynb` in your preferred editor
2. `git push` to this repository
3. Trigger: `gh workflow run kaggle-push.yml -f notebook_dir=<dir>`
4. GitHub Actions runs [`kaggle kernels push`](.github/workflows/kaggle-push.yml) to upload the notebook
5. Auto-submit via Kaggle API (`competition_submit` with kernel output download)

### RPi5 Self-Hosted Runner

For long-running notebooks that exceed GitHub Actions' 90-min timeout, RPi5 runs `kaggle-submit.sh` with:
- No timeout limit (6h polling)
- GPU → CPU auto-fallback (any GPU failure triggers CPU retry, not just quota errors)
- Auto-submission via `kaggle.api.competition_submit()` (download output CSV → submit)
- Score polling → W&B recording → Discord notification

### Key Findings

- **`enable_internet: false`** is required for code competition submissions — Internet ON prevents the notebook from being eligible
- **`competition_sources`** mounts data at `/kaggle/input/competitions/<slug>/` (not `/kaggle/input/<slug>/`)
- **Submission differs by competition type (verified 2026-06 on an active comp):** for **code competitions** the API path is blocked both ways — `CreateCodeSubmission` returns **403** and file submission (`kaggle competitions submit -f`) returns **400** (the upload is rejected and leaves no entry). Code-competition submission must go through the notebook **"Submit to Competition"** UI. File submission (`submit -f`) works only for **regular** competitions (downloadable test + `sample_submission`, not flagged "code competition"), e.g. S6E6 / BirdCLEF.

**Blog post:** [DEV.to](https://dev.to/yasumorishima/kaggle-code-competitions-without-a-local-gpu-github-actions-kaggle-api-cloud-workflow-m3)

---

## 🔬 Experiment Management (EXP + child-exp)

### Credit & Origin

The experiment management methodology (EXP + child-exp structure, role division, EXP_SUMMARY.md, CLAUDE_COMP.md) is inspired by [chimanさんの記事](https://zenn.dev/chiman/articles/b233cc808d6af3) — a write-up on winning a Kaggle gold medal using Claude Code / Codex.

The following parts are our own design, built to work in a GPU-less local environment (Celeron N4500 / 4GB RAM):

| Component | Origin |
|---|---|
| EXP + child-exp directory structure | chimanさんの記事 |
| Role division (ideas = human, implementation = AI) | chimanさんの記事 |
| EXP_SUMMARY.md (experiment history as AI guardrail) | chimanさんの記事 |
| CLAUDE_COMP.md (competition-specific AI guardrails) | chimanさんの記事 |
| RPi5 xrdp remote desktop (session persists on disconnect) | 独自設計 |
| xdotool keepalive (systemd, 30min interval) | 独自設計 |
| Google Drive for Desktop ↔ Colab sync | 独自設計 |
| Colab file monitor notebook (auto-detect & run) | 独自設計 |

### Architecture

```
[Local PC]                  [RPi5 (xrdp)]               [Google Colab (Free)]
Claude Code                 Remote Desktop session       Experiment Runner v4
  ↓ Write config/code        ↓ RDP for Colab setup        ↓ Auto-run train.py
  ↓                          ↓ Session persists on DC      ↓
Google Drive (for Desktop) ←――――――――――――――――――→ Google Drive (mount)
  EXP/config/child-exp005.yaml                          Detect new config → execute
  EXP/output/child-exp005/result.json                   Save results to Drive
```

- **RPi5 xrdp** provides remote desktop for Colab setup; session persists after disconnect
- **xdotool keepalive** (systemd service) sends keystrokes every 30 min to prevent Colab idle timeout
- **Claude Code** writes experiment configs to Google Drive
- **Colab** auto-detects new configs and runs `train.py`
- **[colab-mcp](https://github.com/googlecolab/colab-mcp)** enables direct Colab GPU interaction from Claude Code via MCP. PC起動中かつColabノートブックをブラウザで開いている間のみ利用可能（WebSocket接続のため）
- **Kaggle kernels** used only for final submission

### Directory Structure

```
<comp-slug>/
├── CLAUDE_COMP.md              # Competition-specific guardrails for AI
├── EXP_SUMMARY.md              # Experiment history (AI memory)
├── docs/Idea_Research/         # Hypothesis memos, Deep Research results
├── EXP/
│   ├── EXP001/
│   │   ├── train.py
│   │   ├── config/
│   │   │   ├── child-exp000.yaml  # Baseline
│   │   │   ├── child-exp001.yaml  # + feature engineering
│   │   │   └── child-exp002.yaml  # + loss change
│   │   └── output/
│   │       ├── child-exp000/
│   │       │   ├── oof.csv
│   │       │   └── result.json
│   │       └── ...
│   └── EXP002/                 # New EXP when pipeline changes significantly
└── submit/                     # Final Kaggle kernel for submission
```

### Role Division

| Human | AI (Claude Code) |
|---|---|
| Hypotheses & ideas | Implementation (train.py, features) |
| CV design decisions | OOF error analysis & visualization |
| Interpret results → next action | Config generation & experiment tracking |
| Domain knowledge | Notebook/Discussion summarization |

### Key Principles

- **Never ask AI "improve the score"** — provide specific hypotheses
- **EXP_SUMMARY.md is a guardrail**, not a strategy generator
- **1 competition, 50+ experiments** — depth over breadth
- **OOF analysis by AI, next action by human**

### Google Drive ↔ GitHub Actions Integration

Experiment results on Google Drive are synced to GitHub Actions for W&B recording and Kaggle/SIGNATE/DrivenData submission.

```
Google Drive (EXP results)
  ↓ Google Drive API (Service Account)
GitHub Actions
  ├── EXP W&B Sync      → Record cv_score, config to W&B
  └── EXP to Kaggle     → Fetch best model → Push to Kaggle kernel
```

#### Setup (for your own fork)

1. Create a Google Cloud project (free) and enable the **Google Drive API**
2. Create a **Service Account** → download JSON key
3. Share your Drive experiment folder with the service account email (Viewer permission)
4. Add GitHub Secrets to your repository:
   - `GOOGLE_SERVICE_ACCOUNT_KEY`: Service account JSON key contents
   - `DRIVE_SHARED_FOLDER_ID`: Google Drive folder ID (from the folder URL)

No billing required — Google Drive API is free within standard quotas.

#### Usage

```bash
# Sync experiment results to W&B
gh workflow run "EXP W&B Sync" \
  -f comp=s6e3-churn -f exp=EXP001 -f memo="sync all results"

# Submit best experiment to Kaggle
gh workflow run "EXP to Kaggle Submit" \
  -f comp=s6e3-churn -f exp=EXP001 -f child=child-exp005 \
  -f notebook_dir=playground-series-s6e3-work \
  -f kernel_id=yasunorim/s6e3-churn-optuna-stacking-work \
  -f memo="best config submit"
```

---

## 🏆 Competition Results

### Kaggriculture (Active)

**Competition:** [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) | **Deadline:** 2026-09-30 | **Prize:** $50,000 | **Metric:** ladder rating

Two players run neighbouring farms for 30 days of 24 turns, buying land, hiring hands, sowing, tending livestock and selling into a shared market. No GPU is involved, so a whole season simulates in 3–4 seconds and the bottleneck is experiment design rather than compute. Code lives in [`kaggriculture/`](kaggriculture); every episode runs in GitHub Actions (`Kaggriculture Eval` / `Analyze` / `Sweep` / `Record` / `Optimize` / `Schedule`), never locally.

**The score is a rating, not money.** Downloading the leaderboard shows the field spans −129 to 3,172, so an agent's own final cash says nothing about where it stands; only paired head-to-head play does.

#### What the environment actually pays for (read off the published environment source, then confirmed in play)

- **The town's drain rate is exactly computable.** Each unlocked shop instance consumes 1 unit of each product it wants every 4 turns (2× for single-product shops), plus 1 of everything per day from the town centre. Demand outruns what one farm can grow, so the live regime is scarcity: over a measured season strawberry averaged $222 against a $120 base and milk $290 against $160, and prices *rose* all season. Only genuinely oversupplied goods crash.
- **The published per-tile rates are what perfect husbandry pays, not a typical yield.** A cow's care bonus accumulates and is released on its production day, so a fed-and-cared cow yields 3 milk per 2 days — exactly the tabular 1.50 — while an uncared one yields 0.50. Sheep and geese work out the same way. Measured yields of 0.71–0.76 are husbandry completeness, not a hidden penalty.
- **Labour is priced on a Fibonacci curve.** The n-th hire of a day costs `fib(n)`: three hands cost $4 a day, eight cost $54, twelve cost $376 and fourteen cost $986. Twelve is the knee, and hands cannot be substituted for land.
- **Two bugs found by putting our own action breakdown beside a strong opponent's.** `PLANT` is validated atomically — if the turn's requests for a crop exceed the seeds held, the environment drops *every* request for that crop, so a dozen hands aiming at one seed sowed nothing at all (894 `PLANT` actions against 243 harvests). And `SELL` draws only from the shed, so produce still in a hand's pack cannot be sold; shed runs had to be scheduled explicitly. Fixing both took the agent from 28,170 to 40,796 against the same opponent.

#### Measuring instead of guessing (`kaggriculture/sim/`)

| Tool | What it answers |
|---|---|
| `evaluate.py` | Paired-seed head-to-head — every seed played twice with sides swapped, so the season draw cancels. Reports the delta's 95% CI, and refuses to score a side the environment stopped |
| `sweep.py` | Many settings over one identical seed list, each compared with the reference **within** the same season draw. A change is adopted only when that paired CI clears 0 |
| `knob_bite.py` | Before spending a sweep: does this setting change what the agent returns, on twelve representative boards? A setting no board reaches is reported as such, so the gap gets a board built for it rather than a sweep spent on it |
| `record.py` | Records an episode as an action list and emits an agent that replays it — plus the plan's own labour density |
| `route_shape.py` / `layout.py` | How many jobs a plan does per stop and how far it walks between them; and, by integrating spawns and moves, which tiles it works. **Read with the caveat below** — it counts idle-filling actions as work |
| `analyze.py` | One episode broken down: daily cash, farm composition, per-product sale prices, action mix, husbandry rates |
| `plan_economy.py` | What economy a plan runs, read off its action list alone: hires, purchases, plantings and sales per day — and how much of the herd's daily care actually landed |
| `optimize.py` | Hill-climbs the 720-step action list itself, with held-out seeds scored throughout |
| `schedule.py` / `optimize_schedule.py` | Hill-climbs the *capital calendar* — what the farm owns, day by day — and leaves the field work to the policy |
| `sched_bite.py` / `test_sched_ops.py` | Whether the calendar reaches the orders at all, and whether every mutation operator fires and actually changes the calendar. Also the acceptance rule itself, with the environment replaced by arithmetic — a child that is lucky only on the screening seeds must be thrown out, a child too wide to measure must never reach the confirmation |
| `noise_band.py` | How wide the ruler is, one operator at a time, and what any acceptance rule would have been worth on episodes already played. Storing the matrix means a rule can be re-argued for nothing: three proposed changes were settled this way without spending a single episode |

#### Findings

- **A season's variance dwarfs the effects being chased, so unpaired means are useless here.** The sweep originally reported each variant's own spread; switching to a within-seed paired comparison is what made a ±2,000 effect readable at all. Two changes that topped a sweep (+5,364) failed to replicate on fresh seeds (+2,353 ± 4,346) and were dropped.
- **Bundling knobs hides good changes inside bad ones.** A three-part opening change measured −4,971 and was nearly discarded; split apart, it was +17,928 and −3,399 mixed together.
- **A knob that does nothing looks exactly like a knob that does nothing useful.** Four settings came back from 40-game sweeps with a delta of precisely zero — the same actions every step. `knob_bite.py` now catches that in a second, and immediately found that one weight was inert at 1.0 because it cancelled the distance discount exactly, only biting from 3.0.
- **A dead agent looks exactly like a weak one.** A replay scoring precisely $0 with the opponent's score identical to the decimal across every game is not a bad farm; it is an agent the environment stopped. Two separate causes hid behind that signature — and the second was an off-by-one: `env.steps[t].action` is the action that *produced* state t, so recording it at index t replays every action one turn late, which is fatal rather than merely worse.
- **The strongest published agents are not policies at all.** They ship a fixed 720-step action list, compressed into the notebook, and replay it with a thin repair layer for the weeds and roster the season randomises. Decoding one gives its whole plan: where its animals sit, what it buys on each day, when it sells.
- **The instrument was measuring the wrong thing, and six conclusions came off it.** This agent appeared to do 1.33 jobs per stop against a strong plan's 2.07, and to walk 60% of its actions against their 43%. Both numbers counted any non-move, non-`PASS` action as work — and a sweep offering a second daily `CARE` visit collapsed by 60,493, which proved the second visit is worth nothing. 655 of the opponent's 967 `CARE` actions could not have counted: they fill idle hands the way this farm fills them with `PASS`. Removing the filler, care delivered is 100% of their herd's animal-days against 62% of this one's, on nearly the same total — spread over 27 animals instead of 12. Six levers had been drawn off the contaminated figures and every one came back tie or worse. Check what a number counts before making it a target.
- **A recording of a reactive policy is welded to its season.** Hill-climbing the 720-step action list ran twice for four hours. On average money it raised the training seeds 9.3% while the held-out set *fell*; on wins it moved training from 7 to 8 of 12 while the held-out count sat at 1 for all 137 generations, and the result lost 24 games out of 24. The same list earns 60k on the seeds it was climbed on and 25k on fresh ones, while a published list scores 156k and 171k on the same two sets. Most of those 720 steps encode which tile happened to need water at hour 9.
- **Capital is the part that transfers.** What separates the two farms is what they own, not where they stand: 12 animals against 27, wheat net +290 against net −55, quadrants bought on days 6 and 10 against day 8 alone. "Carry eight cows and buy the second quadrant on day 6" is as true of the next season as of this one, so the calendar became the genome and the policy kept the field work. Seventeen generations took it to 12 wins out of 12 on both the training *and* the held-out seeds — against 137 generations of the action list that moved the held-out count not at all.
- **Beating your own previous agent is a floor, not a goal.** That calendar beat the agent on the leaderboard 91.7% of the time, +15,381 ± 3,008 over 24 paired games. Put against the top published plan it lost 24 out of 24, 48,747 against 130,842. A weak opponent does not compete for the same market, so the economy itself is different — optimising against one is optimising a different game.
- **A knob's verdict is conditional on the capital it was measured under.** Giving a unit first refusal on the tile it is standing on measured +662 — a tie — and was shelved. With the calendar holding the herd at twelve instead of twenty-seven it is +16,529 ± 5,897, replicated at +11,683 ± 6,829 on fresh seeds. Nine other settings were refuted in the same series, including one that measured +12,101 alone and vanished to +653 once the first was on: the same bottleneck reached by two roads, not two gains.
- **A losing objective has no gradient.** Climbing `wins` against the top plan ran 135 generations and never left 0 of 12, because `wins` is flat at zero while you lose everything and the tie-break margin it falls back on is dominated by the *opponent's* score, which this farm barely moves. Five hours of walking on noise. While you cannot win a single game, the thing to climb is your own money.
- **An accept can be confirmed and still be worthless.** Screening candidates cheaply and re-measuring the survivor on episodes it was not chosen on removes the winner's curse — for a candidate named in advance. Accepting *on the strength of* that confirmation names it afterwards, and the bias walks back in. Fifty-one generations accepted eleven edits whose confirmations summed to +56,394; the calendar it started from and the calendar it produced, played over 96 fresh episodes, came out **−3,419 ± 3,691 apart**. Four hours spent going backwards. Raising the significance bar does not repair this: it converges on accepting nothing, which is merely a cheaper way to stand still.
- **The fix was to refuse to judge what cannot be judged.** Spread is a property of the child, not of the search: a labour-dial edit came back +263 with a standard deviation of 1,119, while a rescheduled quadrant swung 15,000 an episode. Twelve screening episodes cannot separate +250 from noise at that width, so a wide child reaching the confirmation is a coin, and a coin that lands well is accepted. Dropping children whose screening spread exceeds a cut turned the true value of an accept from −129 to **+209**, and of a generation from −28 to **+105**; the same climb went from −3,419 to +589 against its own starting calendar. Two further proposals — requiring the screen's own floor to be positive, and measuring fewer children more thickly — were replayed against the stored matrix and both came back worse, so neither was written.
- **An even draw over ten operators is an even spend over effects that are not even.** Measured one at a time, the labour and herd dials returned +560 and +916 a child while two others returned −4,615 and −4,761 — and the big-swinging operators were the destructive ones, so an argmax over a noisy screen was selecting for exactly the wrong thing. Worse, thirteen labour dials shared one operator among ten, which put a named dial at one draw in a hundred and thirty: the one dial the measurement liked was, in practice, never drawn. Weighting the draw by what each operator has been worth opened that family — the calendar went from two dials off their default to six.
- **The opponent's shape is not the opponent's edge.** Their farm runs 27 animals to this one's 15, and the obvious reading is that the herd is the difference. Scored over 96 fresh episodes, a calendar carrying 27 animals is **−14,022** against the incumbent and their own capital blueprint **−14,839**: more animals spread the same labour thinner, and care per head collapses. Care per head really is the mechanism — and it still cannot be bought with a priority multiplier, because a percentage weight only reorders work that is already offered. Pushed to 50, 150, 200, 300 and 400 percent, every setting lost between 2,600 and 7,600 against leaving it alone.
- **A breakdown is a diagnosis, not a prescription.** Putting the two farms' seasons side by side showed the calendar was not being followed — four sheep asked for on day 3 arrived on day 15, and the herd starved back from four to zero. Every fix drawn from that reading failed: reserving the herd's cost from the seed budget, −4,878; buying deeper feed, −11,838; conceding less of the market to the opponent's supply, −5,648. Buying the herd early is what *their* farm is built for; eighteen tiles of seed is not a bug here. That is four hypotheses-from-statistics refuted in a row on this problem.
- **Six weeks of tuning never once looked at the thing being scored.** Kaggle's episode service returns, per episode, both agents' final money, both teams' ratings before and after, and who the opponent was; `ListEpisodes` rejects a `teamId` filter and accepts a `submissionId`, and the submission IDs are in the v1 submissions endpoint rather than the CLI. Pulled for three submissions it is 117 real games, and it contradicted the number the whole project had been steering by. Against the top published plan this farm is 75,994 behind. Against the ladder it is **4,453 behind, t = −1.50** — the opponents cluster either side of it. Re-scoring those same 117 games with this farm's money shifted by a constant prices the gap directly: **+1,000 moves the win rate from 47.0% to 52.1%**, +10,000 to 62.4%. The improvement that had been dismissed as too small to matter was worth five points of win rate.
- **The top of this leaderboard is not richer than that; it is more consistently rich.** Crawling the episode graph outward from our own games — each episode names its opponents' submissions, so the search can climb toward higher ratings — gives 5,064 agent-games up to rating 3,196. Median money by band: 50,025 below 600, 74,049 at 600–700, and **87,782 above 2,200**. From rating 600 to the very top, money moves 14k. So the target is not "double it": bronze sits at rating 1,966, between measured bands of 83,425 and 89,316, which is **+23,000 from here**. The cross-sectional fit predicts this team's own money to within 2% of what it actually earns, which is the check that the instrument reads true.
- **Sparring against the strongest opponent in the field shrinks the board by a third.** Same agent, same 48 seeds, both sides, only the opponent swapped: **74,262** against a median-strength opponent, **52,133** against the top published plan, which itself takes 128,127. Every rejection in this project was measured in that second world. The ladder data alone would have said otherwise — a regression of the two rewards puts the cost of an opponent's dollar at five cents — but that regression is confounded by how rich a season is, and the controlled swap says twenty-four. The distortion is real; it did not, however, reverse anything. Seven settings that loosen how much the farm is allowed to produce were re-measured against the median-strength opponent and not one changed sign: raising the crop ceilings costs **−10,267**, and conceding nothing to the opponent's supply **−2,091**. The town's demand ceiling was already set correctly.
- **A setting that bites is not a setting that acts.** `knob_bite` reports idle-land planting as changing the agent's output, and it does — on the turn it is asked about. Over 96 full seasons the same setting moved the money by *exactly zero*, and so did the late-season wheat conversion. Reading the planting code for the cause found one shared to both: strawberry is allotted last and takes the remaining budget *as its cap*, so every rule downstream of it runs with nothing left to spend. Neither setting can do anything until the rule that respects sowing deadlines frees that budget — and that rule, tested alone, has nothing to spend the budget on and reads as a tie. Three settings that only work as a set had each been measured alone and each written off. The one-turn question and the whole-season question have different answers, and only the second one is the money.
- **More land is not more farm.** The calendar buys a second quadrant on day 7 and stops, where the top published plan works three, and that was the last structural difference left after the climb had independently found the opponent's hiring curve. Forcing a third from day 8, 10, 12 or 16 costs **−4,849 to −5,741** — and days 8, 10 and 12 return the identical figure, because the farm cannot afford the quadrant until well after all three, so the date was never the variable. Forty-two tiles already stand idle at season's end; capacity was not the constraint, and the cash the quadrant costs was.
- **The hands were being served in the wrong order, and three of this agent's knobs existed to paper over it.** Units are asked in roster order and each claims the tile it picked, so the farmer and the low-numbered hands take the best tiles on the whole farm before the hand standing next to one is asked; the loser then walks. Scoring every (unit, job) pair first and filling from the best pair down uses the same information in an order that does not let unit 0 outbid someone already standing there. Measured against a median-strength opponent over three disjoint seed bands of 96 games: **+6,098 / +5,551 / +4,554, pooled +5,441 ± 1,461 (t = 7.30)**. The rule alone is only a tie — the gain needs stickiness turned off with it, because stickiness was a patch on roster order and now blocks the better pair (+848 with it, +2,657 without). `stand_first` is the same kind of patch, and turning it off too swings between +6,908 and +3,248 across bands, so it stays.
- **The metric moved; the mechanism had not.** The reason given for that change was walking — 56% of 5,910 actions, against care reaching only 216 of 304 animal-days. So the season was broken down again after the win, and walking was **still 56%**. What had changed was sowing: 211 plantings had become 112, because every pair is scored before any is taken, so all units are offered the same crop and the ones past the seed count fall through to something that is not sowing at all. Splitting the two apart: the assignment rule is worth **+2,657** with sowing restored, and not sowing a second-choice crop is worth **+3,441** on top. Both real, neither the stated reason. And it is not "plant less" — the control that simply lowers the crop ceilings is **−1,243**; it is that this particular sowing was worth less than the watering and the care it displaced.
- **A calendar's verdict is conditional on the policy it was climbed under.** Re-pointing the capital-calendar climb from the top published plan to a median-strength opponent turned 45 generations from +589 (t = 0.46) into **+8,579 (t = 2.40)**, and the labour dials moved for the first time in the project. Played under the *new* assignment rule on 96 fresh episodes, that calendar is **4,244 ± 3,603 behind** the one it was supposed to replace. The same conditionality that makes a knob's verdict depend on the capital it was measured under makes a calendar's depend on the policy it was climbed under — so the climb has to be re-run, not the result adopted.
- **It was re-run, and the answer was nothing.** 55 generations, 27 accepts, 3,187 episodes on the corrected acceptance rule and against the corrected opponent: over 96 fresh episodes the calendar it produced is **+529 ± 2,696** against the one it started from. Its own held-out set had been saying so all along — 74,957, then 78,840, then 79,845, then 74,620, then 77,386, a walk whose middle reading is *worse* than its start. That is three calendar climbs in a row returning nothing, and the acceptance rule has now been fixed twice, so the remaining suspect is the twelve held-out episodes the whole climb is steered by. One useful byproduct: offered both the incumbent calendar and the previous climb's output as starting points, it scored them 72,195 and 67,027 and chose the incumbent — reproducing, from inside the search, the finding that a calendar climbed under the old policy loses under the new one.
- **A knob that does not bite in a single turn can still be worth 4,313.** Lowering how steeply travel discounts a job, from 1.0 to 0.7, was measured on three disjoint bands of 96 games: **+4,626 ± 2,144**, **+5,438 ± 2,753**, **+2,411 ± 2,964**, inverse-variance **+4,313 ± 1,469**. `knob_bite` cannot tell 0.7 from 1.0 on any of its boards, because the top-ranked job on a single turn is rarely the one whose ranking flips; the flips accumulate over 720 steps and eleven hands. The value is a located optimum rather than a maximum re-picked out of noise — 0.5 and 0.9 sit about 3,000 below it in both bands that measured them, and it is not a licence to ignore distance either, since 0.2 costs 9,549 and 0.0 costs **45,939**, the agent simply thrashing. It never settled before because it had only ever been measured under the roster assignment rule, bundled with another change. Under global matching this number decides whether the matcher may send a hand across the farm for a big job, which is a different question.
- **The gap to the top is not land, and it is not care; it is what is standing on the land.** One season put side by side with the top published plan: 11–17 strawberry plants against their 36, 110 units sold against 243, in a town holding the price at $238 for both farms — three ice-cream shops and two smoothie shops, and the market not close to saturated. So the obvious hypothesis was that the third quadrant had been rejected only because it was being filled with wheat, and that land and crop had to be tested as a set. Tested as a 2×2 that is exactly backwards: the third quadrant alone is **−6,364** (an independent reproduction of the earlier −5,741) and the two together are **−11,659**, while cutting wheat and tomato to make room destroys the strawberry gain outright — wheat here is not a cash crop but the herd's feed, and its floor is computed from the herd. Raising the care weight, the other visible gap in the breakdown — 967 of their actions against 321 of ours — is **−574**, a tie. **Raising the strawberry target did measure +3,750 ± 3,101, and then failed to replicate: −340 ± 2,172 on a fresh band once the travel discount above was adopted, with no dose-response behind it — 1.5× is −2,359 and 4× is −8,185.** So none of the four visible differences has survived as a cause, and the one that looked like one was a sweep maximum. The side-by-side told us where the money is; it has now been wrong about why four times in a row.
- **The order the budget is spent in looked like the cause, and is not.** The planner hands a fixed tile budget to each product in a written sequence and whatever comes last receives the remainder. Strawberry, at $250 the dearest thing the farm can grow, was written last; tomato, at $67 and returning 29 units for $1,617 across a whole season, took its cap first. Sorting the cash crops by what a tile of each is worth at the live price costs **−4,663**; putting strawberry first and dropping tomato entirely costs **−12,058**. The hand-written sequence was not an accident — the cheap crops take small caps and leave early, and giving the dearest one first refusal starves them without buying anything, because its own ceiling is the town's demand rather than the budget. The order stays a setting, with the written sequence as the default.
- **The offline ruler reads true on the ladder.** The assignment rewrite measured +5,441 in sparring; submitted, the team's rating moved from 594.7 to **669.5** within four hours, 3,924th to 3,477th of 6,218, and had not finished converging. The episode service is the wrong tool for reading that: it answers a 429 with `Retry-After: 30` but stays blocked for far longer, and rejected calls appear to spend quota of their own, so backing off in small steps never recovers. The public leaderboard CSV gives the same number in one request.
- **A third of a season was invisible to the instrument, and the fix moved the target.** Reconstructing sales from what leaves the shed under-counts whoever harvests most, because a harvest refills in the same step a sale empties it. On seed 5100 that credited the top replay with 131,091 while it finished on 175,295. Taking the money field itself and attributing each turn's change to the orders on the wire settles what kind of error it is: **no money at all moved on a turn with no orders**, for either farm, so there is no channel that had been missed — the reconstruction was simply blind. Reading sales off the `SELL` orders instead puts this farm at 108,378 asked against 243,603, and the true gap at **85,000 rather than 31,000**. Every product-level comparison drawn from the old figures was reading a number that was 30% short for one side and 6% short for the other.
- **And then the arithmetic on top of it was wrong too.** The corrected figures say the top plan sold 320 units of milk to this farm's 145, which read as 1.33 a head a day against 0.69 — half the published rate, and a $42,400 line all by itself. Counting cow-days off the daily table rather than multiplying the closing herd by thirty gives **0.98 against 1.74**, and 1.74 is above the 1.50 that perfect husbandry pays, so part of their milk is bought rather than milked (`BUY_PRODUCT` 99 against this farm's 42). The husbandry left on the table is about 80 units, **$19,400**, not $42,400. Check your own division before building on it.
- **The town's quote is front-loaded, and two goods run the other way.** Printing the book day by day: wheat 25 → 47, strawberry 120 → 229 by day 18 and flat after, milk 160 → 246 and flat from day 12, wool 200 → 239 and flat from day 14 — but **fertilizer falls 100 → 45 without ever turning, and melon 250 → 131**. So holding stock earns nothing after the middle of the season, which prices the idea of buying late and selling later at zero. It also breaks a rule the agent has carried all along: `reserve_frac` refuses to sell under the opening price *because scarcity lifts prices all season*, which is true of seven goods and false of two. For those two the floor is not patience, it is a stop order that fires on day three and never lifts — `sellable_qty` returns nothing from then on and what escapes trickles out through the pace fallback at whatever the price has fallen to. This farm moved 85 units of fertilizer at a mean of 61.7 where the top plan moved 300 at 70.3. One global fraction cannot be right for two kinds of good, which is the likeliest reason sweeping that fraction ever read as noise.
- **Care cannot be bought with a bigger number, measured twice now.** Raising the care weight by half costs 4,513 and doubling it 5,275; adding feed to it costs 3,205; weeding weights cost 3,641 to 4,640 whether alone or paired. What the code does have is an asymmetry rather than a shortfall: `FEED` carries an urgency of 1.5 rising to 4.0, `WATER` carries 3.0 for a plant that dies tonight, and `CARE` carries nothing at all — yet a care day left unspent is gone for good while a watering can wait a turn. That is a different lever from a flat multiplier, and it is being measured with the flat multiplier sitting in the same sweep as its control.
- **Care is already weighted correctly, and the way that was established is the point.** `CARE` alone among the tending jobs carries no urgency, so a care day that runs out is forfeit while a watering merely waits — an asymmetry worth testing. Shaped as a multiplier that only applies in the last few hours of the day it measured +1,075, with the flat-multiplier control sitting at −4,223 in the same sweep and at −4,513 and −5,275 in another: shaping recovers everything the flat version loses. Then the shaped version failed to replicate, −294 on a fresh band, with all six cells inside ±370 and the apparent "sharper is better" gradient gone. The finding that survives is the control, not the treatment.
- **Walking share does not order agents, so it was never the constraint.** The top published plan walks 43% of its actions to this farm's 57%, which reads as the whole difference in productive labour. In the same episode the median-strength opponent — the one this farm is actually scored against, and beats — walks **62%** and earns 70,530, while a build of this farm that got its walking down to 54% earned 20,358. The 43% is a property of that particular plan, not a property of strength.
- **The route planner in this codebase never feeds an animal.** It was written to attack exactly the walking figure above and left switched off. Turned on it costs **60,919 over 96 games at a 0% win rate**, and three of its variants return the same number to the coin because the route branch reads neither `stand_first` nor `stickiness`. The breakdown says why: under it the farm issues no `FEED`, no `CARE`, no `PICKUP`, no `FERTILIZE`, no `COLLECT_FERTILIZER` and no `DROP` at all, and passes on 1,562 of 5,952 actions. Nothing is ever fed, and `CARE` is only offered to an animal that has been fed, so the entire herd goes unmanaged by cascade. With the premise that motivated it withdrawn, it stays off and unrepaired.
- **Three sweep maxima in one day, and all three evaporated.** Raising the strawberry target measured +3,750 and came back −340 and then −3,638. Shaping the care urgency measured +1,075 and came back −294. Exempting three goods from the sell floor measured +2,731 and came back **−757**, with its whole family inside ±1,434 — so the floor written for seven goods is not, after all, costing anything in the three it silently stops. At a paired interval near ±2,000 over 96 games, taking the best of six or seven variants manufactures a +2,700 to +3,700 finding as a matter of course. The one change adopted today was tested against its own neighbours and held its sign across four disjoint bands; nothing else did. A single sweep's BETTER is a hypothesis, and the cost of treating it as a result is paid later, in the levers built on top of it.
- **Six mechanism readings in a row have been wrong.** Land, cutting wheat, care weighting, strawberry acreage, the order the tile budget is spent in, and walking — each was read off a side-by-side against the top plan, each was plausible, and each measured tie or worse. The side-by-side has been right about *where* the money is every time (this farm sells about half of what that plan sells, in every product at once) and wrong about *why* every time. Since half-of-everything is what a labour shortage and a demand shortage look like alike, the next measurement asks that directly rather than reading another statistic: scale the calendar's hand count and watch the response. It also exposed that the calendar's hand column overrides `max_hands` and `jobs_per_hand` outright, so those two knobs have been dead letters in every agent carrying a calendar — which is all of them.
- **Copying the strong plan's crop mix wholesale is worse, and it has a dose-response.** The two action lists put side by side are not uniformly apart: they sell 300 strawberries to this farm's 58, 154 wool to 62 and 126 melon to 50, while selling *no* egg and *no* tomato at all, and they plant 37 strawberry tiles against 9. So the mix was moved as a set rather than a dial — a multiplier over the whole calendar's per-crop targets, tomato to zero, melon to the measured ratio. Over 96 paired games: 2× strawberry **−1,910**, 4× **−7,365**, the full measured composition **−5,526**, and a 6× variant with carrot dropped too **−4,784**. Monotone in the wrong direction. That is the fifth refutation of the strawberry reading and the first with a dose curve behind it, and it holds in the other regime too: against the top published plan, 2× strawberry is **−7,313**. The side-by-side has now been right about where the money is and wrong about why seven times.
- **Every knob that had never been swept is also at its optimum.** Four settings that existed only as reasoned commentary in the source — how big a load justifies a shed trip, how much the value being carried should weigh, a bias that leaves the farmer out where tomorrow's hands are not, and re-asking the crop when the first choice runs out of seed — came back **+443 (replicated at +37), +64, −2,453 and −2,828** over 96 paired games. One of them is actively expensive: weighting a shed run by the value carried, at 0.30 instead of 0.10, costs **−23,863**. Nothing in the parameter set is left holding money.
- **The instrument was ranking the wrong column.** The sweep ordered variants by their own final money, but an episode is not won by earning, it is won by out-earning — and in a shared elastic market a change that floods the town lifts the price floor for both farms. Both sides' money already came back from every episode, so the margin costs nothing to report. Its first table showed exactly the shape it was built for: against the top published plan, the two variants that *gained* our own money, +1,966 and +1,506, moved the margin **−5,989** and **−5,903**. Whether that is real is a second question — all six arms came out negative and none of them had been asked how wide it was, so the column now prints its own interval before anyone reads a pattern into it.
- **A knob the agent does not have looks exactly like a knob that does not matter.** `P.update()` adds keys nobody reads, so sweeping a setting against an agent emitted before that setting existed spends the whole job measuring nothing — and it reports a clean tie, not an error. Six labour arms run against a frozen agent came back byte-identical to each other: same mean, same interval, same winrate, six times. That signature is the only warning it gives. The sweep now refuses the run and names the missing knob.
- **Accepting on the confirmation names the child twice, and a run of that is a random walk.** The confirmation set is disjoint from the screen, so it is unbiased for a candidate the screen named in advance — but accepting only when it comes out positive names the candidate a second time, on the confirmation's own number, and the accepted set is the upper half of whatever it happened to say. For a child truly worth zero that is a coin. 55 generations and 27 accepts arrived at **+529 ± 2,696** over 96 fresh games, which is what a walk over neutral edits looks like. The climb now draws a third disjoint set before an accept is committed, and pays for it only on the generations that would otherwise have accepted.
- **A fixed action list is not season-fragile.** Played over 32 fresh seeds from both sides, the top published plan takes **134,399** and ranges 75,900 to 172,888, winning 64 games out of 64 against a median-strength agent that averages 37,856. Whatever makes an open-loop plan work here, it is not that the seasons are alike.
- **The strong opponent costs 43% of the farm's money, and reverses nothing.** Same agent, same seeds, both sides, only the opponent swapped: **83,402** against a median-strength opponent and **47,417** against the top published plan. Every settled verdict was re-measured in that second world — the travel discount at 0.5, 0.7 and 1.0, the assignment rule, the labour scale, the crop mix — and not one changed sign; the intervals simply widen to ±3,400–4,800 because our own money is noisier when someone else is draining the town first. The regime distorts the ruler without moving the answers.

- **The one adopted gain survives being asked the harder question.** The travel discount at 0.7 was adopted on its own money, which the margin column now says is the wrong column. Re-measured on 128 fresh paired games with five settings around it, 0.7 is still the best of six, and against the old default of 1.0 it is **+1,786 ± 2,447 on our money and +2,314 ± 1,911 on the margin** — the margin clears zero by 403, so the gain is genuinely taken *from* the opponent rather than conjured by lifting the town's price floor for both farms. Pooling all five disjoint bands by inverse variance revises the headline down without threatening it: **+3,443 ± 1,136 over 512 games (t = 5.9)**, against the +3,898 published from four. What it does *not* explain is the ladder: the submission carrying it reads 634.1 and then 622.6 in its first five hours, below the previous agent's 669.5 at a comparable age. Two readings inside a rating's convergence are not a result, but they are not the confirmation either, and the honest position is that the money instrument and the ladder have not yet been shown to agree.
- **The conversion from money to rating is an upper bound, and was read as an estimate.** "+1,000 of money moves the win rate 5.1 points" was computed by re-scoring 117 real ladder games with *this farm's* money shifted by a constant. A policy change does not shift one farm's money by a constant: the market is shared, so whatever we do to our own harvest also moves what the town pays the opponent. Every target derived from that curve — including "bronze is +23,000 from here" — is therefore optimistic by however much the opponent gains alongside us, which is exactly the quantity the sweep had never recorded.
### ROGII - Wellbore Geology Prediction (Closed 2026-08-05)

**Competition:** [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) | **Deadline:** 2026-08-05 | **Prize:** $50,000 | **Metric:** RMSE

Predict True Vertical Thickness (TVT) along horizontal wellbores to automate geosteering. Per-well data (`*__horizontal_well.csv` logs with `[MD, X, Y, Z, GR, TVT_input]` + `*__typewell.csv` reference stratigraphy); predict the unrevealed "toe" rows (`TVT_input` is NaN, ~3,800 of ~5,300 rows per well) as a **delta from the last known TVT**. **Code competition** — submission is by notebook "Submit to Competition", not file upload.

| Approach | Score (RMSE, lower better) |
|---|---|
| Leaderboard top | 4.859 (public LB, 2026-07-16) |
| Medal lines (5,063 teams, 2026-07-16) | gold 6.118 / silver 7.043 / bronze 7.084 (public LB) |
| **Dual-pipeline fork — sp45 + fleongg blend + guarded override (best submitted)** | **7.311 (public LB — rank 480 / 3,589 when submitted 2026-06; 1,406 / 5,063 as of 2026-07-16)** |
| Public best (DWT/DTW-based clone group, *claimed*) | ~9.25 (public LB) |
| Champion — surface + dip-beam aligner (this repo) | 9.978 (leak-free GroupKFold CV; not LB-submitted) |
| GBT + TCN ensemble (this repo) | 9.905 (public LB) |
| Re-training fork baseline — GBT only | 10.224 (public LB) |

- **Anti-fork wall:** the public ~9.25 notebooks depend on a private BYOD image (`gcr.io/kaggle-private-byod`), ravaghi's `Trainer` pickles, and a custom `hill_climbing` module — a blind fork breaks. Only the public-image `9.251 DWT-based` notebook is forkable, and its sole custom dependency is `from hill_climbing import Climber`.
- **Fork-runnable baseline (Strategy A):** surgically transform the public source via `rogii-wellbore-baseline/generate_notebook.py` — (1) drop the `hill_climbing` import and inline an equivalent Caruana greedy-ensemble `Climber`; (2) the public artifacts dataset ships `*_trainer_*.pkl` (Trainer/BYOD) not the `models.pkl` cache the notebook's guard expects, so force a re-train from the precomputed `train.csv` features; (3) pin CatBoost to a single GPU (`devices="0:1"` → `"0"`). A `SMOKE` flag validated the full pipeline before the full run (3 LGB + 3 CatBoost → Climber → Optuna post-proc → Savitzky-Golay smoothing → valid `submission.csv`, 14,151 rows, no NaN).
- **Reproduction gap diagnosed (10.224 vs claimed 9.25):** pulling the submitted kernel and diffing it cell-by-cell against the original shows **no substantive difference** — `lgb_params`/`cb_params` and the entire post-processing are byte-identical (the only deltas are the inlined `Climber`, the single-GPU pin, and a smaller Optuna trial count that converges to the same value). The re-trained models are **per-fold on par with or better than** the author's saved models (our LGB fold-0 RMSE 9.49 vs the author's 9.64, recovered by loading the `*_trainer_*.pkl` with a stub `Trainer` class). Conclusion: **10.224 is the honest from-scratch reproduction of this GBT approach**; the title-claimed 9.251 only arises from the notebook's *load* branch (pre-trained `models.pkl`/`oof_preds.pkl`), which the public artifacts do not contain — so it is not cleanly reproducible from public materials.
- **Submission finding (verified on an active comp):** `kaggle competitions submit -f` returns **400** on this code competition — file submission is structurally disabled; submission must go through the notebook UI. Scoring this code competition takes ~70–90 min (notebook re-run on the hidden test set), not seconds.
- **Sequence-model lever worked — new best 9.905 via a GBT + TCN ensemble:** a flat per-row GBT ignores the row-to-row continuity of the toe trajectory (it only patches it post-hoc with Savitzky-Golay smoothing). A **TCN** (dilated residual 1D-CNN) that convolves along the ordered toe rows of each well captures it directly. Fed the *same* ~195 engineered features as the GBT, the TCN reached CV toe RMSE **10.60** standalone (`rogii-wellbore-tcn-hybrid/`) after stabilizing batch-size-1 training (**GroupNorm** instead of BatchNorm, gradient clipping, cosine LR) — 3 of 5 folds beat the GBT. It is then injected as a **7th ensemble member** into the proven GBT pipeline (`rogii-wellbore-ensemble/`): the TCN's OOF / test predictions are added as a `tcn` column to the `oof_preds`/`test_preds` dicts right before the hill-climb, so the existing greedy `Climber` weights all seven members with no other change (`allow_negative_weights=False` ⇒ a useless member gets weight 0, never worse than the GBT alone). Because the sequence and flat models have partly decorrelated errors, the blend improved the LB from **10.224 → 9.905** (verified). GPU note: the Kaggle P100 (sm_60) is incompatible with the current Torch build, so the kernel pins `machine_shape: NvidiaTeslaT4` (sm_75).
- **Rank reality check (fresh LB, 3,465 teams):** the 9.905 public LB is rank **1,657 / 3,465 (top 48%)** — mid-pack. Medal lines: **gold 6.72 (rank 17) / silver 7.229 (rank 174) / bronze 7.29 (rank 347)**; leaders 5.35–5.7. The whole public-clone lineage tops out ~9–10, so it sits far outside the medal zone — closing a **−3.2 RMSE gap to gold is an order of magnitude beyond any ±0.1–0.3 incremental tune** (target fixed first, gap back-calculated). A structural change of formulation, not another feature, is the only path.
- **Evaluation structure (verified) — optimize CV, not public LB:** the 3 visible test wells are *byte-identical* to 3 training wells (same `MD`/`GR`; the visible-prefix `TVT_input` equals the train `TVT`), so the public-LB sample is train-overlapping and the hidden toe answers literally exist under `train/`. But this is a notebook-rerun code competition with 3,127 teams and **no ~0 scores**, so the scored/private set is unseen (non-overlapping) wells. We therefore optimize **GroupKFold(by well) CV** — the honest unseen-well proxy — and treat the public LB as overlap-inflated (no leak exploitation; private decides medals).
- **Global formation structural-surface features (CV win):** the public pipelines reconstruct the train-only formation-top columns spatially via a per-well-median **local KNN plane**, which discards within-well dip. We replace that collapse with a per-formation absolute-depth surface `S_f(X,Y)` = **global 2nd-order WLS trend + local residual IDW**, fit from **per-row** samples across all wells (recovers dip), datum-separated, and **leave-one-well-out fold-safe** (per-well normal-equation downdate + self-excluded residuals — features valid under any GroupKFold split). Injected as new `tvtS_*` delta members (`rogii-wellbore-surface/`), this improved the self-contained tuned base's GroupKFold OOF from **10.32 → 10.19**, confirming the structural lever generalizes to unseen wells.
- **Surface + TCN stack — CV 10.19 → 10.06 (`rogii-wellbore-medal/`):** the TCN sequence model added as a Ridge-stack member on top of the surface base improved GroupKFold OOF absolute RMSE from 10.19 to **10.06**, and took the **largest stack weight (0.431 of 5 members)** — the sequence model carries unseen-well signal the flat GBTs miss.
- **Aligner core — the public beam is geometry-blind (the structural lever):** the public DTW/beam correlator (`_beam_jit`) walks the toe gamma-ray against the typewell with a symmetric index-move penalty `mc·|d|` and **never uses the well trajectory (`Z`, `MD`) at all** — it is pure GR matching. But `TVT = −Z + S_f(X,Y) + b`, so the expected per-step change is `ΔTVT = −ΔZ + dip·Δ(horizontal)`, all known or heel-estimable. A **dip-aware transition** centers each move on that geometric expectation instead of on zero. This is why the public-clone lineage plateaus ~9–10: it throws away the geometry.
- **Verified go/no-go (773 wells, leak-free):** reconstructing a held-out pseudo-toe (last 30% of each well's *known* zone; dip estimated from the heel 70% only, mirroring heel→toe deployment) the dip-aware beam cuts mean per-well RMSE from **6.933 → 5.753 (−1.18)** versus the geometry-blind beam at *matched search width* (the fair baseline is the same routine with zero expected-drift, which reproduced the public beam's 6.933 exactly — a clean apples-to-apples check). Geometry-only (no GR) scores 9.68, so GR and dip-geometry are complementary.
- **Step lesson — dip belongs in the transition, not as a feature:** handing the GBT a raw dip column first *hurt* CV (10.19 → 10.23) and was reverted — a tree cannot reproduce the along-trajectory integration that turns dip into TVT; the gain only materializes inside the aligner's path search.
- **Champion — dip-beam as an intra-well stack member (`rogii-wellbore-surface/`, CV 9.978):** the dip-aware beam's toe TVT, emitted as a feature (`tvt_dipbeam_d`) computed from each well's *own* known-zone dip, typewell, and trajectory only (**fold-safe by construction, no cross-well leakage**), improved the surface base's GroupKFold OOF absolute RMSE from **10.19 → 9.978 (−0.215)** — the single largest lever and the current champion. Confirmed that dip helps *only inside the aligner transition*: the same signal handed to the GBT as a raw feature hurt CV. The dip-beam also makes the TCN redundant (`medal+dipbeam` 10.058 > surface+dipbeam 9.978), so the TCN is dropped from the champion.
- **GBT-feature levers exhausted (10 experiments, all ±0.1–0.3, none beat 9.978):** GR-alignment tuning, surface-gradient dip (`surfdip` 10.107), ANCC cross-well kriging (`krig` 10.110), multi-scale wavelet GR texture (`wav` 10.266), and track-aligner replacement (10.171) each stayed inside the 9–10 band. **The GBT-feature core is mined out**; every axis that converts an inversion signal into a tree feature collapses back to the public-clone ceiling.
- **Pivot — inversion core (the formulation gap):** deep research (ROGII's own production patent US 11,480,045 + geosteering state-space/SMC literature) located the 7.x-vs-9–10 divide not in features but in **formulation**: the public forks do per-row DWT-feature → GBT *regression*, while the principled approach is **forward-model inversion** — build a synthetic log from the typewell under candidate (dip × typewell × thickness) and select the geology that best explains the observed horizontal GR. Caveat (verified): **no public artifact was confirmed to reach 7.x** (best public inversion-flavored notebook is still ~9.96), so there is no copyable winning recipe — the inversion core is a principled, high-risk research bet inferred from domain SOTA. A self-contained **go/no-go diagnostic** (`rogii-wellbore-invcore/`) gates each candidate core against the champion's dip-aware beam on a long pseudo-toe before any full build.
- **Learned-inversion core — NO-GO (decisive):** a from-scratch GR→stratigraphy inversion net (1D-CNN + typewell cross-attention, residual-on-geometry), trained directly on real known-zone pairs with good convergence, scored **14.8 vs the dip-aware beam's 5.58** and lost even to geometry-only (9.58), winning on just 22% of held-out wells. The DP-global-optimal beam is not beaten by a small learned aligner. The cheap diagnostic saved weeks of a full MTP build.
- **Particle-filter core (online dip) — NO-GO:** a distance-direction particle filter (state = TVT + OU-bounded local dip, GR likelihood vs typewell, confidence-gated blend) targets the *fixed-heel-dip* degradation. On a representative 40-well sample it was a **coin-flip** (blend 23.1 vs beam 16.9, wins 40%); an 8-well smoke that looked like a win (+1.8) was a lucky-sample artifact. Online-dip-from-GR does not robustly beat the fixed-dip beam.
- **Stretch lever — real but not extrapolable (key finding):** an **oracle** test (inject the *true* per-step typewell-index advance into the beam) collapses the long-toe RMSE **16.9 → 11.1 (−5.8, helps 68% of wells)** — by far the largest lever signal seen, proving typewell↔horizontal bed-thickness *stretch* is the dominant long-toe error. **But it does not extrapolate from the heel:** injecting the leak-free heel-mean stretch (mean ratio ≈ 0.985 ≈ 1) made it *worse* (18.7 vs 16.9). The heel matches geometry by construction (dip is fit there); the toe's structural deviation is not foreshadowed by the heel.
- **Conclusion (2026-06-23) & next lever:** across learned-inversion, particle-filter, and stretch, **the toe's structural deviation is not predictable from within-well data (heel + own typewell + GR)** — the within-well alignment frame is exhausted at ~9–10. The remaining gold-direction is **cross-well**: constrain an unseen well's toe TVT from neighboring *training* wells' known tops at similar (X,Y) — a stronger 3D spatial formulation than the per-formation WLS+IDW surface (−0.13) and ANCC kriging (±0.3) already tried.

#### Strategy pivot (2026-06-25) — public LB is an override mirage; private is decided by the base blend

- **The public frontier is a dual-pipeline fork, and we matched it (7.311).** The strongest public lineage (`fle3n-rogii-v5` / `rogii-dual-pipeline` / `rogii-lb-7-159/201`) is a two-pipeline blend — **Pipeline A "ridge-sp45"** (selector physics + LightGBM/CatBoost/Ridge stack + projection) and **Pipeline B "fleongg"** (likelihood-PF + GBM stack) — combined `0.55·A + 0.45·B`, then a **guarded contact override** on the few test wells that duplicate `train/`. Forking it (`rogii-wellbore-dualpipe/`) reached **public LB 7.311** — a ~2.6 jump over the prior 9.905. That was rank 480 / 3,589 (top 13.4%) when submitted in 2026-06; by 2026-07-16 the same score sits at **rank 1,406 / 5,063** as the field grew and converged on the same public lineage — the public board is override-saturated and flat (bronze 7.084 / silver 7.043 differ by 0.04 across hundreds of teams), which is exactly why GroupKFold CV/OOF, not public score, is the only honest signal here.
- **But the override does not transfer to the private leaderboard — the public notebooks say so themselves.** The 7.159 frontier notebook's own results summary states the guarded override is a *"public-LB-only gain"* and the gold-prefix calibration overlay *"can push the public LB toward ~7.2–7.3 but is a leakage path that does not transfer to the private leaderboard."* The visible test wells are train-overlapping; the override reconstructs them near-exactly (≈0.01 ft), which inflates only the public score. The honest blend number (no override) is ~7.5–7.6 on public, ~9.2 on unseen-well GroupKFold CV. **So on the private/medal leaderboard every override-using fork collapses back to its base blend — the medal goes to whoever has the best genuine base, and the field is clustered near the ~9.2 lineage ceiling.** The gold lever is therefore to lower the *base* CV; a small base gain can move private rank where the field is dense.
- **TCN as a decorrelated Ridge-stack member — verified base gain (−0.213 GroupKFold OOF).** Injecting the sequence-model TCN into Pipeline A's `oof_preds`/`test_preds` dicts (the proven 10.224→9.905 pattern) lets the positive-constrained Ridge stack weight it: pipeline-A Ridge OOF dropped **10.4197 → 10.2068** on GroupKFold (the unseen-well / private proxy). GPU note: the kernel pins `machine_shape: NvidiaTeslaT4` (the assigned P100 sm_60 is Torch-incompatible → CPU fallback otherwise). A re-generatable patcher (`generate_notebook.py`, `SMOKE` flag, byte-identical-override round-trip asserts) drives the injection.
- **Blend-level fixed-weight 3-way — rejected by validation.** Riding the standalone TCN over the blend (`(1−w)·base + w·tcn`) has no faithful blend-level OOF; its only signal — true-RMSE on the train-overlapping public wells — is **in-sample** (the GBTs memorize those wells, a CV~10 sequence model cannot match), so it favors the wrong thing and worsens in-sample. Kept dormant (`w_tcn = 0`); the Ridge-weighted A-injection (which *is* validated on the unseen-well proxy) is shipped instead.
- **Surface + dip-beam stacked into the dual-pipeline base — verified.** The two features that beat the pub_dwt base in our own `rogii-wellbore-surface` champion — the **global formation structural surface** (`tvtS_*`, −0.123) and the **dip-aware beam** (`tvt_dipbeam_d`, −0.215) — are injected into Pipeline A's feature build (`rogii-wellbore-dualpipe/generate_notebook.py`), forcing a GBT re-train (`FORCE_RETRAIN`) on the augmented feature set. Same Ridge-OOF diagnostic vs the 10.4197 baseline: the GBT base alone drops **10.4197 → 10.3195**, and with the TCN sequence member re-added on top the pipeline-A Ridge OOF reaches **10.0453 (−0.175 vs the surf-less TCN base)** — the maximally-stacked genuine base (surface + dip-beam + TCN), all measured on GroupKFold (the unseen-well / private proxy).
- **Blend-level fixed-weight member C — rejected by validation (again).** A standalone GBT trained on *only* the 12 structural surf/dip-beam columns, ridden over the 2-way base as a separate blend member, smoke-scored **CV delta-RMSE 12.65** — far weaker than the ~9.2 base, so a fixed blend weight over-weights a weak member (the same failure that kept the standalone TCN dormant, worse). **Lesson: decorrelated members belong *inside* Pipeline A's positive-constrained Ridge stack — which optimally weights them (surf+dip-beam −0.10, TCN −0.27) and zeroes out the useless ones — not as a fixed-weight blend addition.** The member-C plumbing is kept dormant (`w_c = 0`).
- **BiLSTM 2nd sequence member — shelved.** A bidirectional LSTM (recurrent, structurally decorrelated from the dilated-conv TCN) ran (CV 11.04) but OOM-killed the kernel when stacked *after* the TCN (two sequence models + the full GBT re-train exceeded host RAM), and its expected marginal was low (TCN-correlated, same features). Kept as code, gated off (`STACK_LSTM=False`).
- **Stronger global surface (thin-plate spline) — measured WORSE, rejected.** The 2nd-order WLS + IDW surface (−0.123) was swapped for a smoothed **thin-plate-spline** structural surface (one representative centroid node per well, exact per-well LOO). It crashed the kernel with no Python traceback; a **subprocess-isolation probe** (a child process that runs the suspect build and whose parent captures the return code + last flushed line, so the kernel still reaches `COMPLETE` and Kaggle saves the log) proved the build runs fine standalone — the failure was an **OOM-kill** (the heavy build's peak memory coexisting with the resident 7.4 GB `train.csv`), not the suspected LAPACK singularity. Isolating the surface build in a subprocess fixed the crash, but the measurement was decisive: pipeline-A Ridge OOF **without_TCN 10.5318 vs the 2nd-order surf's 10.3195 (+0.21 worse)**. Collapsing each well to one centroid node discards the per-row local structure the 2nd-order trend + IDW retained, and the noisier columns also degraded the TCN input (TCN CV 10.3 → 11.5). Reverted to the verified-best 2nd-order base. **The surface-feature axis (surf / dip / kriging / wavelet / TPS) is now exhausted in the 10.0–10.5 band.**
- **Transformer-encoder sequence member — measured, rejected.** With the surface axis mined out, the remaining proven base lever is a 2nd decorrelated *backbone*. A **Transformer encoder** (self-attention — a distinct inductive bias from the dilated-conv TCN, where the recurrent BiLSTM was too TCN-correlated) was injected as a Ridge-stack member, with **windowed attention (W ≤ 512)** so memory stays O(W²) and long toe sequences cannot OOM. It OOM-killed ~27 s into its cell on two full runs: the arena-return (`malloc_trim`) + per-column-statistics mitigation was insufficient because the cell still materialized a *second* ~3 GB normalized sequence copy on top of the TCN's residue. The decisive fix was to have the Transformer **reuse the TCN cell's already-built per-well sequences** (the same globally-standardized arrays, aliased with no copy; the TCN's `del` moved to the end of the Transformer cell) — total sequence memory stays at one copy, the peak the TCN already proved fits. The full run then completed, and the `delta_xf` Ridge-OOF diagnostic was decisive against the verified base 10.0453: **without_XF 10.0270 → with_all 10.0310 (delta_xf +0.0039)** — the Transformer does not decorrelate beyond the TCN (same outcome as the shelved BiLSTM), so it was dropped and the base restored to the surface + dip-beam + TCN champion (Ridge-OOF 10.0453). **The decorrelation-backbone axis (TCN, BiLSTM, Transformer) is now exhausted alongside the surface axis.** (ROGII private LB is hidden until the 2026-08-05 deadline and the public LB is override-saturated and flat, so GroupKFold CV/OOF — not public score — is the only honest signal for base improvements.)

- **DenseANCC structural interpolation — a self-inclusion artifact, rejected.** To probe whether a spatial structural interpolant could beat the physics candidates directly, a toe-masked train-well holdout scored each candidate's toe against the true TVT. A naïve pass looked spectacular — the `dense_ancc_*` structural-imputation candidates scored ~3.9 toe-RMSE, roughly *half* the best physics candidate. But that build imputed with the query well left *in* its own offset-well pool (distance-0 self-match). Switching the two imputer call sites to proper **leave-one-well-out** (`self_wid=wid`, production-safe because a hidden-test well is absent from the train-built pool and so excludes nothing) collapsed the dense candidates below every physics candidate — the ~3.9 was pure self-inclusion optimism. A useful reminder that any cross-well imputer must be scored with the query well held out before its holdout number is trusted.

- **Ops trap — `kaggle kernels push` exits 0 on push errors, and a stale COMPLETE can impersonate success.** When the weekly GPU quota is exhausted, `kaggle kernels push` prints `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.` **and still returns exit code 0**, so a CI push step "succeeds" while no new kernel version exists; a naive status poll then reads the *previous* run's `COMPLETE` and reports the whole pipeline green — for a run that never happened. The shared `kaggle-push.yml` now (1) requires the `successfully pushed` marker in the push output, and (2) captures the kernel's `lastRunTime` (via `kaggle kernels list -v`) *before* pushing, and trusts a terminal status during startup only after `lastRunTime` has advanced past it. Related measurements: the weekly GPU quota resets Saturdays 00:00 UTC, and Kaggle's CPU-only machines have a *smaller* usable-memory envelope than the T4 GPU machines — a pipeline that holds the 7.4 GB `train.csv` resident plus a GBT re-train fits on the T4 host but is OOM-killed on the CPU host (and the capped CPU GBT re-train alone took ~3.1 h), so "just run it on CPU while the quota recovers" is not a viable escape hatch for heavy kernels.

---

### Playground Series S6E6 - Stellar Classification (Finished 2026-06-30)

**Competition:** [Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6) | **Deadline:** 2026-06-30 (finished) | **Final:** public 0.95944 / **private 0.95939**

3-class classification of astronomical objects (GALAXY 65% / QSO 20% / STAR 14%, SDSS-style). 577k train rows, 10 features (2 categorical). Label submission (`[id, class]`).

| Approach | Public LB |
|---|---|
| **Two-stage pseudo-label distillation (153k confident test rows join fold-train)** | **0.95944** |
| Full ensemble (LGB + XGB + CatBoost, 5-fold ES) + pairwise-diff features + per-class probability weights | 0.95884 |
| Full ensemble + per-class probability weights (no diff features) | 0.95866 |
| + original SDSS17 dataset in fold-train (external data) | 0.95741 (rejected) |
| LGB-only 3-fold smoke baseline | 0.95466 |

- **Metric finding:** the LB metric is macro-F1 and OOF macro-F1 matches it within ~0.001, so per-class probability weights tuned on OOF by coordinate ascent are a proven lever (+0.004 LB over plain argmax; balanced class weights alone hurt argmax but win after weight tuning).
- **Feature lever:** pairwise differences of all numeric columns (generalized color indices), validated by an A/B smoke (+0.00093 OOF) before spending a full run (+0.0002 LB).
- **External-data lesson:** appending the original SDSS17 dataset to the training side of each fold RAISED OOF (+0.0002, near-duplicates of validation rows leak in) but DROPPED LB (-0.0014). OOF deltas only predict LB deltas while the training distribution stays unchanged.
- **Pseudo-label lever (proven):** test rows predicted with 0.995-plus confidence join the fold-train side in a second stage: +0.0006 LB while STAGE2 OOF only rose +0.00025 - test-distribution alignment gains do not show in OOF.
- **Levers exhausted (champion holds at 0.95944):** pseudo round 2 (LB 0.95907), MLP/NN diversity blend (NN OOF far below GBT, every blend weight worse), physics interaction features (OOF -0.0003), and an Optuna-tuned LGB ported into the full ensemble (STAGE2 OOF 0.95790 < champion 0.95796) were each tested and rejected - GBT ceiling confirmed. The proxy HPO gain (+0.00068 on a 3-fold LGB-only proxy) did not survive the 5-fold LGB+XGB+CatBoost ensemble. OOF / test-probability .npy artifacts stay persisted (kernel_sources) for any future stacking.

- **Final result (private revealed):** the champion scored **private 0.95939** vs public 0.95944 — a gap of only 0.00005, and it was our best submission on both splits, so Kaggle's default "best public" auto-selection was the correct pick. The metric finding held end to end: with macro-F1 and no distribution shift, OOF ≈ public ≈ private, which is why every OOF-validated lever above transferred (and why the external-data lever, the one that *did* change the training distribution, was the only one whose OOF gain inverted on the LB).
- **Pipeline:** `playground-series-s6e6/generate_notebook.py` → `kaggle-push.yml` (GitHub Actions) → Kaggle kernel. A `SMOKE` flag validates the plumbing (LGB-only, 3-fold) before the full ensemble (1 seed x 5 folds).
- **Gotcha confirmed:** `competition_sources` mounts data at `/kaggle/input/competitions/<slug>/` (the notebook auto-discovers the dir via `os.walk`).

---

### BirdCLEF+ 2026 - Acoustic Species Identification (Ended)

**Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) | **Deadline:** 2026-06-03 (ended)

Identify 234 species (birds, insects, amphibians, reptiles) from audio recordings in the Pantanal, Brazil. Evaluated with macro ROC AUC. **Final: private 0.92187 / public 0.92685** (improved-ensemble fork, 2026-04-18); no further submissions after 2026-04-23.

| Approach | LB |
|---|---|
| improved-ensemble fork (Perch v2 + ProtoSSM v5 + ResidualSSM + TTA + rank-aware + delta smooth) | **0.926** |
| Perch v2 + Bayesian prior + LogReg probe (fork) | 0.908 |
| public-blend-v6 fork (lb862 + lb872 blend) | 0.890 |
| eca_nfnet_l0 mel baseline (single fold0, CV 0.969) | 0.768 |
| BEATs-SED + Attention Pooling (archived) | 0.745 |

- **Current strategy (2026-04-23):** Reproduction base reached (0.926 ≒ public max 0.929 claimed). Moving to **proven-stacking phase**: external Xeno-Canto data (pipeline active), multi-backbone ensemble, class balancing — all recurrent techniques across BirdCLEF 2024/2025 top solutions. Note: `unlabeled_soundscapes/` not provided in 2026 (unlike 2024/2025) → pseudo-label distillation replaced by external data.
- **External data pipeline (active):** Xeno-Canto (XC) v3 API → 11,563 Aves recordings filtered (Q A|B, non-ND license, 159 species, 138.9h total). Dataset: [`yasunorim/xc-birdclef-2026-target-urls`](https://www.kaggle.com/datasets/yasunorim/xc-birdclef-2026-target-urls). Embedding kernel: [`yasunorim/xc-perch-v2-embed-birdclef-2026`](https://www.kaggle.com/code/yasunorim/xc-perch-v2-embed-birdclef-2026) — outputs Perch v2 embeddings + logits (1536-dim, 234-class) for ProtoSSM ingestion. Integration design: [XC_INTEGRATION_DESIGN.md](birdclef-2026-work/docs/XC_INTEGRATION_DESIGN.md). **Lesson:** Perch v2 is a CPU-only SavedModel — must set `CUDA_VISIBLE_DEVICES=""` before `import tensorflow` on GPU-enabled Kaggle machines to avoid `InvalidArgumentError`.
- **Prior-years solution survey:** [PRIOR_YEARS_SOLUTION_SURVEY.md](birdclef-2026-work/docs/PRIOR_YEARS_SOLUTION_SURVEY.md) — 2024 3rd (jfpuget/TheoViel) + 2025 2nd (VSydorskyy) + 2025 5th (myso1987) writeups distilled into actionable gap list.
- **Competition rules confirmed:** External data allowed under Section 2.6 (publicly available + equally accessible). Xeno-Canto, iNaturalist, past BirdCLEF data all eligible. No pre-deadline disclosure thread obligation in 2026 rules.
- **Notebooks:**
  - `birdclef-2026-improved-ensemble-fork` (CPU): yuriygreben claimed LB 0.929 fork — LB **0.926** (current best, reproducibility variance -0.003)
  - `birdclef-2026-perch-v2-repro` (CPU): Perch v2 fork — LB 0.908
  - `birdclef-2026-public-blend-v6-fork` (CPU): public blend reproduction — LB 0.890 (below our base, archived)
- **Archived:** BEATs (0.745), nfnet_l0 from-scratch (0.768) — ruled out before reproduction base was reached; from-scratch architectures stay blocked, proven-stacking techniques are active.

---

### Deep Past Challenge - Akkadian to English Translation (Ended)

**Competition:** [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

Ancient cuneiform (Akkadian) transliteration → English translation task. Evaluated with BLEU + chrF++.

**Notebook:** [Deep Past Cloud Workflow + TF-IDF Baseline](https://www.kaggle.com/code/yasunorim/deep-past-cloud-workflow-tfidf-baseline) *(public)* 🥉

| Approach | Public Score |
|---|---|
| TF-IDF char n-gram nearest neighbor | 5.6 |

- **Approach:** Character n-gram TF-IDF (2-5), cosine similarity nearest neighbor
- Pushed via GitHub Actions cloud workflow (see above)

---

### S6E2 - Predicting Heart Disease (Ended)

**Competition:** [Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) | **Deadline:** 2026-02-28

Binary classification (Presence / Absence) with AUC-ROC evaluation.

**Notebook:** [S6E2 Heart Disease - EDA & Ensemble](https://www.kaggle.com/code/yasunorim/s6e2-heart-disease-eda-ensemble-wandb)

| Model | CV AUC |
|---|---|
| **Ensemble (avg)** | **0.95528** |
| CatBoost | 0.95524 |
| LightGBM | 0.95515 |
| XGBoost | 0.95513 |

- **LB Score:** 0.95337
- **Approach:** LightGBM + XGBoost + CatBoost (GPU), 5-fold Stratified CV, 6 interaction features
- **Blog:** [Zenn](https://zenn.dev/shogaku/articles/kaggle-s6e2-github-wandb-gpu-workflow)

**Tech Stack:** LightGBM, XGBoost, CatBoost, W&B, GPU

---

<details>
<summary><h2>🥉 Bronze Medal Notebooks (14)</h2></summary>

### 1. CAFA 6 - Protein Function Prediction

**Notebook:** [Baseline with Regularization](https://www.kaggle.com/code/yasunorim/baseline-with-regularization)

Multi-label classification of protein functions using Gene Ontology (GO) terms.

**Approach:**
- TF-IDF k-mer features (3-grams) from amino acid sequences
- MLP with regularization (Dropout 0.5, Weight Decay, Early Stopping, BatchNorm)
- 1500 GO terms across 3 aspects (Biological Process, Molecular Function, Cellular Component)
- GO hierarchy propagation

**Tech Stack:** PyTorch, scikit-learn, pandas, numpy

---

### 2. NFL Big Data Bowl 2026 - Prediction

**Notebook:** [Geometric Rules Baseline - 2.921 RMSE (No ML)](https://www.kaggle.com/code/yasunorim/geometric-rules-baseline-2-921-rmse-no-ml)

Sports analytics using NFL player tracking data.

**Approach:**
- Physics-based geometric rules (no ML)
- Targeted receivers → direct path to ball landing point
- Defensive coverage → distance-based offset from receivers

**Performance:** RMSE 2.921 yards, <5 seconds execution

**Tech Stack:** Python, pandas, polars, numpy

---

### 3. PhysioNet - Digitization of ECG Images

**Notebook:** [PhysioNet ECG Baseline](https://www.kaggle.com/code/yasunorim/physionet-ecg-baseline)

Submission format guide for ECG image digitization challenge.

**Key Contributions:**
- Correct submission format documentation
- Common mistakes and how to avoid them
- Working baseline with verified format

**Tech Stack:** Python, pandas, numpy

---

### 4. Diabetes Prediction (S5E12) - EDA & Baseline

**Notebook:** [Diabetes Prediction - EDA & Baseline](https://www.kaggle.com/code/yasunorim/diabetes-prediction-eda-baseline-s5e12)

Comprehensive EDA and LightGBM baseline. CV AUC 0.72687.

**Tech Stack:** Python, pandas, LightGBM, scikit-learn, matplotlib, seaborn

---

### 5. Diabetes Prediction (S5E12) - Rank-Based Ensemble

**Notebook:** [Diabetes Prediction - Rank-Based Ensemble](https://www.kaggle.com/code/yasunorim/diabetes-prediction-rank-based-ensemble)

Rank-based blending with dual LightGBM models. Blended OOF AUC 0.72716.

**Tech Stack:** Python, pandas, LightGBM, scikit-learn

---

### 6. MLB Statcast - Senga's Ghost Fork (2023-2025)

**Notebook:** [Senga Ghost Fork Analysis](https://www.kaggle.com/code/yasunorim/senga-ghost-fork-analysis-2023-2025)

Statcast data analysis of Kodai Senga's forkball ("Ghost Fork") across 3 seasons. Movement comparison, release point analysis (FF vs FO), batter splits, and performance by batting order.

**Tech Stack:** Python, pybaseball, DuckDB, matplotlib, seaborn

---

### 7. MLB Statcast - Kikuchi's Slider Revolution (2019-2025)

**Notebook:** [Kikuchi Slider Revolution](https://www.kaggle.com/code/yasunorim/kikuchi-slider-revolution-2019-2025)

Statcast data analysis of Yusei Kikuchi's pitching evolution from Mariners to Blue Jays to Astros. Pitch mix changes, slider usage trends, and movement analysis across 7 seasons.

**Tech Stack:** Python, pybaseball, DuckDB, matplotlib, seaborn

---

### 8. MLB Bat Tracking - Japanese MLB Batters (2024-2025)

**Notebook:** [Bat Tracking: Japanese MLB Batters (2024-2025)](https://www.kaggle.com/code/yasunorim/bat-tracking-japanese-mlb-batters-2024-2025)

MLB bat speed and swing metrics analysis for Japanese MLB batters using Baseball Savant bat tracking data.

**Tech Stack:** Python, pandas, matplotlib, seaborn

---

### 9. March Machine Learning Mania 2026 - Baseline

**Notebook:** [March Machine Learning Mania 2026 Baseline](https://www.kaggle.com/code/yasunorim/march-machine-learning-mania-2026-baseline)

NCAA basketball tournament prediction using historical game data.

**Approach:**
- LightGBM + Logistic Regression ensemble
- Feature engineering from seed differences and historical win rates
- Brier score optimization

**Tech Stack:** Python, LightGBM, scikit-learn, pandas

---

### 10. WBC 2026 Scouting - MLB Statcast Spray Charts

**Notebook:** [MLB Statcast Spray Charts for WBC 2026 Players](https://www.kaggle.com/code/yasunorim/mlb-statcast-spray-charts-for-wbc-2026-players)

Spray charts and pitch zone charts for WBC 2026 players using Baseball Savant Statcast data and baseball-field-viz.

**Approach:**
- Spray charts by batter (hit direction + distance)
- Pitch zone charts by pitcher (location heatmaps)
- Visualization using baseball-field-viz (self-published PyPI package)

**Tech Stack:** Python, pybaseball, baseball-field-viz, matplotlib

---

### 11. Deep Past Challenge - Cloud Workflow + TF-IDF Baseline

**Notebook:** [Deep Past Cloud Workflow + TF-IDF Baseline](https://www.kaggle.com/code/yasunorim/deep-past-cloud-workflow-tfidf-baseline)

Akkadian cuneiform transliteration → English translation baseline using TF-IDF character n-grams. Demonstrates GitHub Actions cloud workflow for Kaggle code competitions.

**Approach:**
- Character n-gram TF-IDF (2-5), cosine similarity nearest neighbor
- Fully managed via GitHub Actions (`git push` → Kaggle)

**Tech Stack:** Python, scikit-learn, pandas

---

### 12. Titanic - Japanese Optuna Test

**Notebook:** [Titanic Japanese Optuna Test](https://www.kaggle.com/code/yasunorim/titanic-japanese-optuna-test)

Titanic survival prediction with Optuna hyperparameter optimization. Japanese-language notebook demonstrating automated tuning workflow.

**Tech Stack:** Python, Optuna, LightGBM, scikit-learn, pandas

---

### 13. Matplotlib & Seaborn 日本語化テンプレート

**Notebook:** [【日本語化】Matplotlib & Seaborn 文字化け解消テンプレート](https://www.kaggle.com/code/yasunorim/matplotlib-seaborn)

Kaggle環境でMatplotlibとSeabornの日本語フォント文字化けを解消するテンプレートノートブック。

**Tech Stack:** Python, matplotlib, seaborn

</details>

---

<details>
<summary><h2>📓 Study Notes (5)</h2></summary>

Located in [`study-notes/`](./study-notes/).

| # | Competition | Notebook | Blog |
|---|---|---|---|
| 1 | Titanic | [Kaggle](https://www.kaggle.com/code/yasunorim/a-journey-to-0-789-with-feature-engine-optuna) | [解説](./study-notes/01-feature-engine-optuna.md) |
| 2 | House Prices | [Kaggle](https://www.kaggle.com/code/yasunorim/japanese-stacking-feature-engineering-guide) | [解説](./study-notes/02-stacking-feature-engineering.md) |
| 3 | Spaceship Titanic | [Kaggle](https://www.kaggle.com/code/yasunorim/japanese-spaceship-titanic) | [解説](./study-notes/03-spaceship-titanic.md) |
| 4 | Commodity Prediction | [Kaggle](https://www.kaggle.com/code/yasunorim/forward-looking-target-fix) | [解説](./study-notes/04-forward-looking-target-fix.md) |
| 5 | LLM Classification | [Kaggle](https://www.kaggle.com/code/yasunorim/japanese-llm-classification) | [解説](./study-notes/05-llm-classification.md) |

</details>

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **ML Libraries** | LightGBM, XGBoost, CatBoost, PyTorch, scikit-learn |
| **Data Processing** | pandas, numpy, polars |
| **Visualization** | matplotlib, seaborn |
| **Experiment Tracking** | Weights & Biases |
| **CI/CD** | GitHub Actions, Kaggle API |
| **Development** | Claude Code, Jupyter Notebook |

---

**Kaggle:** [@yasunorim](https://www.kaggle.com/yasunorim)

*Built with Claude Code*