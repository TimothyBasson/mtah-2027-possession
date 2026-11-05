# Data points for possession

Every signal we can measure, or work out, from the match data, and what it might say about
possession. It's a reference for choosing clues. None of it is tested yet unless it says so.
[DATA.md](DATA.md) describes the raw files themselves.

Each one helps with up to three questions:

- **who**: which team has the ball
- **dead**: whether the ball is out of play
- **when**: the moment possession changes

## 1. Tracking, every frame of BAR-ATH

Positions are 2D: x and y on the pitch, with no height, for the ball as well as the players. The
ball's channels in the ATD metadata are `x`, `y` and `s` (speed).

### Ball

| Data point | Helps with | How |
|---|---|---|
| Ball position | who, dead | The input to the nearest-player clue. A ball seen past a line is out of play. It also shows which half the ball is in, and how near each goal. |
| Ball speed | who, when | A fast ball is a pass or shot in flight: nobody is on it, but possession usually stays with the team that kicked it. A ball that stays still suggests a set piece being lined up. |
| Change in ball velocity (see below) | when | Marks a touch. Possession can only change when the other team touches the ball, not when a player runs past it. |
| Ball missing | dead | The ball is missing on 41% of graded frames. In BAR-ATH a missing ball looks like a dead one: players move at a median 1.0 m/s with the ball missing and 1.9 m/s with it seen. Metrica shows the same pattern. |
| How long it has been missing, and where it was last seen | dead | Lost at the touchline and gone for 10 s: probably out. Lost in midfield for a moment: probably in the air or hidden behind players, and still live. |
| Last-seen velocity, projected forward | who | Estimates where a missing ball went, such as the landing spot of a long ball. Where it reappears, and who is near it then, says who got it. |

**Change in ball velocity, exactly.** For the ball position `p` at frames `t-k`, `t` and `t+k`,
with `k` about 4 frames (0.16 s):

```
v_before = (p[t]   - p[t-k]) * 25 / k        # 2D velocity arriving at t, m/s
v_after  = (p[t+k] - p[t])   * 25 / k        # 2D velocity leaving t

speed change      = | |v_after| - |v_before| |       kicks and traps
direction change  = angle(v_before, v_after)        deflections, flicks
velocity change   = |v_after - v_before|            both at once: the single number to use
```

Plain speed says the ball is moving. Velocity change says something just hit it. A ball rolling
across the pitch at a steady 12 m/s has a high speed and a velocity change near zero. A
deflection that keeps the speed at 12 m/s but turns the ball 40° has a velocity change of 8.2 m/s.
databallpy counts a touch at a speed change above 5 m/s or a turn above 10°, with a player
nearby. Positions jitter by roughly 0.1–0.3 m, so `k` must span a few frames: over a single frame
that jitter alone gives up to 7 m/s.

### Players

46 tracking slots, each with a known team. A slot is not a person, and only about 20.5 of the 22
players are tracked on average.

| Data point | Helps with | How |
|---|---|---|
| Player positions | who | The nearest player from each team, how much closer one is than the other, and how many of each team are within 5 m of the ball. |
| Player speed | dead | Everyone walking means a stoppage. Running means live play. A sprint burst suggests a turnover or counter. |
| Speed missing | data quality | Marks a position the tracker estimated rather than saw. Trust it less. |
| Number of players tracked | data quality | A sudden drop is the tracker losing people, not anything in the game. |

### Team shape

Worked out per frame, per team, from the outfield players. With no identities, the goalkeeper is
taken as the player deepest toward their own goal. `x` runs along the pitch in the team's
attacking direction and `y` runs across it.

| Data point | Calculation | Helps with | What it says |
|---|---|---|---|
| Team centre | the average x and the average y of the team's players | who | Where the team is as a block |
| Team centre velocity | how far the centre moved over about 1 s, in m/s | who, when | Moving forward means in possession. Both centres reversing at once means a turnover. |
| Length | the spread along the pitch | who | Attacking teams stretch, defending teams compact |
| Width | the spread across the pitch | who | Attacking teams go wide |
| Spread | average distance from the team centre | who | The same, and less thrown by one missing player |
| Area | area of the outline around all the players | who | The same, in m² |
| Defensive line height | mean x of the deepest 3–4 outfield players, from their own goal | who | The team in possession pushes its line up |
| Pressure | opponents within about 5 m of the ball or of a player | who, when | The team pressing is the one without the ball. A jump in pressure is a moment to watch. |
| Attacking direction per half | from where each goalkeeper stands | (needed for the rest) | Without it, "forward" means nothing. It flips at half-time. |
| Set-piece formations | player layout | dead, who | <ul><li>Lined up at halfway: kickoff.</li><li>A crowd in the box and one player at the corner flag: corner.</li><li>One player on the touchline: throw-in.</li><li>Keeper in the six-yard box, opponents pushed back: goal kick.</li></ul> The player taking it gives the team. |

Compare the two teams rather than using raw values. "Home is more spread out than away" holds up
across teams and situations. A fixed cutoff like "spread above 15 m" doesn't.

**Catches:**
- A missing winger changes the width by 15 m, so use spread or percentiles rather than extremes.
- Duplicated tracks pull the team centre toward them.
- Shape takes a few seconds to change after a turnover, so on its own it puts the change late.
- Team style blurs it. A team keeping the ball deep can look compact, and a high press can look stretched.

## 2. Smart Tagging, BAR-ATH only

638 tags, produced automatically by Metrica and not reviewed, so they contain mistakes. Times are
seconds from the first frame of the tactical video.

| Data point | Helps with | How |
|---|---|---|
| Set pieces, with type and time | dead, when | A restart ends a dead-ball stretch, so the tag gives the frame play resumes. Most carry the team taking it. Every throw-in is tagged `N/A`. |
| Phases with a team: build-up, progression, final third… | who | A strong hint over a stretch. But a phase isn't a possession: one can cover two, or stop short. |
| Attacking and defensive transitions (24 each) | when | Mark the moments the ball changes hands. |
| `LONG BALL` (29) | when | Marks long balls. Whether the team tag means the kicker or the winner is unchecked. |
| Goals (4) | dead | A long stoppage, then a kickoff by the team that conceded. |

## 3. Homography, every frame

| Data point | Helps with | How |
|---|---|---|
| Where the camera points | ball position | The camera follows the ball, so the centre of the view on the pitch roughly marks where the ball is, even when the ball is missing. Unverified. |
| Camera pans and zooms | dead, when | The camera may settle at a set piece and swing across on a long ball. Unverified. |

## 4. Video

| Data point | Helps with | How |
|---|---|---|
| Tactical video pixels | ball position | Detect the ball ourselves where the tracking lost it. Heavy work. |
| Broadcast cuts, close-ups and replays | dead | These nearly always happen during a stoppage. Needs syncing first, because the broadcast runs on its own timeline. No outside match has video to test this on. |
| Broadcast tracking (ball seen 50.5% of the time) | ball position | A second view of the ball, which may fill some gaps once it is synced. |

## 5. Fixed facts

| Fact | Helps with |
|---|---|
| Kickoff frames 11185 and 113742, and the halves (FAQ.md) | Everything graded before a kickoff is dead. Each half is a separate sequence. |
| Pitch size, 105.3 × 68.0 m | Converting to metres, and placing the lines |

## 6. Other matches, for testing and training

| Data | Use | Catch |
|---|---|---|
| 10 outside matches with labels (`external/`: 7 IDSSE, 3 Metrica) | Measure whether a data point actually helps. Learn how much to trust it. | Tracking only: no tags and no video. Their possession definitions differ from the challenge's. IDSSE's ball is dropped at random by `--seen`, unlike ours. |
| Public 5 minutes of BAR-ATH | Looking only | Not allowed for tuning (RULES.md) |

## Looking forward in time is allowed

The task is offline. We label a finished match, so a frame's label can use data from after it.
Nothing in the rules says otherwise. It also helps, because spellF1 punishes lag:

- Centred windows compare `[t-n, t)` with `(t, t+n]`, so a change such as a jump in pressure shows up at the frame it happens, not `n` frames later.
- The velocity change above already uses `t+k`.
- Viterbi smoothing decides each frame using the whole half.

Rules for windows:
- Don't let a window cross a half or a kickoff.
- Choose `n` on the outside matches.
- Compute every data point the same way on BAR-ATH as on the outside matches.

## Worked case: long balls and clearances

CHALLENGE.md has three rules that matter here:
- Possession changes only on **controlled** possession by the other team.
- While the ball is loose or contested, possession stays with the last team that had it.
- A touch without control, a deflection, or an aerial duel won without control is not a change.

So through a long ball:

1. The kicking team keeps the ball in flight, even when the opponents stand nearer to where it lands.
2. A header or knock-down by the other team doesn't flip it.
3. It flips at the moment the other team gains control, not at the header or landing.
4. A clearance into touch is D.

The data points that help most, in order:

1. **Velocity change plus speed after it.** This finds the kick. The nearest player at that frame gives the kicking team.
2. **A control test after the landing.** It looks forward. One team stays nearest for about 2 s, or the ball is slow with one player within 1–2 m, or the same team touches it twice.
3. **Ball missing, with its last-seen velocity.** It gives the landing area and where the ball reappears.
4. **Player speed.** A missing ball with the players sprinting is in the air and live, not dead.
5. **Players running to the same spot.** That estimates the landing spot, and who gets there first.
6. **Ball out of play.**
7. **Smart Tagging `LONG BALL`.** For checking long balls by eye in the viewer.
8. **Team centre velocity after the landing.** A slower confirmation of who won the second ball.
