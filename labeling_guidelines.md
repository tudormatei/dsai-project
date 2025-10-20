# Concept Labeling Guidelines

## Purpose
This document defines EXACT criteria for labeling movement concepts to ensure consistency across all 4 labelers.

---

## The Three Concepts

### 1. PERIODICITY (Pattern Regularity)
**Question**: How regular/repetitive is the movement pattern?

**1.0 - High Periodicity**
- Clear, consistent repeating pattern throughout the window
- Examples:
  - Walking: Regular steps with consistent timing
  - Jogging: Steady running rhythm
  - Upstairs/Downstairs: Regular stepping pattern
- Visual: You can clearly identify repeating cycles

**0.5 - Medium Periodicity**
- Some repeating pattern but irregular or interrupted
- Examples:
  - Walking with pauses or speed changes
  - Movement that starts/stops mid-window
  - Pattern exists but timing varies
- Visual: Pattern is visible but not consistent

**0.0 - No Periodicity**
- No repeating pattern, random or chaotic movement
- Examples:
  - Sitting: Random fidgeting or adjustments
  - Standing: Small random movements
  - Transitioning between activities
- Visual: No clear cycles visible

---

### 2. TEMPORAL STABILITY (Pattern Consistency Over Time)
**Question**: Does the pattern stay the same throughout the 3-second window?

**1.0 - Stable**
- Movement pattern is consistent from start to end
- Intensity and characteristics don't change
- Examples:
  - Continuous steady walking
  - Stationary sitting/standing
  - Steady jogging at constant pace
- Visual: Pattern looks uniform across the entire window

**0.5 - Moderately Stable**
- Pattern changes once during the window
- Transition or single shift in movement
- Examples:
  - Start sitting, then shift position
  - Begin walking, accelerate once
  - One distinct change in pattern
- Visual: You can identify a clear "before and after"

**0.0 - Unstable**
- Constantly changing pattern
- Multiple transitions or highly variable
- Examples:
  - Transitioning between activities
  - Irregular movements with no consistency
  - Multiple direction/speed changes
- Visual: No consistent segment; keeps changing

---

### 3. COORDINATION (Multi-axis Movement Relationship)
**Question**: How coordinated are the movements across X, Y, Z axes?

**1.0 - High Coordination**
- All three axes move together in a coordinated pattern
- Clear relationship between X, Y, Z movements
- Examples:
  - Walking: Forward motion (Y) + vertical bounce (Z) + some lateral (X)
  - Jogging: Strong coordinated multi-axis pattern
  - Upstairs: Coordinated climbing motion
- Visual: All three axes show related patterns that move together

**0.5 - Medium Coordination**
- Some axes show coordination, others independent
- Partial relationship between movements
- Examples:
  - Standing with arm movements
  - Walking with independent upper body motion
  - Mixed coordinated and uncoordinated movements
- Visual: 2 axes might correlate, 1 is independent

**0.0 - No Coordination**
- Axes move independently with no clear relationship
- Random or minimal movement across axes
- Examples:
  - Sitting still (minimal random movements)
  - Standing (small independent adjustments)
  - Very chaotic unstructured movement
- Visual: Each axis looks unrelated to the others

---

## Labeling Process

### Step-by-Step:
1. **Visualize** the 3-second window (plot X, Y, Z axes)
2. **Periodicity**: Count cycles → regular? → 0.0, 0.5, or 1.0
3. **Temporal Stability**: Does pattern change? → 0.0, 0.5, or 1.0
4. **Coordination**: Do axes move together? → 0.0, 0.5, or 1.0
5. **Double-check**: Review all 3 labels before saving

### Red Flags (When to discuss with team):
- ⚠️ Unclear which value (0.0 vs 0.5, or 0.5 vs 1.0)
- ⚠️ Pattern seems to fit multiple categories
- ⚠️ Very unusual/ambiguous movement
- ⚠️ Data quality issues (noise, gaps)

---

## Quick Reference Table

| Activity | Typical Periodicity | Typical Temporal | Typical Coordination |
|----------|---------------------|------------------|---------------------|
| Walking | 0.5 - 1.0 | 0.5 - 1.0 | 0.5 - 1.0 |
| Jogging | 1.0 | 0.5 - 1.0 | 1.0 |
| Sitting | 0.0 | 1.0 | 0.0 - 0.5 |
| Standing | 0.0 | 0.5 - 1.0 | 0.0 - 0.5 |
| Upstairs | 0.5 - 1.0 | 0.5 | 0.5 - 1.0 |
| Downstairs | 0.5 - 1.0 | 0.5 | 0.5 - 1.0 |

**Note**: These are GUIDELINES, not rules. Trust the data, not the table!

---

## Edge Cases & How to Handle Them

**1. Window spans two activities** (e.g., sitting → standing)
   → Label based on the MAJORITY of the window
   → If 50/50: Mark for team discussion

**2. Very short movements** (< 1 second)
   → Consider temporal stability = 0.0 (unstable)
   → Label the dominant pattern if clear

**3. Noisy data** (sensor glitches)
   → If movement is still clear: label it
   → If completely obscured: Flag for review

**4. Uncertain labels** (really can't decide)
   → Mark with comment for team review
   → Don't guess - ask the team!

---

## Quality Control Checkpoints

- **After first 10 windows**: Compare with 1 other person
- **After 25 windows**: Team check-in on disagreements
- **After 50 windows**: Calculate Cohen's Kappa (target > 0.75)

---

## Contact Protocol

**If unsure about a label:**
1. Add comment in your sheet: "Window X - unclear periodicity"
2. Continue labeling (don't block on one window)
3. Batch discussion at end of session

**Daily sync** (15 min):
- Review flagged windows
- Discuss systematic differences
- Update guidelines if needed

