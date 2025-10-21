# 🚨 CRITICAL CLARIFICATION: Periodicity vs Temporal Stability

## The Key Difference (NEW DEFINITION)

### PERIODICITY = "Does a REPEATING CYCLE exist?"
**Focus**: Is there a PATTERN that REPEATS itself?
- Look for: Step-step-step, bounce-bounce-bounce, cycle-cycle-cycle
- **High (1.0)**: Clear repeating cycles (walking steps, jogging rhythm)
- **Medium (0.5)**: Some repetition but irregular intervals
- **Low (0.0)**: No repeating pattern (random movements, single transitions)

**Example - Walking**: 
- High Periodicity = You can count distinct steps repeating
- Low Periodicity = Just shifting weight once (no cycle)

---

### TEMPORAL STABILITY = "Does the ACTIVITY/INTENSITY remain constant?"
**Focus**: Is the person doing the SAME THING throughout the window?
- Look for: Changes in activity, speed, or intensity
- **High (1.0)**: Same activity, same intensity for full 3 seconds
- **Medium (0.5)**: One change (start/stop, speed up/slow down)
- **Low (0.0)**: Multiple changes or transitioning between activities

**Example - Walking**:
- High Stability = Walks at constant speed for full 3 seconds
- Low Stability = Starts walking, then stops midway

---

## KEY INSIGHT: These concepts are INDEPENDENT!

| Scenario | Periodicity | Temporal Stability |
|----------|-------------|-------------------|
| **Steady walking** | High (regular steps) | High (constant pace) |
| **Accelerating while walking** | High (still regular steps!) | Medium (speed changes) |
| **Standing still** | Low (no repeating pattern) | High (stays standing) |
| **Sitting → Standing** | Low (no cycle) | Low (activity changes) |
| **Jogging at varying pace** | High (still rhythmic!) | Low (pace varies) |

---

## Common Confusion Points

### ❌ WRONG: "Regular pattern = stable"
- **Periodicity**: Looks at SPATIAL pattern (are there cycles?)
- **Stability**: Looks at TEMPORAL consistency (does it stay the same?)

### ✅ CORRECT: They measure different things!
**You can have:**
- High periodicity + Low stability (running but changing speed)
- Low periodicity + High stability (standing still the whole time)
- High periodicity + High stability (steady rhythmic walking)
- Low periodicity + Low stability (transitioning between activities)

---

## Specific Labeling Instructions

### For PERIODICITY:
1. Count cycles in the signal
2. Are steps/bounces/patterns repeating?
3. Ignore changes in intensity - just look at SHAPE

### For TEMPORAL STABILITY:
1. Does the activity TYPE change?
2. Does the INTENSITY/SPEED change significantly?
3. Ignore whether there's a pattern - just look at CONSISTENCY

---

## Example Re-labeling Exercise

**Window 12 (Standing, 12.0-15.0s)** - Where team disagreed most:

**Current disagreement:**
- Periodicity: Everyone agrees (Low = 0.0) ✓
- Temporal Stability: Harveer=High, Maria=Low, Piotr=Low, Tudor=High
- Coordination: Harveer=High, Maria=High, Piotr=Low, Tudor=Low

**Correct labels should be:**
- **Periodicity = 0.0**: Standing has no repeating cycle
- **Temporal Stability = ?**: 
  - If standing still entire time → 1.0 (High)
  - If shifting/moving during window → 0.5 or 0.0
- **Coordination = ?**: Need to review actual accelerometer data

The disagreement suggests some saw stable standing, others saw movement during the window.

---

## ACTION ITEMS FOR TEAM

1. **Re-calibration meeting** (30 min):
   - Review these 5 windows together: 18, 12, 5, 10, 14
   - Look at the actual accelerometer plots
   - Discuss: "Is this PERIODIC?" vs "Is this STABLE?"

2. **Update mental model**:
   - Periodicity = PATTERN (cycles/repetitions)
   - Temporal Stability = CONSISTENCY (stays same vs changes)

3. **Practice labels** (10 windows):
   - Each person re-label windows 0-9
   - Compare answers
   - Calculate new κ score

4. **If κ still < 0.6**: Consider redefining or merging concepts

