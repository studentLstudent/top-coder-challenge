# Reverse Engineering Analysis

## CRITICAL REQUIREMENT
**The system IS CONFIRMED to be an analytical function with a perfect solution (score = 0 on eval.sh)**
- This means NO randomness
- This means there IS an exact formula
- Goal: Get 1000/1000 exact matches on eval.sh

## Progress Tracking
- Current score: Not yet tested
- Target score: 0.00 (perfect match on all 1000 cases)

## Key Findings
1. System is deterministic (no duplicate inputs with different outputs)
2. Mileage rates are tiered (decreasing with distance)
3. Receipt treatment is non-linear (penalties for small amounts, partial reimbursement for large)

## Next Steps
1. Find exact breakpoints for mileage tiers
2. Find exact receipt handling formula
3. Identify any "bugs" or quirks that must be preserved
4. Implement in run.sh
5. Iterate until eval.sh score = 0