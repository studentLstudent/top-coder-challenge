#!/usr/bin/env python3
import re

# Read interviews
with open('INTERVIEWS.md', 'r') as f:
    text = f.read().lower()

# Search for key phrases as per THE BIG PLAN section 2
print("=== MINING INTERVIEWS FOR STRUCTURAL CLUES ===")

# Per diem mentions
per_diem_matches = re.findall(r'(per diem.*?[\.\n])', text)
print("\nPER DIEM MENTIONS:")
for match in per_diem_matches[:5]:
    print(f"  - {match.strip()}")

# Look for specific numbers that might be rates
print("\nSPECIFIC DOLLAR AMOUNTS MENTIONED:")
dollar_amounts = re.findall(r'\$(\d+)', text)
unique_amounts = sorted(set(int(x) for x in dollar_amounts))
print(f"  Amounts: {unique_amounts}")

# Mileage rate mentions
print("\nMILEAGE MENTIONS:")
mileage_mentions = re.findall(r'(mileage.*?[\.\n]|mile.*?rate.*?[\.\n]|cents per mile.*?[\.\n])', text)
for match in mileage_mentions[:5]:
    print(f"  - {match.strip()}")

# Look for specific numbers near "mile"
mile_numbers = re.findall(r'(\d+)\s*mile', text)
print(f"\nMileage thresholds mentioned: {sorted(set(int(x) for x in mile_numbers if int(x) < 2000))}")

# Receipt mentions
print("\nRECEIPT MENTIONS:")
receipt_mentions = re.findall(r'(receipt.*?[\.\n]|cap.*?[\.\n]|threshold.*?[\.\n])', text)
for match in receipt_mentions[:5]:
    print(f"  - {match.strip()}")

# Look for rounding mentions
print("\nROUNDING MENTIONS:")
rounding_mentions = re.findall(r'(round.*?[\.\n]|cent.*?[\.\n])', text)
for match in rounding_mentions[:3]:
    print(f"  - {match.strip()}")

# Look for specific trip length mentions
print("\nTRIP LENGTH PATTERNS:")
day_mentions = re.findall(r'(\d+)[-\s]day', text)
day_counts = {}
for d in day_mentions:
    if d in day_counts:
        day_counts[d] += 1
    else:
        day_counts[d] = 1
print(f"  Most mentioned day counts: {sorted(day_counts.items(), key=lambda x: -x[1])[:10]}")

# Sweet spot mentions
print("\nSWEET SPOT MENTIONS:")
sweet_mentions = re.findall(r'(sweet spot.*?[\.\n])', text)
for match in sweet_mentions:
    print(f"  - {match.strip()}")

# Efficiency mentions  
print("\nEFFICIENCY MENTIONS:")
efficiency_mentions = re.findall(r'(efficien.*?[\.\n]|miles per day.*?[\.\n])', text)
for match in efficiency_mentions[:3]:
    print(f"  - {match.strip()}")

# Bug/quirk mentions
print("\nBUG/QUIRK MENTIONS:")
bug_mentions = re.findall(r'(bug.*?[\.\n]|quirk.*?[\.\n]|weird.*?[\.\n])', text)
for match in bug_mentions[:3]:
    print(f"  - {match.strip()}")

# 58 cents specifically (standard mileage rate)
print("\nSTANDARD MILEAGE RATE (58 CENTS):")
rate_58 = re.findall(r'58 cent', text)
print(f"  Found {len(rate_58)} mentions of 58 cents/mile")

# Look for tiered/stepped mentions
print("\nTIERED/STEPPED MENTIONS:")
tiered_mentions = re.findall(r'(tier.*?[\.\n]|step.*?[\.\n]|drop.*?[\.\n])', text)
for match in tiered_mentions[:3]:
    print(f"  - {match.strip()}")