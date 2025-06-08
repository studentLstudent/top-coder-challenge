import json
import sys

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Calculate reimbursement using simplified formula without ML."""
    # All calculations in cents to avoid float precision issues
    days = int(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    # Base per diem 
    base_cents = days * 10000  # $100/day
    
    # Mileage calculation
    if miles <= 100:
        mileage_cents = int(miles * 58 + 0.5)
    else:
        mileage_cents = 5800  # First 100 miles
        remaining = miles - 100
        if remaining <= 200:
            mileage_cents += int(remaining * 55 + 0.5)
        else:
            mileage_cents += 11000  # Next 200 miles
            mileage_cents += int((remaining - 200) * 50 + 0.5)
    
    # Receipt handling
    if receipts == 0:
        receipt_contribution = 0
    elif receipts < 50:
        # Small receipts penalty
        receipt_contribution = int(receipts * 20)  # Only 20%
    elif receipts < 200:
        receipt_contribution = int(receipts * 70)  # 70%
    elif receipts < 500:
        receipt_contribution = int(receipts * 85)  # 85%
    elif receipts < 1000:
        receipt_contribution = int(receipts * 95)  # 95%
    else:
        # Diminishing returns for high receipts
        receipt_contribution = int(95000 + (receipts - 1000) * 50)  # 50% after $1000
    
    # Total base calculation
    total_cents = base_cents + mileage_cents + receipt_contribution
    
    # Efficiency bonus/penalty (miles per day)
    mpd = miles / max(days, 1)
    if 180 <= mpd <= 220:
        total_cents = int(total_cents * 1.15)  # 15% bonus
    elif 150 <= mpd < 180:
        total_cents = int(total_cents * 1.08)  # 8% bonus
    elif mpd > 300:
        total_cents = int(total_cents * 0.85)  # 15% penalty
    
    # Day-specific adjustments
    if days == 5:
        total_cents = int(total_cents * 1.06)  # 6% bonus for 5-day trips
    elif days > 10:
        total_cents = int(total_cents * 0.92)  # 8% penalty for long trips
    
    # Convert back to dollars
    return total_cents / 100.0

def main():
    if len(sys.argv) == 4:
        # Command line mode for eval.sh
        try:
            days = float(sys.argv[1])
            miles = float(sys.argv[2])
            receipts = float(sys.argv[3])
            result = calculate_reimbursement(days, miles, receipts)
            print(f"{result:.2f}")
        except Exception:
            sys.exit(1)
    else:
        print("Usage: python main_fast.py <days> <miles> <receipts>")
        sys.exit(1)

if __name__ == "__main__":
    main()