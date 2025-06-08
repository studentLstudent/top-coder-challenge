import json
import numpy as np

def load_data(filename='public_cases.json'):
    """Load test cases from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_similar_trips():
    """Find trips with similar characteristics to isolate variables."""
    data = load_data()
    
    print("=== ISOLATING VARIABLES ===\n")
    
    # Find trips with same days and miles, different receipts
    print("Same days/miles, different receipts:")
    trips_by_key = {}
    for i, case in enumerate(data):
        inp = case['input']
        key = (inp['trip_duration_days'], inp['miles_traveled'])
        if key not in trips_by_key:
            trips_by_key[key] = []
        trips_by_key[key].append((i, inp['total_receipts_amount'], case['expected_output']))
    
    for key, trips in trips_by_key.items():
        if len(trips) > 1:
            days, miles = key
            print(f"\nDays={days}, Miles={miles}:")
            trips.sort(key=lambda x: x[1])  # Sort by receipts
            for idx, receipts, output in trips:
                print(f"  Case {idx}: Receipts=${receipts:.2f} -> Output=${output:.2f}")
    
    # Analyze very simple cases
    print("\n\n=== SIMPLE CASES (1 day, low miles, low receipts) ===")
    for i, case in enumerate(data[:50]):
        inp = case['input']
        if inp['trip_duration_days'] == 1 and inp['miles_traveled'] < 100 and inp['total_receipts_amount'] < 50:
            days = inp['trip_duration_days']
            miles = inp['miles_traveled']
            receipts = inp['total_receipts_amount']
            output = case['expected_output']
            
            # Try to decompose the output
            base_per_diem = 100
            mileage = miles * 0.58
            receipt_portion = receipts
            
            calculated = base_per_diem + mileage + receipt_portion
            diff = output - calculated
            
            print(f"Case {i}: D={days}, M={miles}, R=${receipts:.2f}")
            print(f"  Output=${output:.2f}")
            print(f"  Base calc: $100 + ${miles}*0.58 + ${receipts:.2f} = ${calculated:.2f}")
            print(f"  Difference: ${diff:.2f}")

def check_integer_patterns():
    """Check if outputs follow integer cent patterns."""
    data = load_data()
    
    print("\n\n=== CHECKING INTEGER PATTERNS ===")
    
    # Check if all outputs are exact cents
    all_cents = True
    for case in data:
        output = case['expected_output']
        cents = round(output * 100)
        if abs(output - cents/100) > 0.0001:
            all_cents = False
            print(f"Non-cent output found: ${output}")
            break
    
    if all_cents:
        print("All outputs are exact cents!")
    
    # Check for patterns in cent values
    cent_endings = {}
    for case in data[:100]:
        output = case['expected_output']
        cents = int(round(output * 100))
        ending = cents % 100
        if ending not in cent_endings:
            cent_endings[ending] = 0
        cent_endings[ending] += 1
    
    print("\nMost common cent endings:")
    for ending, count in sorted(cent_endings.items(), key=lambda x: -x[1])[:10]:
        print(f"  .{ending:02d}: {count} times")

def test_receipt_transformation():
    """Test different receipt transformation hypotheses."""
    data = load_data()
    
    print("\n\n=== TESTING RECEIPT TRANSFORMATIONS ===")
    
    # Test hypothesis: receipts might be capped at certain amount
    for max_receipt in [1000, 1500, 2000]:
        errors = []
        for case in data:
            inp = case['input']
            days = inp['trip_duration_days']
            miles = inp['miles_traveled']
            receipts = min(inp['total_receipts_amount'], max_receipt)
            
            # Simple calculation
            output_calc = days * 100 + miles * 0.58 + receipts * 0.8
            error = abs(case['expected_output'] - output_calc)
            errors.append(error)
        
        avg_error = np.mean(errors)
        print(f"Receipt cap at ${max_receipt}: avg error=${avg_error:.2f}")
    
    # Test hypothesis: receipts have different rates based on amount
    print("\nTesting tiered receipt rates:")
    for case in data[:20]:
        inp = case['input']
        receipts = inp['total_receipts_amount']
        output = case['expected_output']
        
        # Calculate base (without receipts)
        base = inp['trip_duration_days'] * 100 + inp['miles_traveled'] * 0.58
        receipt_contribution = output - base
        
        if receipts > 0:
            rate = receipt_contribution / receipts
            print(f"Receipts=${receipts:.2f}, contribution=${receipt_contribution:.2f}, rate={rate:.2f}")

if __name__ == "__main__":
    analyze_similar_trips()
    check_integer_patterns()
    test_receipt_transformation()