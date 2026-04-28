import sys
sys.path.append('d:/Work/PROJECTS/flowstate-ai-v2')

from main import Engine

# Create engine
engine = Engine()

# Run healthy test simulation
results = engine.simulate(
    age=41, sex=1, sbp=115, dbp=75, waist_cm=85, heart_rate=65,
    egfr=95, pack_years=0, alcohol_g_day=5, sodium_mg_day=2300, years=30
)

# Print trajectory at 5-year intervals
print("Healthy Male Profile (Age 41, SBP 115, eGFR 95)")
print("=" * 60)
print(f"{'Year':<6} {'SBP':<8} {'DBP':<8} {'PP':<8} {'eGFR':<10} {'Survival':<10}")
print("-" * 60)
for i in range(0, 31, 5):
    if i < len(results['year']):
        print(f"{results['year'][i]:<6} {results['sbp'][i]:<8.1f} {results['dbp'][i]:<8.1f} "
              f"{results['pp'][i]:<8.1f} {results['egfr'][i]:<10.1f} {results['survival'][i]:<10.1f}%")

# Run intervention simulation
from intervention import get_intervention
ace = get_intervention("ACE Inhibitor")
ace.start_year = 5

baseline, treated = engine.simulate_with_intervention(
    age=41, sex=1, sbp=115, dbp=75, waist_cm=85, heart_rate=65,
    egfr=95, pack_years=0, alcohol_g_day=5, sodium_mg_day=2300, 
    intervention=ace, years=30
)

print("\n\nACE Inhibitor vs Baseline (from year 5)")
print("=" * 60)
print(f"{'Year':<6} {'SBP_B':<8} {'SBP_T':<8} {'eGFR_B':<10} {'eGFR_T':<10} {'Surv_B':<10} {'Surv_T':<10}")
print("-" * 60)
for i in range(0, 31, 5):
    if i < len(baseline['year']):
        print(f"{baseline['year'][i]:<6} {baseline['sbp'][i]:<8.1f} {treated['sbp'][i]:<8.1f} "
              f"{baseline['egfr'][i]:<10.1f} {treated['egfr'][i]:<10.1f} "
              f"{baseline['survival'][i]:<10.1f}% {treated['survival'][i]:<10.1f}%")
