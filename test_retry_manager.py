
from decision_engine.retry_manager import RetryManager

manager = RetryManager(max_attempts=3)

# Trial 1
print("\n--- TRIAL 1 ---")
print("Attempt:", manager.start_attempt())
print("Result:", manager.finish_attempt("INCONCLUSIVE"))

# Trial 2
print("\n--- TRIAL 2 ---")
print("Attempt:", manager.start_attempt())
print("Result:", manager.finish_attempt("INCONCLUSIVE"))

# Trial 3
print("\n--- TRIAL 3 ---")
print("Attempt:", manager.start_attempt())
print("Result:", manager.finish_attempt("INCONCLUSIVE"))

# Check that a fourth attempt is not allowed
print("\n--- TRIAL 4 ---")
print("Attempt:", manager.start_attempt())
