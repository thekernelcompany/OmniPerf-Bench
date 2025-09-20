
from app import test_h100, test_a100, test_l40s

SCRIPT = "working_test_generators/0f40557a_test_case_generator.py"
ARGS_H100 = ["--reference", "--json-out", f"/results/b690e34824fd5a5c4054a0c0468ebfb6aa1dd215-H100.json"]
ARGS_A100 = ["--reference", "--json-out", f"/results/b690e34824fd5a5c4054a0c0468ebfb6aa1dd215-A100.json"]
ARGS_L40S = ["--reference", "--json-out", f"/results/b690e34824fd5a5c4054a0c0468ebfb6aa1dd215-L40S.json"]

print(test_h100.remote(SCRIPT, ARGS_H100))
print(test_a100.remote(SCRIPT, ARGS_A100))
print(test_l40s.remote(SCRIPT, ARGS_L40S))
