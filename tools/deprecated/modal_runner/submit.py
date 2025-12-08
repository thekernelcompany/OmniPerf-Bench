import modal

SCRIPT = "generated_test_generators_v4/2deb029d_test_case_generator.py"
ARGS_H100 = ["--reference", "--json-out", f"/results/2deb029d115dadd012ce5ea70487a207cb025493-H100.json"]
ARGS_A100 = ["--reference", "--json-out", f"/results/2deb029d115dadd012ce5ea70487a207cb025493-A100.json"]
ARGS_L40S = ["--reference", "--json-out", f"/results/2deb029d115dadd012ce5ea70487a207cb025493-L40S.json"]

test_h100 = modal.Function.from_name("vllm-agent-tests-v5", "test_h100")
test_a100 = modal.Function.from_name("vllm-agent-tests-v5", "test_a100")
test_l40s = modal.Function.from_name("vllm-agent-tests-v5", "test_l40s")

print(test_h100.remote(SCRIPT, ARGS_H100))
print(test_a100.remote(SCRIPT, ARGS_A100))
print(test_l40s.remote(SCRIPT, ARGS_L40S))
