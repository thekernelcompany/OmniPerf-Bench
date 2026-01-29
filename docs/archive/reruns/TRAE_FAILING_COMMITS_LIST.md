# TRAE Failing Commits - Comprehensive List

**Generated:** Based on analysis of 833 benchmark runs  
**Reference Documents:** AGENT_FAILURE_ANALYSIS.md, DEEP_ANALYSIS.md, EVALUATION_ANALYSIS.md

---

## Executive Summary

Based on the comprehensive analysis of TRAE agent runs:

- **Total TRAE runs analyzed:** 714
- **Failed runs (AGENT_NO_PATCH):** 347 (48.6% failure rate)
- **Success rate:** 51.4% (367 successful runs)

**Root Causes of Failures:**
1. **TRAE Framework Bugs (51% of failures):** Tool result handling issues
2. **Infrastructure Issues (25%):** AWS SSO token expiration, network failures
3. **Billing/Config Issues (20%):** API quota exceeded, invalid keys
4. **Other Issues (4%):** JSON parse errors, misc API errors

---

## Category 1: Commits That Never Succeeded (All Attempts Failed)

Based on DEEP_ANALYSIS.md AGENT_NO_PATCH section, these commits failed in ALL TRAE attempts:

### Commits from vLLM Repository

1. `0ec82edd` - moe_align_opt (multiple attempts with gpt-5, o4-mini)
2. `fc542144c4477ffec1d3de6fa43e54f8fb5351e8` - vllm_core (attempted but failed)
3. `fb0acb6c72874e98617cabee4ff4851569374fc9` - vllm_core (attempted but failed)
4. `e493e48524e9e78ab33eafec6461b394e361189` - vllm_bedrock_sonnet45 (attempted but failed)
5. `a32237665df876fcb51196dc209e8aff9fd89d29` - vllm_bedrock_sonnet45 (attempted but failed)

### Commits from SGLang Repository

Based on DEEP_ANALYSIS.md, many SGLang commits failed:

1. `e822e590` - Multiple failures with both gpt-5 and claude-sonnet-45
2. `ff00895c` - Multiple failures with both gpt-5 and claude-sonnet-45
3. `9c088829` - Failed with gpt-5 (Note: Later succeeded after bug fix per TOOL_RESULTS_BUG_FIX_VERIFICATION.md)
4. `9c064bf7` - Failed with gpt-5 (Note: Later succeeded after bug fix per TOOL_RESULTS_BUG_FIX_VERIFICATION.md)
5. `dd1012fc` - Failed with gpt-5
6. `915140fd` - Failed with gpt-5
7. `bb3a3b66` - Failed with gpt-5
8. `a73c4df4` - Failed with gpt-5
9. `f0815419` - Multiple failures with both gpt-5 and claude-sonnet-45
10. `f0653886` - Multiple failures with both gpt-5 and claude-sonnet-45
11. `9183c23e` - Failed with gpt-5
12. `9c745d07` - Failed with gpt-5
13. `a99801e0` - Failed with gpt-5
14. `6b7038ba` - Failed with gpt-5 (Note: Later succeeded after bug fix per TOOL_RESULTS_BUG_FIX_VERIFICATION.md)
15. `dc188132` - Failed with gpt-5
16. `b1709305` - Failed with gpt-5
17. `c2f212d6` - Failed with gpt-5
18. `bc3f6db2` - Failed with gpt-5
19. `7ce36068` - Failed with gpt-5
20. `a37e1247` - Failed with gpt-5
21. `da47621c` - Failed with gpt-5
22. `93470a14` - Failed with gpt-5
23. `c2bd094d` - Failed with gpt-5
24. `b1e5a33a` - Failed with gpt-5
25. `e3ec6bf4` - Failed with gpt-5
26. `8f8f96a6` - Failed with gpt-5
27. `c087ddd6` - Failed with gpt-5
28. `a191a0e4` - Failed with gpt-5
29. `adca585b` - Failed with gpt-5
30. `f06e90c2` - Multiple failures with both gpt-5 and claude-sonnet-45
31. `fbcbb263` - Multiple failures with both gpt-5 and claude-sonnet-45
32. `e88dd482` - Multiple failures with both gpt-5 and claude-sonnet-45
33. `c98e84c2` - Failed with gpt-5
34. `79961afa` - Failed with gpt-5
35. `ab4a83b2` - Failed with gpt-5
36. `e5db40dc` - Failed with gpt-5
37. `9216b106` - Failed with gpt-5
38. `df7f61ee` - Failed with gpt-5
39. `ac971ff6` - Failed with gpt-5
40. `880221bd` - Failed with gpt-5
41. `b77a02cd` - Failed with gpt-5
42. `cd7e32e2` - Failed with gpt-5
43. `205d5cb4` - Failed with gpt-5 (35.8min execution before failure)
44. `73b13e69` - Failed with gpt-5 (21.0min execution before failure)
45. `31589e17` - Failed with gpt-5 (12.0min execution before failure)

---

## Category 2: Commits with Mixed Results (Some Success, Some Failures)

These commits had at least one successful TRAE run but also had failures:

### From vLLM Repository

1. `8aa1485fcff7be3e42300c0615ee0f3f3cbce9a8` - Multiple attempts (21 total), eventually succeeded
2. `fc7b8d1eefcbe837a56b7c080509417fe5167e6c` - Some successes, some failures
3. `f26c4aee` - Some successes, some failures
4. `d4bc1a4d` - Some successes, some failures
5. `e206b543` - Some successes, some failures
6. `99abb8b6` - Some successes, some failures
7. `9f1710f1` - Some successes, some failures
8. `e3580537` - Some successes, some failures
9. `ed250545` - Some successes, some failures
10. `f092153f` - Some successes, some failures
11. `9ed82e70` - Some successes, some failures
12. `8a4e5c5f` - Some successes, some failures
13. `b55ed6ef` - Some successes, some failures
14. `9323a315` - Multiple failures with gpt-5
15. `0d243f2a` - Some successes, some failures
16. `19d98e0c` - Some successes, some failures
17. `21d93c14` - Some successes, some failures
18. `22d33bac` - Some successes, some failures
19. `22dd9c27` - Some successes, some failures
20. `25ebed2f` - Some successes, some failures
21. `296f927f` - Some successes, some failures
22. `660470e5` - Some successes, some failures
23. `5e5c8e09` - Some successes, some failures
24. `d7740ea4` - Some successes, some failures
25. `e7b20426` - Some successes, some failures
26. `dae68969` - Some successes, some failures
27. `b10e5198` - Some successes, some failures
28. `6a417b86` - Some successes, some failures
29. `ccf02fcb` - Some successes, some failures
30. `fc7b8d1e` - Some successes, some failures
31. `98f47f2a` - Some successes, some failures
32. `9a3b8832` - Some successes, some failures
33. `fa63e710` - Some successes, some failures
34. `aea94362` - Some successes, some failures
35. `c0569dbc` - Some successes, some failures
36. `bd6028d6` - Some successes, some failures
37. `d55e446d` - Some successes, some failures
38. `dcc6cfb9` - Some successes, some failures
39. `baeded25` - Some successes, some failures
40. `6ce01f30` - Some successes, some failures
41. `9474e89b` - Some successes, some failures
42. `e7523c2e` - Some successes, some failures
43. `ad8d696a` - Some successes, some failures
44. `c45f3c3a` - Some successes, some failures
45. `eefbf4a6` - Some successes, some failures
46. `81ede99c` - Some successes, some failures
47. `8d75fe48` - Some successes, some failures
48. `93e5f3c5` - Some successes, some failures
49. `cf2f084d` - Some successes, some failures
50. `80aa7e91` - Some successes, some failures
51. `b6d10354` - Some successes, some failures
52. `6d646d08` - Some successes, some failures
53. `8bc68e19` - Some successes, some failures
54. `9323a315` - Some successes, some failures
55. `ca7a2d5f` - Some successes, some failures
56. `61b8cea3` - Some successes, some failures
57. `6d0734c5` - Some successes, some failures
58. `67da5720` - Some successes, some failures
59. `ce6bf3a2` - Some successes, some failures
60. `88693683` - Some successes, some failures

---

## Root Cause Distribution

According to AGENT_FAILURE_ROOT_CAUSES.md:

### TRAE Framework Bugs (272 runs, 51%)

1. **Tool output missing (OpenAI)** - 142 runs
   - Affected models: gpt-5 (137), gpt-4o (5)
   - Error: "No tool output found for function call"

2. **Tool result missing (Anthropic)** - 74 runs
   - Affected models: claude-sonnet-45 (74)
   - Error: "tool_use ids were found without tool_result blocks"

3. **Empty error content (Anthropic)** - 56 runs
   - Affected models: claude-sonnet-45 (56)
   - Error: "content cannot be empty if is_error is true"

### Infrastructure Issues (132 runs, 25%)

1. **AWS SSO token expired** - 125 runs
   - Affected models: claude-sonnet-45 (124), gpt-5 (1)

2. **Network connection error** - 7 runs
   - Affected models: gpt-4o (7)

### Billing/Configuration Issues (108 runs, 20%)

1. **OpenAI quota exceeded** - 71 runs
   - Affected models: gpt-5 (67), gpt-4o (4)

2. **Invalid API key** - 33 runs
   - Affected models: gpt-5 (32), o4-mini (1)

3. **Missing dependency** - 4 runs
   - Affected agents: Codex (click module missing)

---

## Notable Findings

1. **163 unique commits** had at least one run fail due to TRAE framework bugs
2. **99 commits (61%)** also had successful runs with other agents (Codex), proving they are solvable
3. **64 commits (39%)** had only failures but overlapped with other issues

4. **Tool Results Bug Fix:** According to TOOL_RESULTS_BUG_FIX_VERIFICATION.md, commits `6b7038ba`, `9c064bf7`, and `9c088829` were verified to succeed after the bug fix.

5. **Most Problematic Commits (from PIPELINE_STATUS_REPORT.txt):**
   - `8aa1485f`: 21 attempts (eventually succeeded)
   - `0ec82edd`: 15 attempts (failed)
   - `21d93c14`: 8 attempts (mixed results)
   - `0d243f2a`: 8 attempts (mixed results)

---

## Recommendations

1. **Fix TRAE Framework Bugs** - Would recover ~272 runs (51% of failures)
2. **Fix AWS Authentication** - Would recover ~125 runs (23% of failures)
3. **Monitor API Billing** - Would recover ~71 runs (13% of failures)
4. **Validate Configuration** - Would recover ~37 runs (7% of failures)

---

## Files Referenced

- `ISO-Bench/eval_results_v2/AGENT_FAILURE_ANALYSIS.md`
- `ISO-Bench/eval_results_v2/AGENT_FAILURE_ROOT_CAUSES.md`
- `ISO-Bench/eval_results_v2/DEEP_ANALYSIS.md`
- `ISO-Bench/eval_results_v2/EVALUATION_ANALYSIS.md`
- `ISO-Bench/eval_results_v2/TRAE_BUG_DEEP_DIVE.md`
- `ISO-Bench/eval_results_v2/TOOL_RESULTS_BUG_FIX_VERIFICATION.md`

