# Sanity check: eval_results_v2 speedups

Total improved results: 72\n\nCounts by confidence:
- high: 8
- med: 18
- low: 44
- bs: 2

Heuristics (quick):
- `bs`: opt path not hit or obvious mismatch
- `low`: <2% gain, non-exact eq, tiny baseline (<0.01ms), or very noisy
- `med`: >=5% gain but noisy / tails worse / mixed percentiles
- `high`: >=5% gain, exact eq, opt path hit, and percentiles improve

|conf|speedup|item_id|device|eq|run_id|baseline_ms|patched_ms|issues|path|
|---:|---:|---|---|---|---|---:|---:|---|---|
|high|1.9102x|vllm_core-0013|cpu|exact|vllm_core_codex-90a1c13f|1.74637|0.914236||vllm/codex/gpt-5/90a1c13f/vllm_core-0013/test_results.json|
|high|1.4145x|vllm_bedrock_sonnet45-0013|cpu|exact|vllm_claude_sonnet45_retry-5d58acda|1.1624|0.821769||vllm/trae/claude-sonnet-45/5d58acda/vllm_bedrock_sonnet45-0013/test_results.json|
|high|1.3095x|vllm_bedrock_sonnet45-0013|cpu|exact|vllm_claude_sonnet45-0a51aaa8|2.44839|1.86977||vllm/trae/claude-sonnet-45/0a51aaa8/vllm_bedrock_sonnet45-0013/test_results.json|
|high|1.2996x|vllm_bedrock_sonnet45-0066|cpu|exact|vllm_claude_sonnet45_retry-5d58acda|0.0391656|0.0301358||vllm/trae/claude-sonnet-45/5d58acda/vllm_bedrock_sonnet45-0066/test_results.json|
|high|1.2859x|vllm_bedrock_sonnet45-0026|cpu|exact|vllm_claude_sonnet45_retry-5d58acda|4.68938|3.64687||vllm/trae/claude-sonnet-45/5d58acda/vllm_bedrock_sonnet45-0026/test_results.json|
|high|1.2482x|vllm_core-0026|cpu|exact|vllm_core_codex-90a1c13f|4.47418|3.58456||vllm/codex/gpt-5/90a1c13f/vllm_core-0026/test_results.json|
|high|1.1477x|sglang_068_dd1012fc|cpu|exact|sglang_claude_sonnet45-c0645fb7|7.50286|6.5374||sglang/trae/claude-sonnet-45/c0645fb7/sglang_068_dd1012fc/test_results.json|
|high|1.0937x|vllm_core-0025|cpu|exact|vllm_core-beffe4cd|3.96476|3.62492||vllm/trae/gpt-5/beffe4cd/vllm_core-0025/test_results.json|
|med|1.3440x|vllm_core-0020|cuda|exact|vllm_core-a40b2039|0.11942|0.088857|baseline_noisy:1.27, patched_noisy:1.39|vllm/trae/gpt-5/a40b2039/vllm_core-0020/test_results.json|
|med|1.3385x|vllm_core-0020|cuda|exact|vllm_core-beffe4cd|0.120458|0.0899968|patched_noisy:1.46|vllm/trae/gpt-5/beffe4cd/vllm_core-0020/test_results.json|
|med|1.2859x|vllm_core-0021|cuda|exact|vllm_core_codex-90a1c13f|0.119824|0.0931802|patched_noisy:1.26|vllm/codex/gpt-5/90a1c13f/vllm_core-0021/test_results.json|
|med|1.2851x|vllm_bedrock_sonnet45-0021|cuda|exact|vllm_claude_sonnet45-0a51aaa8|0.112753|0.0877414|baseline_noisy:1.28, patched_noisy:1.35|vllm/trae/claude-sonnet-45/0a51aaa8/vllm_bedrock_sonnet45-0021/test_results.json|
|med|1.1161x|sglang_042_9216b106|cpu|exact|sglang_claude_sonnet45-c0645fb7|0.773087|0.692675|p50_not_better, p95_not_better|sglang/trae/claude-sonnet-45/c0645fb7/sglang_042_9216b106/test_results.json|
|med|1.0957x|vllm_core-0042|cuda|exact|vllm_core_codex-90a1c13f|4.47224|4.08154|baseline_noisy:1.38|vllm/codex/gpt-5/90a1c13f/vllm_core-0042/test_results.json|
|med|1.0932x|vllm_core-0056|cpu|exact|vllm_core_codex-90a1c13f|0.601513|0.55024|baseline_noisy:1.25|vllm/codex/gpt-5/90a1c13f/vllm_core-0056/test_results.json|
|med|1.0919x|vllm_core-0048|cpu|exact|vllm_core-9641716f|0.0115712|0.0105975|baseline_noisy:1.50|vllm/trae/gpt-5/9641716f/vllm_core-0048/test_results.json|
|med|1.0569x|vllm_core-0063|cpu|exact|vllm_core-9641716f|0.0307777|0.0291213|patched_noisy:1.35, p95_not_better|vllm/trae/gpt-5/9641716f/vllm_core-0063/test_results.json|
|med|1.0507x|sglang_042_9216b106|cpu|exact|sglang_core-bd68ff67|0.622131|0.592115|p50_not_better|sglang/trae/gpt-5/bd68ff67/sglang_042_9216b106/test_results.json|
|med|1.0425x|vllm_bedrock_sonnet45-0056|cpu|exact|vllm_claude_sonnet45_retry-5d58acda|0.587703|0.563744|p50_not_better|vllm/trae/claude-sonnet-45/5d58acda/vllm_bedrock_sonnet45-0056/test_results.json|
|med|1.0364x|sglang_core-0073|cpu|exact|sglang_core-389be848|76.5612|73.8702||sglang/codex/gpt-5/389be848/sglang_core-0073/test_results.json|
|med|1.0332x|vllm_core-0051|cpu|exact|vllm_core_codex-90a1c13f|0.010906|0.0105553||vllm/codex/gpt-5/90a1c13f/vllm_core-0051/test_results.json|
|med|1.0313x|sglang_025_62757db6|cpu|exact|sglang_core-ae58875a|0.231524|0.224494||sglang/trae/gpt-5/ae58875a/sglang_025_62757db6/test_results.json|
|med|1.0308x|sglang_032_6fc17596|cuda|exact|sglang_claude_sonnet45-c0645fb7|0.0374864|0.0363667|baseline_noisy:1.26|sglang/trae/claude-sonnet-45/c0645fb7/sglang_032_6fc17596/test_results.json|
|med|1.0277x|vllm_bedrock_sonnet45-0042|cuda|exact|vllm_claude_sonnet45_retry-5d58acda|4.25885|4.14396||vllm/trae/claude-sonnet-45/5d58acda/vllm_bedrock_sonnet45-0042/test_results.json|
|med|1.0250x|vllm_bedrock_sonnet45-0023|cpu|exact|vllm_claude_sonnet45-0a51aaa8|0.0106002|0.0103416||vllm/trae/claude-sonnet-45/0a51aaa8/vllm_bedrock_sonnet45-0023/test_results.json|
|med|1.0208x|vllm_core-0023|cpu|exact|vllm_core_codex-90a1c13f|0.0106058|0.0103898||vllm/codex/gpt-5/90a1c13f/vllm_core-0023/test_results.json|
|low|2.1696x|vllm_core-0041|cpu|numeric|vllm_core-beffe4cd|8.49659|3.91629|eq:numeric, baseline_noisy:1.38, patched_noisy:20.22|vllm/trae/gpt-5/beffe4cd/vllm_core-0041/test_results.json|
|low|1.7279x|vllm_bedrock_sonnet45-0044|cpu|numeric|vllm_claude_sonnet45_retry-5d58acda|1.92277|1.11276|eq:numeric, baseline_noisy:11.95, patched_noisy:6.74, p50_not_better|vllm/trae/claude-sonnet-45/5d58acda/vllm_bedrock_sonnet45-0044/test_results.json|
|low|1.2301x|sglang_007_1bf1cf19|cpu|behavioral|sglang_claude_sonnet45-c0645fb7|0.0307775|0.0250202|eq:behavioral|sglang/trae/claude-sonnet-45/c0645fb7/sglang_007_1bf1cf19/test_results.json|
|low|1.1589x|sglang_core-0029|cpu|behavioral|sglang_core-389be848|0.00139667|0.00120519|eq:behavioral, tiny_baseline, baseline_noisy:1.39|sglang/codex/gpt-5/389be848/sglang_core-0029/test_results.json|
|low|1.0569x|sglang_049_a73c4df4|cpu|numeric|sglang_core-bd68ff67|74.1773|70.1858|eq:numeric|sglang/trae/gpt-5/bd68ff67/sglang_049_a73c4df4/test_results.json|
|low|1.0503x|sglang_035_7ce36068|cuda|numeric|sglang_claude_sonnet45-c0645fb7|0.86608|0.824626|eq:numeric|sglang/trae/claude-sonnet-45/c0645fb7/sglang_035_7ce36068/test_results.json|
|low|1.0459x|vllm_core-0059|cuda|numeric|vllm_core_codex-90a1c13f|0.0186131|0.0177958|eq:numeric, baseline_noisy:1.40|vllm/codex/gpt-5/90a1c13f/vllm_core-0059/test_results.json|
|low|1.0421x|sglang_core-0036|cuda|numeric|sglang_core-389be848|0.847573|0.813371|eq:numeric, baseline_noisy:1.28|sglang/codex/gpt-5/389be848/sglang_core-0036/test_results.json|
|low|1.0296x|sglang_022_5239d795|cuda|numeric|sglang_core-ae58875a|0.0112493|0.0109254|eq:numeric, baseline_noisy:1.42, p50_not_better|sglang/trae/gpt-5/ae58875a/sglang_022_5239d795/test_results.json|
|low|1.0264x|sglang_067_dc67d976|cpu|numeric|sglang_claude_sonnet45-c0645fb7|8126.51|7917.58|eq:numeric|sglang/trae/claude-sonnet-45/c0645fb7/sglang_067_dc67d976/test_results.json|
|low|1.0200x|sglang_001_09deb20d|cuda|numeric|sglang_core-ae58875a|0.0341222|0.0334547|eq:numeric, small, baseline_noisy:1.31|sglang/trae/gpt-5/ae58875a/sglang_001_09deb20d/test_results.json|
|low|1.0198x|vllm_bedrock_sonnet45-0051|cpu|exact|vllm_claude_sonnet45_retry-5d58acda|0.0106542|0.0104476|small|vllm/trae/claude-sonnet-45/5d58acda/vllm_bedrock_sonnet45-0051/test_results.json|
|low|1.0190x|sglang_058_bc3f6db2|cpu|behavioral|sglang_claude_sonnet45-c0645fb7|7.24045|7.1051|eq:behavioral, small|sglang/trae/claude-sonnet-45/c0645fb7/sglang_058_bc3f6db2/test_results.json|
|low|1.0161x|vllm_core-0096|cuda|numeric|vllm_core_codex-90a1c13f|0.184762|0.181826|eq:numeric, small|vllm/codex/gpt-5/90a1c13f/vllm_core-0096/test_results.json|
|low|1.0160x|sglang_034_79961afa|cuda|exact|sglang_claude_sonnet45-c0645fb7|0.0207985|0.0204711|small|sglang/trae/claude-sonnet-45/c0645fb7/sglang_034_79961afa/test_results.json|
|low|1.0127x|sglang_core-0031|cpu|behavioral|sglang_core-389be848|2190.73|2163.24|eq:behavioral, small|sglang/codex/gpt-5/389be848/sglang_core-0031/test_results.json|
|low|1.0124x|sglang_core-0034|cpu|behavioral|sglang_core-389be848|1.47569|1.45756|eq:behavioral, small|sglang/codex/gpt-5/389be848/sglang_core-0034/test_results.json|
|low|1.0090x|sglang_core-0017|cuda|exact|sglang_core-389be848|0.113215|0.112205|near_1, baseline_noisy:1.26, p50_not_better|sglang/codex/gpt-5/389be848/sglang_core-0017/test_results.json|
|low|1.0083x|vllm_core-0005|cpu|exact|vllm_core-4be69dfd|1.50489|1.49254|near_1|vllm/trae/gpt-5/4be69dfd/vllm_core-0005/test_results.json|
|low|1.0083x|vllm_core-0039|cuda|exact|vllm_core-beffe4cd|4.25266|4.21781|near_1, p95_not_better|vllm/trae/gpt-5/beffe4cd/vllm_core-0039/test_results.json|
|low|1.0071x|sglang_030_6e2da515|cpu|behavioral|sglang_claude_sonnet45-c0645fb7|2171.32|2155.91|eq:behavioral, near_1|sglang/trae/claude-sonnet-45/c0645fb7/sglang_030_6e2da515/test_results.json|
|low|1.0066x|sglang_033_73b13e69|cpu|behavioral|sglang_claude_sonnet45-c0645fb7|1.51861|1.50865|eq:behavioral, near_1|sglang/trae/claude-sonnet-45/c0645fb7/sglang_033_73b13e69/test_results.json|
|low|1.0063x|vllm_bedrock_sonnet45-0008|cpu|exact|vllm_claude_sonnet45-0a51aaa8|0.0190041|0.0188858|near_1, patched_noisy:1.27, p95_not_better|vllm/trae/claude-sonnet-45/0a51aaa8/vllm_bedrock_sonnet45-0008/test_results.json|
|low|1.0062x|sglang_core-0069|cpu|exact|sglang_core-389be848|6.62083|6.58008|near_1|sglang/codex/gpt-5/389be848/sglang_core-0069/test_results.json|
|low|1.0054x|vllm_bedrock_sonnet45-0059|cuda|numeric|vllm_claude_sonnet45_retry-5d58acda|0.018176|0.0180781|eq:numeric, near_1, patched_noisy:1.51, p95_not_better|vllm/trae/claude-sonnet-45/5d58acda/vllm_bedrock_sonnet45-0059/test_results.json|
|low|1.0052x|sglang_core-0023|cuda|numeric|sglang_core-389be848|0.0106432|0.0105882|eq:numeric, near_1|sglang/codex/gpt-5/389be848/sglang_core-0023/test_results.json|
|low|1.0051x|vllm_core-0025|cpu|exact|vllm_core-a40b2039|3.59784|3.57958|near_1|vllm/trae/gpt-5/a40b2039/vllm_core-0025/test_results.json|
|low|1.0043x|sglang_016_2bd18e2d|cuda|exact|sglang_claude_sonnet45-c0645fb7|0.116397|0.115897|near_1, baseline_noisy:1.28, p50_not_better|sglang/trae/claude-sonnet-45/c0645fb7/sglang_016_2bd18e2d/test_results.json|
|low|1.0038x|vllm_core-0007|cpu|exact|vllm_core-9641716f|0.018795|0.0187234|near_1, baseline_noisy:1.27, patched_noisy:1.31, p95_not_better|vllm/trae/gpt-5/9641716f/vllm_core-0007/test_results.json|
|low|1.0026x|vllm_core-0067|cuda|numeric|vllm_core-9641716f|1.5271|1.52313|eq:numeric, near_1|vllm/trae/gpt-5/9641716f/vllm_core-0067/test_results.json|
|low|1.0024x|vllm_core-0056|cuda|numeric|vllm_core-9641716f|0.0179514|0.0179078|eq:numeric, near_1, baseline_noisy:1.37, p50_not_better|vllm/trae/gpt-5/9641716f/vllm_core-0056/test_results.json|
|low|1.0022x|sglang_core-0068|cpu|numeric|sglang_core-389be848|7196.34|7180.21|eq:numeric, near_1|sglang/codex/gpt-5/389be848/sglang_core-0068/test_results.json|
|low|1.0020x|sglang_045_9c088829|cuda|behavioral|sglang_claude_sonnet45-c0645fb7|560.963|559.86|eq:behavioral, near_1, baseline_noisy:1.73, patched_noisy:1.71|sglang/trae/claude-sonnet-45/c0645fb7/sglang_045_9c088829/test_results.json|
|low|1.0018x|sglang_067_dc67d976|cpu|numeric|sglang_core-bd68ff67|7008.41|6996.1|eq:numeric, near_1|sglang/trae/gpt-5/bd68ff67/sglang_067_dc67d976/test_results.json|
|low|1.0016x|sglang_core-0045|cuda|numeric|sglang_core-389be848|0.044224|0.0441517|eq:numeric, near_1, baseline_noisy:1.27, patched_noisy:1.30, p95_not_better|sglang/codex/gpt-5/389be848/sglang_core-0045/test_results.json|
|low|1.0015x|sglang_016_2bd18e2d|cuda|exact|sglang_core-ae58875a|0.116444|0.116268|near_1, p50_not_better|sglang/trae/gpt-5/ae58875a/sglang_016_2bd18e2d/test_results.json|
|low|1.0011x|sglang_core-0038|cpu|numeric|sglang_core-389be848|81.8626|81.7692|eq:numeric, near_1|sglang/codex/gpt-5/389be848/sglang_core-0038/test_results.json|
|low|1.0009x|sglang_core-0002|cuda|numeric|sglang_core-389be848|0.0345933|0.0345606|eq:numeric, near_1, baseline_noisy:1.26, patched_noisy:1.34, p95_not_better|sglang/codex/gpt-5/389be848/sglang_core-0002/test_results.json|
|low|1.0008x|vllm_core-0018|cuda|numeric|vllm_core-beffe4cd|11.2173|11.2083|eq:numeric, near_1|vllm/trae/gpt-5/beffe4cd/vllm_core-0018/test_results.json|
|low|1.0008x|sglang_017_2ed68d7a|cpu|exact|sglang_claude_sonnet45-c0645fb7|3.71691|3.71398|near_1|sglang/trae/claude-sonnet-45/c0645fb7/sglang_017_2ed68d7a/test_results.json|
|low|1.0008x|sglang_027_6b231325|cpu|behavioral|sglang_core-ae58875a|4014.69|4011.64|eq:behavioral, near_1|sglang/trae/gpt-5/ae58875a/sglang_027_6b231325/test_results.json|
|low|1.0006x|sglang_core-0018|cpu|exact|sglang_core-389be848|3.73631|3.73394|near_1, p50_not_better|sglang/codex/gpt-5/389be848/sglang_core-0018/test_results.json|
|low|1.0004x|sglang_031_6f560c76|cpu|behavioral|sglang_claude_sonnet45-c0645fb7|0.920608|0.920204|eq:behavioral, near_1, p95_not_better|sglang/trae/claude-sonnet-45/c0645fb7/sglang_031_6f560c76/test_results.json|
|low|1.0000x|vllm_core-0095|cuda|numeric|vllm_core-9641716f|0.695812|0.695811|eq:numeric, near_1, p95_not_better|vllm/trae/gpt-5/9641716f/vllm_core-0095/test_results.json|
|bs|1.0243x|vllm_bedrock_sonnet45-0016|cuda|behavioral|vllm_claude_sonnet45-0a51aaa8|18.8084|18.3629|opt_path_not_hit, eq:behavioral|vllm/trae/claude-sonnet-45/0a51aaa8/vllm_bedrock_sonnet45-0016/test_results.json|
|bs|1.0079x|vllm_core-0016|cuda|behavioral|vllm_core_codex-90a1c13f|18.8811|18.7328|opt_path_not_hit, eq:behavioral, near_1|vllm/codex/gpt-5/90a1c13f/vllm_core-0016/test_results.json|
