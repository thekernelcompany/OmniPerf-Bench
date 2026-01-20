# SGLang Docker Image Analysis

**Date:** 2026-01-20
**Total Commits on DockerHub:** 71 unique SGLang commits
**Working Images:** 14 (as of Phase 6)

---

## Executive Summary

| Category | Count | Status |
|----------|-------|--------|
| Total unique commits | 71 | On DockerHub |
| Working images (confirmed GPU) | 5 | `021f76e4`, `777688b8`, `c087ddd6`, `9c745d07-v3`, `10189d08-v3` |
| `-src` images (GPU validated today) | 9 | **ALL 9 WORKING** |
| v0.1.x commits | ~15 | **UNFIXABLE** - skip these |
| v0.4.x+ commits | ~42 | Requires different build approach |
| Remaining buildable | ~5 | v0.3.5+ with vllm 0.6.x |

---

## Commit Categories by Version

### v0.1.x Commits (UNFIXABLE - SKIP)

**Date Range:** 2024-01 to 2024-07
**Count:** ~15 commits
**Status:** **DO NOT BUILD** - Multiple rebuild attempts failed

| Commit | Version | vLLM Required | Issue |
|--------|---------|---------------|-------|
| 6f560c76 | v0.1.9 | >=0.2.5 | outlines module missing |
| 09deb20d | v0.1.14 | >=0.4.2 | libcuda.so.1 hard import |
| 9216b106 | v0.1.14 | >=0.3.3 | vllm API incompatibility |
| 2a754e57 | v0.1.17 | ==0.5.0 | vllm LoadConfig missing |
| 564a898a | v0.1.20 | ==0.5.1 | outlines.fsm.guide missing |
| 6a2941f4 | v0.1.20 | ==0.5.1 | outlines.fsm.guide missing |
| ac971ff6 | v0.1.21 | ==0.5.1 | outlines.fsm.guide missing |

**Root Causes (from GPU_TESTING_RESULTS.md):**
1. Hard CUDA imports at module load time
2. vllm API functions removed in newer versions
3. outlines library API changed completely

**Conclusion:** Skip all v0.1.x commits. They require source code modifications.

---

### v0.2.x - v0.3.4 Commits (BUILT FROM SOURCE)

**Date Range:** 2024-08 to 2024-11
**Count:** 9 commits
**Status:** ✅ **ALL 9 BUILT AND TESTED WORKING**

| Commit | Version | vLLM | Status | Tag |
|--------|---------|------|--------|-----|
| 62757db6 | v0.2.11 | 0.5.4 | ✅ GPU WORKING | `-src` |
| ab4a83b2 | v0.3.0 | 0.5.5 | ✅ GPU WORKING | `-src` |
| 2854a5ea | v0.3.1.post3 | 0.5.5 | ✅ GPU WORKING | `-src` |
| c98e84c2 | v0.3.2 | 0.5.5 | ✅ GPU WORKING | `-src` |
| 9c064bf7 | v0.3.2 | 0.5.5 | ✅ GPU WORKING | `-src` |
| e5db40dc | v0.3.3.post1 | 0.5.5 | ✅ GPU WORKING | `-src` |
| b1709305 | v0.3.3.post1 | 0.5.5 | ✅ GPU WORKING | `-src` |
| 8f8f96a6 | v0.3.4.post1 | 0.5.5 | ✅ GPU WORKING | `-src` |
| b77a02cd | v0.3.4.post2 | 0.5.5 | ✅ GPU WORKING | `-src` |

**Patches Applied:**
- Event loop fix: `asyncio.get_event_loop()` → `asyncio.new_event_loop()`
- Model registry fix: Disabled duplicate assertion
- outlines patch: Removed types module import

---

### v0.3.5 - v0.3.6 Commits (ALREADY WORKING)

**Date Range:** 2024-11
**Count:** 2 commits
**Status:** ✅ **WORKING** (v3 images)

| Commit | Version | vLLM | Status | Tag |
|--------|---------|------|--------|-----|
| 9c745d07 | v0.3.5.post2 | 0.6.3.post1 | ✅ GPU WORKING | `-v3` |
| 10189d08 | v0.3.6 | >=0.6.3.post1 | ✅ GPU WORKING | `-v3` |

**Note:** These use vllm 0.6.x, different from the v0.5.x used in `-src` builds.

---

### v0.4.x+ Commits (REQUIRES NEW BUILD APPROACH)

**Date Range:** 2025-01 to 2025-06
**Count:** ~42 commits
**Status:** ⚠️ **NOT YET BUILDABLE**

| Commit | Version | Date | Issue |
|--------|---------|------|-------|
| 9183c23e | v0.4.1.post3 | 2025-01-02 | sgl_kernel ABI mismatch |
| ddcf9fe3 | v0.4.3.post2 | 2025-02-21 | sgl_kernel ABI mismatch |
| d1112d85 | v0.4.4.post1 | 2025-03-16 | Different deps |
| 93470a14 | v0.4.x | 2025-04-07 | Different deps |
| 005aad32 | v0.4.x | 2025-04-27 | Different deps |
| 1acca3a2 | v0.4.6.post2 | 2025-05-02 | sgl_kernel ABI |
| 79961afa | v0.4.6.post2 | 2025-05-07 | sgl_kernel ABI |
| c087ddd6 | latest | 2025-05-28 | ✅ WORKING (original) |
| f4a8987f | v0.4.6.post5 | 2025-05-28 | sgl_kernel ABI |
| b1e5a33a | v0.4.6.post5 | 2025-06-09 | sgl_kernel ABI |
| e3ec6bf4 | latest | 2025-06-13 | Different deps |

**Root Cause:** v0.4.x uses `sgl_kernel` which must be compiled from source with GPU. Pre-built wheels have ABI mismatch.

**Fix Required:** Build sgl_kernel from source during Docker build (requires GPU).

---

## Remaining Commits (Not Yet Analyzed)

These commits haven't been categorized yet:

```
005aad32  021f76e4  05b3bf5e  22a6b9fc  2bd18e2d  30643fed
3212c2ad  33b242df  42a2d82b  45d6592d  48efec7b  53475674
58d1082e  5ab20cce  5d264a90  6252ade9  62f15eea  6ea1e6ac
6fc17596  73fa2d49  777688b8  7ce36068  83452dbb  8c7279c2
96c503eb  9c088829  9d5fa68b  a191a0e4  a37e1247  ad506a4e
b04df75a  bb3a3b66  ca4f1ab8  cd687233  cfca4e0e  da19434c
da47621c  db452760  e1792cca  e822e590  ebaa2f31  efb099cd
```

---

## Working Images Summary

### Confirmed Working (14 images)

| Tag | Version | Source |
|-----|---------|--------|
| `021f76e4` | latest | Original |
| `777688b8` | latest | Original |
| `c087ddd6` | latest | Original |
| `9c745d07-v3` | v0.3.5.post2 | v3 rebuild |
| `10189d08-v3` | v0.3.6 | v3 rebuild |
| `62757db6-src` | v0.2.11 | Source build |
| `ab4a83b2-src` | v0.3.0 | Source build |
| `2854a5ea-src` | v0.3.1.post3 | Source build |
| `c98e84c2-src` | v0.3.2 | Source build |
| `9c064bf7-src` | v0.3.2 | Source build |
| `e5db40dc-src` | v0.3.3.post1 | Source build |
| `b1709305-src` | v0.3.3.post1 | Source build |
| `8f8f96a6-src` | v0.3.4.post1 | Source build |
| `b77a02cd-src` | v0.3.4.post2 | Source build |

---

## Recommendations

### Priority 1: Use Existing Working Images
For benchmarking, use the 14 confirmed working images above.

### Priority 2: Skip v0.1.x
Do NOT attempt to build v0.1.x images. They've been tested multiple times and are unfixable without source code modifications.

### Priority 3: v0.4.x Builds (Future Work)
Building v0.4.x images requires:
1. A machine with GPU for sgl_kernel compilation
2. Different base image (likely cuda:12.4+)
3. Building sgl_kernel from source: `cd /sglang/sgl-kernel && pip install -e . --no-build-isolation`

### Priority 4: Analyze Remaining Commits
~40 commits haven't been categorized. Most are likely v0.4.x+ based on dates.

---

## Version Detection Script

```bash
#!/bin/bash
# Check SGLang version for a commit
cd /tmp/sglang-full
git checkout $1 --quiet 2>/dev/null
grep -m1 "version" python/pyproject.toml | grep -oP '\d+\.\d+\.\d+'
```

---

*Last Updated: 2026-01-20*
