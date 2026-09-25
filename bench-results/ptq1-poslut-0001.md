# PTQ1 positioned-LUT experiment (ptq1-poslut-0001, NEGATIVE)

Date: 2026-09-25. Idea: pre-shifted per-position tables (4x256x5 uint32,
20 KB constant) remove shift/AND from int assembly (47 -> 27 ops/pack).
Offline proof: 201024 packs, 0 mismatches vs LUT5 assembly.

Result: REGRESSION. pp512 804.71 -> 773.26 (-3.9%), pp1024 ~769 -> ~741,
pp5201 ~745 -> ~719, all configs consistent. REVERTED to LUT5+uniform
(same-day rebench pp512 805.06 confirms restore).

Suspected cause: 20 KB working set thrashes the scalar constant cache
(LUT5 2 KB fit comfortably); extra 3D-index address math adds chains.
Lesson: constant-memory working set size dominates ALU savings here.

Table code was machine-generated to mmq-ptq1-poslut.inc (since deleted);
generator: /tmp/opencode/check_ptq1_poslut.py (scratch, not in repo).
Final shipped state = ptq1-uniform-0001 (LUT5 + uniform loader).
