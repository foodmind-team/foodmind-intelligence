# Recommendation Agent evidence catalog

All Backend evidence is a point-in-time fact with inclusive cutoff
`observed_at <= decisionAt`. Missing nullable facts remain null; they are never
converted to positive evidence. Numbers are finite JSON numbers.

| JSON pointer | Producer | Nullable | Unit/range | Agent use | Log/reason permission |
| --- | --- | --- | --- | --- | --- |
| `/candidates/*/evidence/preferenceMatch` | Backend | no | ratio 0..1 | personal selection, `PREFERENCE_MATCH` | value not logged; reason code allowed |
| `/candidates/*/evidence/wantToTry` | Backend | no | boolean | `WANT_TO_TRY` | value not logged; reason code allowed |
| `/candidates/*/evidence/groupPreferenceRate` | Backend | yes | ratio 0..1 | group eligibility | value not logged; reason code allowed with count threshold |
| `/candidates/*/evidence/groupEligibleMemberCount` | Backend | no | count 0..100 | group eligibility only | value not logged; never UserCF evidence |
| `/candidates/*/evidence/contextMatch` | Backend | yes | ratio 0..1 | `CONTEXT_MATCH` | value not logged; reason code allowed |
| `/candidates/*/evidence/cleanlinessObserved` | Backend | no | boolean | `CLEANLINESS_OBSERVED` | value not logged; reason code allowed; no safety claim |
| `/candidates/*/evidence/novelty` | Backend | no | ratio 0..1 | exploratory bonus | value not logged; no direct reason |
| `/candidates/*/evidence/cuisineCode` | Backend | no | controlled code | diversity penalty | code not logged or returned |
| `/candidates/*/evidence/categoryCode` | Backend | no | controlled code | diversity penalty | code not logged or returned |
| `/predictions/*/userCf` | Inference | no | availability, score 0..1/null, support count | personal evidence, `USER_CF` | only reason code allowed |
| `/predictions/*/itemCf` | Inference | no | availability, score 0..1/null, support count | `ITEM_CF` | only reason code allowed |

Inference echoes the six named Backend signals solely so the Agent can reject
evidence drift. It may not add prose, identifiers, coefficients, or feature
vectors. UserCF and ItemCF support are inference-generated and distinct from
Backend group/personal counts.
