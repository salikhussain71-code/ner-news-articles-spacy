# NER evaluation - sample_conll.txt

Sentences: **15** | Gold entities: **45** | Exact-match entity-level scores (CoNLL labels PER/ORG/LOC/MISC).

## Overall (micro-averaged)

| approach | precision | recall | f1 | macro_f1 | boundary_f1 | seconds | tokens_per_sec |
|---|---|---|---|---|---|---|---|
| hybrid-rules-first:en_core_web_md | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.025 | 7743.569 |
| hybrid-rules-first:en_core_web_sm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.019 | 10141.662 |
| rule | 1.000 | 0.956 | 0.977 | 0.950 | 0.977 | 0.004 | 53594.111 |
| hybrid:en_core_web_md | 0.956 | 0.956 | 0.956 | 0.924 | 1.000 | 0.023 | 8575.488 |
| model:en_core_web_md | 0.956 | 0.956 | 0.956 | 0.924 | 1.000 | 0.022 | 9105.956 |
| hybrid:en_core_web_sm | 0.911 | 0.911 | 0.911 | 0.894 | 1.000 | 0.020 | 9635.488 |
| model:en_core_web_sm | 0.909 | 0.889 | 0.899 | 0.887 | 0.989 | 0.026 | 7652.503 |

## Per entity type

| approach | label | precision | recall | f1 | support | tp | fp | fn |
|---|---|---|---|---|---|---|---|---|
| rule | PER | 1.000 | 0.667 | 0.800 | 6.000 | 4.000 | 0.000 | 2.000 |
| rule | ORG | 1.000 | 1.000 | 1.000 | 17.000 | 17.000 | 0.000 | 0.000 |
| rule | LOC | 1.000 | 1.000 | 1.000 | 17.000 | 17.000 | 0.000 | 0.000 |
| rule | MISC | 1.000 | 1.000 | 1.000 | 5.000 | 5.000 | 0.000 | 0.000 |
| model:en_core_web_sm | PER | 1.000 | 1.000 | 1.000 | 6.000 | 6.000 | 0.000 | 0.000 |
| model:en_core_web_sm | ORG | 0.889 | 0.941 | 0.914 | 17.000 | 16.000 | 2.000 | 1.000 |
| model:en_core_web_sm | LOC | 0.882 | 0.882 | 0.882 | 17.000 | 15.000 | 2.000 | 2.000 |
| model:en_core_web_sm | MISC | 1.000 | 0.600 | 0.750 | 5.000 | 3.000 | 0.000 | 2.000 |
| model:en_core_web_md | PER | 1.000 | 1.000 | 1.000 | 6.000 | 6.000 | 0.000 | 0.000 |
| model:en_core_web_md | ORG | 1.000 | 1.000 | 1.000 | 17.000 | 17.000 | 0.000 | 0.000 |
| model:en_core_web_md | LOC | 0.895 | 1.000 | 0.944 | 17.000 | 17.000 | 2.000 | 0.000 |
| model:en_core_web_md | MISC | 1.000 | 0.600 | 0.750 | 5.000 | 3.000 | 0.000 | 2.000 |
| hybrid:en_core_web_sm | PER | 1.000 | 1.000 | 1.000 | 6.000 | 6.000 | 0.000 | 0.000 |
| hybrid:en_core_web_sm | ORG | 0.895 | 1.000 | 0.944 | 17.000 | 17.000 | 2.000 | 0.000 |
| hybrid:en_core_web_sm | LOC | 0.882 | 0.882 | 0.882 | 17.000 | 15.000 | 2.000 | 2.000 |
| hybrid:en_core_web_sm | MISC | 1.000 | 0.600 | 0.750 | 5.000 | 3.000 | 0.000 | 2.000 |
| hybrid:en_core_web_md | PER | 1.000 | 1.000 | 1.000 | 6.000 | 6.000 | 0.000 | 0.000 |
| hybrid:en_core_web_md | ORG | 1.000 | 1.000 | 1.000 | 17.000 | 17.000 | 0.000 | 0.000 |
| hybrid:en_core_web_md | LOC | 0.895 | 1.000 | 0.944 | 17.000 | 17.000 | 2.000 | 0.000 |
| hybrid:en_core_web_md | MISC | 1.000 | 0.600 | 0.750 | 5.000 | 3.000 | 0.000 | 2.000 |
| hybrid-rules-first:en_core_web_sm | PER | 1.000 | 1.000 | 1.000 | 6.000 | 6.000 | 0.000 | 0.000 |
| hybrid-rules-first:en_core_web_sm | ORG | 1.000 | 1.000 | 1.000 | 17.000 | 17.000 | 0.000 | 0.000 |
| hybrid-rules-first:en_core_web_sm | LOC | 1.000 | 1.000 | 1.000 | 17.000 | 17.000 | 0.000 | 0.000 |
| hybrid-rules-first:en_core_web_sm | MISC | 1.000 | 1.000 | 1.000 | 5.000 | 5.000 | 0.000 | 0.000 |
| hybrid-rules-first:en_core_web_md | PER | 1.000 | 1.000 | 1.000 | 6.000 | 6.000 | 0.000 | 0.000 |
| hybrid-rules-first:en_core_web_md | ORG | 1.000 | 1.000 | 1.000 | 17.000 | 17.000 | 0.000 | 0.000 |
| hybrid-rules-first:en_core_web_md | LOC | 1.000 | 1.000 | 1.000 | 17.000 | 17.000 | 0.000 | 0.000 |
| hybrid-rules-first:en_core_web_md | MISC | 1.000 | 1.000 | 1.000 | 5.000 | 5.000 | 0.000 | 0.000 |

## Error breakdown (counts)

| approach | boundary_error | correct | missed | spurious | type_error |
|---|---|---|---|---|---|
| hybrid-rules-first:en_core_web_md | 0 | 45 | 0 | 0 | 0 |
| hybrid-rules-first:en_core_web_sm | 0 | 45 | 0 | 0 | 0 |
| hybrid:en_core_web_md | 0 | 43 | 0 | 0 | 2 |
| hybrid:en_core_web_sm | 0 | 41 | 0 | 0 | 4 |
| model:en_core_web_md | 0 | 43 | 0 | 0 | 2 |
| model:en_core_web_sm | 0 | 40 | 1 | 0 | 4 |
| rule | 0 | 43 | 2 | 0 | 0 |
