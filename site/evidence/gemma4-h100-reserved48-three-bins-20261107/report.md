# Gemma-4-12B Reserved48 H100 partition comparisons

The CPU reduction covered **36 of 48** planned reserved English articles on H100. Prompt lengths 128, 512 and 2,048 contributed 12 articles each. Each article supplied full, token and chunked partition observations. The resulting **108 within-H100** input contrasts cover **5,076** sampled distribution pair positions. The full plan defines 432 input contrasts and 20,304 distribution pair positions across both backends.

The CPU admission derived 108 FP64 matrices from selected native BF16 rows and checked 100,116 archived target scalars. The summary below reports per-article target logprob MAE ranges and sampled Top-1 counts from the 108 admitted pair reports. Each Top-1 denominator is 36 articles × 47 selected positions. The ranges do not pool unequal token counts across prompt lengths.

| H100 partition pair | Per-article target logprob MAE range (nat) | Largest absolute target logprob difference (nat) | Sampled Top-1 agreement |
| --- | ---: | ---: | ---: |
| full / token | 0.000508–0.047292 | 2.710451 | 1,671/1,692 |
| full / chunk | 0.021210–0.053615 | 2.026391 | 1,663/1,692 |
| token / chunk | 0.021808–0.047208 | 2.406508 | 1,668/1,692 |

Both 8,192-token attempts ended with the recorded terminal status `cleanup_failed` and return code 1. Each contributed zero numeric observations. The interruption cause is unresolved. The aggregate is limited to the completed H100 bins. Full-cohort NLL, quality, numerical thresholds and acceptance are unset. Free-generation quality and paired performance remain outside these results.

The archived CPU reports cover selected full-vocabulary rows and source target scalar checks. Independent all-target normalization, unselected raw-row audit and an independent scalar E02 check remain open for this scope. The aggregate reopened zero native or derived arrays.

The [selected summary](summary.json) includes the exact, unrounded scalar results, original report SHA-256 values, and the two failed-attempt controller digests.
