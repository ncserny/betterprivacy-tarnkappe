# Results (v0.2)

Produced by `python bench.py --spacy` on 2 CPU cores, shipped int8 model + regex layer, one thread.
spaCy models were trained on other corpora with slightly different labelling rules, so their scores here are a reference point, not a like-for-like contest. Our model saw the training parts of EWT and GermEval; `pud_de` and Presidio are out-of-domain for it.

```
== real annotated text: strict F1 on person / organisation / location, and on person alone ==
ewt_test (English web text)
   ours      all F1 0.699 (P 0.748 R 0.656)   person F1 0.822 (P 0.822 R 0.822)   1.30 ms/sentence
   spaCy sm  all F1 0.568 (P 0.558 R 0.578)   person F1 0.633 (P 0.650 R 0.617)   4.18 ms/sentence
   spaCy md  all F1 0.641 (P 0.619 R 0.664)   person F1 0.755 (P 0.751 R 0.759)   4.88 ms/sentence
wnut17_test (social media)
   ours      all F1 0.418 (P 0.587 R 0.325)   person F1 0.491 (P 0.656 R 0.392)   1.66 ms/sentence
   spaCy sm  all F1 0.248 (P 0.230 R 0.269)   person F1 0.274 (P 0.283 R 0.266)   4.30 ms/sentence
   spaCy md  all F1 0.276 (P 0.243 R 0.319)   person F1 0.356 (P 0.414 R 0.312)   4.43 ms/sentence
germeval_test (German Wikipedia/news)
   ours      all F1 0.757 (P 0.800 R 0.718)   person F1 0.822 (P 0.854 R 0.793)   1.82 ms/sentence
   spaCy sm  all F1 0.606 (P 0.545 R 0.682)   person F1 0.697 (P 0.672 R 0.723)   4.75 ms/sentence
   spaCy md  all F1 0.656 (P 0.601 R 0.722)   person F1 0.768 (P 0.753 R 0.783)   5.45 ms/sentence
pud_de (German news/Wikipedia)
   ours      all F1 0.727 (P 0.766 R 0.692)   person F1 0.770 (P 0.797 R 0.744)   1.93 ms/sentence
   spaCy sm  all F1 0.642 (P 0.599 R 0.691)   person F1 0.713 (P 0.710 R 0.715)   4.70 ms/sentence
   spaCy md  all F1 0.714 (P 0.675 R 0.757)   person F1 0.796 (P 0.795 R 0.797)   5.50 ms/sentence

== PII caught at all (any overlap, any label) ==
   presidio (templates unseen in training)       recall 0.833  precision 0.975
   our synthetic, new names + new phrasing       recall 0.942  precision 0.995
   ewt_test persons only                         recall 0.900
   germeval_test persons only                    recall 0.874
      presidio CREDIT_CARD     recall 0.947
      presidio EMAIL_ADDRESS   recall 1.000
      presidio GPE             recall 0.882
      presidio IBAN_CODE       recall 1.000
      presidio IP_ADDRESS      recall 1.000
      presidio ORGANIZATION    recall 0.689
      presidio PERSON          recall 0.940
      presidio PHONE_NUMBER    recall 0.583
      presidio STREET_ADDRESS  recall 0.728
      presidio ZIP_CODE        recall 0.333

== first / last name split (synthetic, new names + new phrasing) ==
   F1 0.858 (P 0.888 R 0.831)

== latency, 1 CPU thread, 2000 texts, avg 16 tokens ==
p50 1.41 ms   p95 2.81 ms   p99 4.02 ms
throughput 650 texts/s, 10394 tokens/s per core
long document (964 tokens, 4759 chars): 78.2 ms
```
