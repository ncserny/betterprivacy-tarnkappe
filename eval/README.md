# Evaluation: test sets and the release bar

## Release bar

`release_bar.json` is the agreed bar for calling a version production-ready. "Found" means an annotated
item is overlapped by at least one detection, whatever its label.

| Item | Must be found |
| --- | --- |
| Person | 95% |
| Email, IBAN | 99% |
| Phone | 95% |
| Street, postcode | 90% |
| Location | 85% |
| Organisation | 80% |

In addition at least 85% of detections must be real PII, and the median latency for a short text must stay under 5 ms.
The bar counts only when measured on a hand-checked set of real pages (below). If the numbers ever change,
change them before looking at results, not after.

## Challenge set v1

`challenge_v1.txt` holds 87 hand-written, fictional snippets (49 English, 38 German, 370 annotated items) in the style
of people-search sites, data brokers, search results, social networks, court and public records, news, professional
directories, forums and blogs. All names, addresses and numbers are invented.

Use it as a regression test, not as proof of quality:

- It was written by the same author as the detector, which makes it easier than real pages.
- The phone and postcode rules were developed after looking at misses on it.

## Building the real test set

1. Save 300 to 500 real pages or snippets as plain-text files, one per file, spread across source types and both languages.
2. `python scripts/prelabel.py pages/ eval/real_v1.txt --category people-search --lang de`
3. Open `eval/real_v1.txt` in a text editor and correct the brackets by hand: add what the model missed, remove what it wrongly flagged.
   `[[P|..]]` person, `[[O|..]]` organisation, `[[L|..]]` location, `[[S|..]]` street, `[[Z|..]]` postcode, `[[E|..]]` email, `[[T|..]]` phone, `[[I|..]]` IBAN.
4. `python bench.py --testset eval/real_v1.txt --gate`

Pre-labelling biases the checker towards the model's view, so read every line and look for what is missing.
Files named `eval/real_*.txt` are ignored by git because they contain real personal data. Keep them in the EU and off public systems.

## What counts as an organisation

From v0.6 on, training data and hand labels follow one rule: **an organisation is anything a person can work for or
be a member of**, because that is what a mention reveals about a person ("works at X", "studied at Y").

| Counts | Does not count |
| --- | --- |
| Companies, startups, agencies, shops, practices | Products, apps, software and platforms (a CMS, a messenger app, a blog platform) |
| Public bodies and authorities (a central bank, a ministry, the armed forces) | Events and conferences |
| Media outlets (broadcasters, newspapers, magazines) | Buildings and places named after an organisation |
| Universities, schools, research institutes | |
| Clubs, associations, parties, foundations | |

A product name counts when it is also the name of the company behind it ("Google", "Facebook"). When a product name
contains the company name ("Acme Editor"), mark the company part only.
