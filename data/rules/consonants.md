# Bengali Consonant System in Obadh

This document is the source contract for runtime consonant rule keys. Consonant aliases are accepted only when they preserve a deterministic, deliberate typing signal.

## Basic Consonants

| Roman Input | Bengali Output | Category | Note |
|-------------|----------------|----------|------|
| k | ক | Velar | ka |
| kh / Kh / KH | খ | Velar | aspirated kha; titlecase and all-caps aliases compose through the same rule |
| g | গ | Velar | ga |
| gh / Gh / GH | ঘ | Velar | aspirated gha; titlecase and all-caps aliases compose through the same rule |
| Ng | ঙ | Velar | velar nasal nga |
| c | চ | Palatal | ca |
| ch / chh / C / Ch / CH / Chh / CHH | ছ | Palatal | aspirated cha family; accepted as one documented alias group |
| J / j | জ | Palatal | ja |
| jh / Jh / JH | ঝ | Palatal | aspirated jha; titlecase and all-caps aliases compose through the same rule |
| NG | ঞ | Palatal | palatal nasal nya |
| T | ট | Retroflex | retroflex ta |
| Th / TH | ঠ | Retroflex | aspirated retroflex tha |
| D | ড | Retroflex | retroflex da |
| Dh / DH | ঢ | Retroflex | aspirated retroflex dha |
| N | ণ | Retroflex | retroflex na |
| t | ত | Dental | dental ta |
| th | থ | Dental | aspirated dental tha |
| d | দ | Dental | dental da |
| dh | ধ | Dental | aspirated dental dha |
| n | ন | Dental | dental na |
| p | প | Labial | pa |
| ph / Ph / PH / f | ফ | Labial | pha; `f` is accepted for common typed input |
| b | ব | Labial | ba |
| bh / Bh / BH / v | ভ | Labial | bha; `v` is accepted for common typed input |
| m | ম | Labial | ma |
| z | য | Semivowel | regular য base; does not become য-ফলা by itself |
| r | র | Semivowel | ra |
| l | ল | Semivowel | la |
| sh / S | শ | Fricative | palatal sha |
| Sh / SH | ষ | Fricative | retroflex sha |
| s | স | Fricative | dental sa |
| h | হ | Fricative | ha |
| R | ড় | Special | Bengali ra with nukta-like dot |
| Rh | ঢ় | Special | aspirated dotted ra |
| y / Y | য় | Special | regular antastha y; inside declared conjunct clusters this signal can serve as the য-ফলা marker |

## Conjunct Interaction

- Regular `z` is the base consonant য.
- Regular `b` is the base consonant ব.
- `y` / `Y` and `w` are phola markers only inside declared valid conjunct clusters.
- Titlecase and all-caps aspirated aliases are not independent correction behavior; they canonicalize into the same consonant components before conjunct formation.
- Missing one-letter alphabetic case variants are accepted as fallback to the exact opposite-case signal only when the typed case is unclaimed and not reserved. Today this admits `B`, `G`, `K`, `P`, `F`, `V`, `L`, and `H` as fallback to their lowercase rules for autocapitalized deliberate input such as `Biggan` → `বিজ্ঞান`; it does not override exact signals such as `T`, `D`, `N`, `S`, `I`, `U`, `O`, `Y`, `M`, or the narrow `Z` marker.
