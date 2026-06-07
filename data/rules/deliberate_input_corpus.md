# Deliberate Input Rule Probe Corpus

This source-controlled corpus exercises rule signals, not memorized words. It is audit and regression material for the deterministic engine; it must not be loaded by the library, CLI, or WASM runtime.

| Category | Roman Input | Bengali Output | Contract |
|----------|-------------|----------------|----------|
| Vowel policy | `kOI kOU boi bou` | `কৈ কৌ বই বউ` | Uppercase `OI`/`OU` are diphthongs; lowercase `oi`/`ou` remain sequences. |
| Conjunct blocker | `kk kok kOk` | `ক্ক কক কোক` | Lowercase `o` blocks conjuncts without visible ও; uppercase `O` is visible. |
| Explicit hasant | `k,,k k,,a kk,,` | `ক্ক ক্আ ক্ক্` | `,,` is an explicit virama/conjunct command, not a spelling guess. |
| Khanda-ta reph | <code>rrt rrt`` rr,,t`` rrt``sa</code> | `র্ত র্ৎ র্ৎ র্ৎসা` | `rrt` remains র্ত; খণ্ড ত uses the explicit <code>t``</code> signal. |
| Reph cluster | `rrkSh rrk,,Sh rrsk rrs,,ka` | `র্ক্ষ র্ক্ষ র্স্ক র্স্কা` | Reph composes over declared valid clusters, implicit or explicit. |
| Phola markers | `ky k,,w zya bwa Rw` | `ক্য ক্ব য্যা ব্বা ড়w` | `y`/`w` are phola markers only in declared clusters. |
| Nasal signals | `ngg sMgo songskrriti shongkha` | `ঙ্গ সংগ সংস্কৃতি শংখা` | `ngg` is velar conjunct shorthand; `M` preserves literal anusvara. |
| Long iya | `tiyw kiywo kiywO` | `তীয় কীয় কীয়ো` | `iyw` is the composable long-ঈয় signal. |
| Non-conjunct ra-ya | `rZyab rrYa Zya kZya` | `র‌্যাব র্যা Zয়া কZয়া` | `rZy` is a narrow ZWNJ signal; unrelated `Z` is not an alias. |
| External alias rejection | `q G gg pph p,,ph` | `q G গগ পফ প্ফ` | Unknown broad aliases remain literal; explicit hasant stays available. |
| Symbols and numbers | `12.34 12.34. $` | `১২.৩৪ ১২.৩৪। ৳` | Decimal periods stay ASCII between number-bearing tokens. |
