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
| Base vs phola separation | `kz kb zoy zy bw` | `কয কব যয় য্য ব্ব` | `z`/`b` remain base letters; `y`/`w` become phola markers only through declared clusters. |
| Nasal signals | `ngg sMgo songskrriti shongkha` | `ঙ্গ সংগ সংস্কৃতি শংখা` | `ngg` is velar conjunct shorthand; `M` preserves literal anusvara. |
| Nasal escape | `songgo sMgo nggho sMgho` | `সঙ্গ সংগ ঙ্ঘ সংঘ` | `M` forces literal anusvara before `g`/`gh`; `ngg`/`nggh` remain deliberate velar conjunct shorthand. |
| Palatal nasal-ja | `NGj nj nJ jinjira noj nz` | `ঞ্জ ঞ্জ ঞ্জ জিঞ্জিরা নজ নয` | `NGj` is the source ঞ+জ signal; `nj`/`nJ` are narrow deterministic aliases, while `o` separates literal নজ and `nz` remains নয. |
| Jna pronunciation | `jNG jn gg ggan biggan gog` | `জ্ঞ জ্ঞ জ্ঞ জ্ঞান বিজ্ঞান গগ` | `gg` is a pronounced shorthand for orthographic জ্ঞ; use `gog` for literal গগ. |
| Long iya | `tiyw kiywo kiywO` | `তীয় কীয় কীয়ো` | `iyw` is the composable long-ঈয় signal. |
| Vocalic r | `rri krri rria rrhi` | `ঋ কৃ ঋআ র্হি` | `rri` is the vocalic ঋ signal; `rr` remains reph before consonants. |
| App vowel signal | `aYp AYp ayp Ayp app kaY kay` | `অ্যাপ অ্যাপ আয়প আয়প আপ্প ক্যা কায়` | `aY`/`AY` are অ্যা vowel signals; lowercase `y` remains the ordinary য় path. |
| Repeated vowel freedom | `a A aa i I ii e E ee o O oo u U uu kaa kee kii koo kuu kU uuupintocala` | `আ আ আআ ই ঈ ইই এ এ এএ অ ও অঅ উ ঊ উউ কাআ কেএ কিই কঅ কুউ কূ উউউপিন্তচালা` | Doubled lowercase vowels are not aliases; repeated signals remain repeated for deliberate invented strings. |
| Vowel sequences | `kai kau kia kio keo` | `কাই কাউ কিয়া কিও কেও` | Documented vowel sequences compose as rule units, not guessed spellings. |
| Marked vowel boundary | `k,,i k^a k:a` | `ক্ই কঁআ কঃআ` | Vowels after explicit hasant, chandrabindu, or visarga start independently. |
| Non-conjunct ra-ya | `rZyab rrYa Zya kZya` | `র‌্যাব র্যা Zয়া কZয়া` | `rZy` is a narrow ZWNJ signal; unrelated `Z` is not an alias. |
| Aspirated alias composition | `KhA KHy Chya jhya fya acCHHa` | `খা খ্য ছ্যা ঝ্যা ফ্যা আচ্ছা` | Accepted aspirated aliases canonicalize into ordinary rule components before vowel/conjunct handling. |
| Case fallback | `Biggan Ggan BhalO Khela Ga T D N Zya` | `বিজ্ঞান জ্ঞান ভালো খেলা গা ট ড ণ Zয়া` | Unclaimed opposite-case rule signals fall back to the exact canonical signal; exact uppercase signals and narrow `Z` remain protected. |
| External alias rejection | `q Q pph p,,ph` | `q Q পফ প্ফ` | Unknown broad aliases remain literal; explicit hasant stays available. |
| Symbols and numbers | `12.34 12.34. $` | `১২.৩৪ ১২.৩৪। ৳` | Decimal periods stay ASCII between number-bearing tokens. |
