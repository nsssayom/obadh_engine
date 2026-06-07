# Simplified Transliteration Rules

This document outlines the core rules and special cases for Bengali transliteration in the Obadh Engine.

## Core Approach

This implementation takes a structured approach to Bengali transliteration, focusing on:

### 1. Vowel Handling

Vowels are handled in different modes based on their position and context:

- **Independent Vowels**: When vowels appear on their own
  - Signal: `o` → `অ`, `O` → `ও`, `ee` → `ঈ`, `oo` → `উ`, `uu` → `ঊ`

- **Vowel Modifiers**: When vowels follow consonants
  - Signal: `ka` → `কা`, `ki` → `কি`, `kee` → `কী`, `koo` → `কু`, `kuu` → `কূ`
  - Signal: `tiyw` → `তীয়`, `jatiywta` → `জাতীয়তা`

- **Inherent Vowel**: The implied 'অ' sound after consonants
  - Signal: `k` → `ক` (inherently includes the 'অ' sound)

### 2. Consonant Handling

Consonants are handled in different modes:

- **Independent**: Individual consonants with inherent vowel
  - Signal: `k` → `ক`, `kh` → `খ`
  - Aspirated aliases accept documented titlecase/all-caps forms: `Kh`/`KH` → `খ`, `Chh`/`CHH` → `ছ`

- **With Vowel Modifiers**: Consonants followed by explicit vowels
  - Signal: `ka` → `কা`, `ki` → `কি`, `KHi` → `খি`, `CHHi` → `ছি`

- **Conjuncts**: Multiple consonants combined with hasant
  - Signal: `kk` → `ক্ক`, `kt` → `ক্ত`
  - Created by adding hasant (্) between consonants
  - Aspirated aliases compose through canonical conjunct rules: `KHy` → `খ্য`, `acCHHa` → `আচ্ছা`
  - জ্ঞ can be typed by component signal `jNG` or shorthand `jn`: `jnan` → `জ্ঞান`

### 3. Special Reph Form

- Double 'r' creates the special reph form
  - Signal: `rrm` → `র্ম`
  - This is different from a normal sequence: `rm` → `রম`
  - Reph form is created by adding hasant (্) between র and the following consonant (e.g. `rrm` → `র্ম`, `rrk` → `র্ক`)
  - A following explicit hasant is redundant before a reph target: `rr,,ka` → `র্কা`
  - Reph can compose over a valid following conjunct cluster without a separate word rule: `rrkSh` → `র্ক্ষ`, `rrsk` → `র্স্ক`
  - The same cluster can be typed with an explicit hasant boundary: `rrk,,Sh` → `র্ক্ষ`
  - <code>rrt</code> is reserved for `র্ত`; use composable <code>rrt``</code> for `র্ৎ`

### 4. Numeric Characters

Bengali has its own numerals that are mapped directly from Latin numerals:

| Latin | Bengali |
|-------|---------|
| 0     | ০       |
| 1     | ১       |
| 2     | ২       |
| 3     | ৩       |
| 4     | ৪       |
| 5     | ৫       |
| 6     | ৬       |
| 7     | ৭       |
| 8     | ৮       |
| 9     | ৯       |

- Numerals are transliterated directly to their Bengali equivalents
- They do not participate in conjunct formation or vowel modification
- Signal: `123` → `১২৩`, `k2` → `ক২`

## Special Rules

### Hasanta Handling

- Represented in Obadh input as `,,`
  - Signal: `k,,` → `ক্‌`
- Between consonants, `,,` is an explicit conjunct command:
  - Signal: `k,,k` → `ক্ক`
- A trailing explicit hasant after a formed conjunct remains visible:
  - Signal: `k,,k,,` → `ক্ক্`
  - Signal: `n,,d,,r,,` → `ন্দ্র্`
- As a standalone marker, `,,` renders the virama directly:
  - Signal: `,,` → `্`
- A vowel typed after an explicit dead consonant is independent, not a dependent kar on that dead consonant:
  - Signal: `k,,a` → `ক্আ`
  - Signal: `k,,i` → `ক্ই`
- Khanda-ta composes with reph without a separate word rule:
  - Signal: <code>t``</code> → `ৎ`
  - Signal: <code>rrt``</code> → `র্ৎ`
  - Uppercase <code>T``</code> remains an accepted alias.
  - A following explicit hasant is redundant before another consonant because <code>t``</code> is already the dead খণ্ড ত form: <code>t``,,sa</code> → `ৎসা`

### Diacritic Marks

- Chandrabindu is represented by `^`
  - Signal: `kA^` → `কাঁ`
- Visarga is represented by `:`
  - Signal: `ku:` → `কুঃ`
- Trailing diacritic marks are explicit ordered marks. The engine preserves the order typed by the user:
  - Signal: `kA^:` → `কাঁঃ`
  - Signal: `kA:^` → `কাঃঁ`
- A vowel typed after an already rendered mark starts independently:
  - Signal: `k^a` → `কঁআ`
  - Signal: `k:a` → `কঃআ`

### Nasal Signals

Nasal input is deterministic and must preserve the user's intended spelling:

- `ng` is anusvara:
  - Signal: `bangla` → `বাংলা`
  - Signal: `songket` → `সংকেত`
- `M` is the explicit anusvara escape:
  - Signal: `sMgo` → `সংগ`
  - Use it before `g`/`gh` when you want literal ংগ/ংঘ rather than the `ngg`/`nggh` shorthand.
- `Ng` is the velar nasal consonant:
  - Signal: `oNgko` → `অঙ্ক`
  - Signal: `oNggo` → `অঙ্গ`
- `ngg` and `nggh` are shorthand for deliberate velar nasal conjuncts:
  - Signal: `bonggo` → `বঙ্গ`
  - Signal: `ngghAt` → `ঙ্ঘাত`

### 'o' as a Blocker

The 'o' character serves special functions:
- Acts as conjunct blocker
  - Signal: `kok` → `কক` (prevents 'k' from forming a conjunct with the following 'k')
- Acts as vowel-modifier blocker

### য-ফলা and ব-ফলা Handling

There's special handling for য-ফলা and ব-ফলা:

- **Regular consonants** use 'z' and 'b': 
  - `z` → `য` (ya)
  - `b` → `ব` (ba)

- **Phola forms** (conjunct versions) use 'y' and 'w':
  - `ky` → `ক্য` (k + hasant + ya)
  - `kw` → `ক্ব` (k + hasant + ba)
  - `zy` / `zY` → `য্য` (regular `z` base + ya-phola marker)
  - `bw` → `ব্ব` (regular `b` base + ba-phola marker)
  - Valid phola clusters may also be typed with an explicit boundary: `k,,y` → `ক্য`, `k,,w` → `ক্ব`, `m,,w,,r` → `ম্ব্র`
  - `mw` → `ম্ব`, `mwr` → `ম্ব্র`
  - `y`/`Y` and `w` are accepted as phola markers only inside declared valid conjunct clusters; invalid explicit clusters remain decomposed
  - After a short-i-bearing consonant or conjunct, `iyw` is a deliberate long-ঈয় signal rather than a ব-ফলা command: `tiyw` → `তীয়`, `ktiYwta` → `ক্তীয়তা`
  - Lowercase `o` after that signal stays an inherent-vowel terminator (`kiywo` → `কীয়`); uppercase `O` gives visible ও-কার (`kiywO` → `কীয়ো`)

- **Regular `z` and `b` do not become phola markers by themselves**:
  - `kz` → `কয` (k + ya, no conjunct)
  - `kb` → `কব` (k + ba, no conjunct)
  - `zoy` → `যয়` (the `o` terminator blocks `zy`)

### Non-Conjunct র‌্য Signal

The source conjunct notes distinguish true conjunct `র্য` from the ZWNJ-separated `র‌্য` form used in loanword spellings such as `র‌্যাব`:

- `rrYa` → `র্যা` (conjunct `র্য` plus vowel)
- `rZya` / `rZYa` → `র‌্যা` (`র` + U+200C + virama + `য` plus vowel)
- `rZyab` → `র‌্যাব`
- `rZya^da` → `র‌্যাঁদা`

`Z` is intentionally narrow here. It is not a general Avro-style ya-phola alias and does not rewrite unrelated sequences such as `Zya` or `kZya`.
