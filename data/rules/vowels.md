# Bengali Vowel System in Obadh

## 1. Basic Vowels (স্বরবর্ণ)

| Roman Input | Independent Vowel | Vowel Symbol (Kar) | Name |
|-------------|-------------------|-------------------|------|
| o | অ | - (inherent) | অ-কার (a-kar) |
| a / A | আ | া | আ-কার (aa-kar) |
| aY / AY | অ্যা | ্যা | অ্যা-কার (ae-kar with য-ফলা shape) |
| i | ই | ি | ই-কার (i-kar) |
| I | ঈ | ী | ঈ-কার (dirgho i-kar) |
| u | উ | ু | উ-কার (u-kar) |
| U | ঊ | ূ | ঊ-কার (dirgho u-kar) |
| e / E | এ | ে | এ-কার (e-kar) |
| OI | ঐ | ৈ | ঐ-কার (oi-kar) |
| O | ও | ো | ও-কার (o-kar) |
| OU | ঔ | ৌ | ঔ-কার (ou-kar) |
| rri | ঋ | ৃ | ঋ-কার (ri-kar) |

`rri` is an atomic vowel signal. It is matched before the shorter `rr` reph signal when both could start at the same position.

## 2. Basic Rules for Vowel Usage

### 2.1 Independent Vowels vs. Vowel Symbols

- **Independent vowels** are used at the beginning of a word or when a vowel appears independently
- **Vowel symbols (kars)** are used when the vowel follows a consonant

### 2.2 Rule Signals

| Position | Roman Signal | Bengali Output | Explanation |
|----------|--------------|----------------|-------------|
| Vowel initial | `a` / `A` | আ | আ as an independent vowel |
| Vowel initial | `aY` / `AY` | অ্যা | explicit অ্যা as an independent vowel |
| Vowel initial | `I` | ঈ | long ঈ as an independent vowel |
| Vowel initial | `u` | উ | short উ as an independent vowel |
| Vowel initial | `U` | ঊ | long ঊ as an independent vowel |
| Vowel initial | `e` / `E` | এ | এ as an independent vowel |
| After consonant | `k` + `i` | কি | ি after ক |
| After consonant | `k` + `aY` / `AY` | ক্যা | ্যা after ক |
| After consonant | `k` + `I` | কী | ী after ক |
| After consonant | `t` + `u` | তু | ু after ত |
| After consonant | `t` + `U` | তূ | ূ after ত |
| After consonant/conjunct | `tiyw`, `ktiYwta` | তীয়, ক্তীয়তা | typed long-ঈয় signal |

## 3. Vowel 'o' as Conjunct Breaker

One of the most important special rules is using the vowel `o` to prevent conjunct formation:

| Typing Pattern | Bengali Result | Explanation |
|----------------|----------------|-------------|
| `kk` | ক্ক | Forms conjunct: ক + ্ + ক |
| `kok` | কক | Prevents conjunct by inserting inherent অ between consonants |
| `kOk` | কোক | Inserts the visible ও / ো vowel |
| `bhokt` | ভক্ত | Uses lowercase `o` as the deliberate inherent-vowel signal before `kt` |

This is crucial when you need to represent two consecutive same letters without forming a conjunct. The vowel 'o' acts as a separator while being minimally pronounced in natural speech.

## 4. Special Vowel Rules

### 4.1 Vowel + Vowel Combinations

| Combination | Roman Input | Bengali Output |
|-------------|-------------|----------------|
| a + i | ai | আই |
| a + u | au | আউ |
| a + e | ae | আএ |
| a + o | ao | আও |
| i + a | ia | ইয়া |
| i + o | io | ইও |
| e + o | eo | এও |

> Doubled lowercase vowels are not long-vowel aliases. They compose as typed: `aa` → `আআ`, `ee` → `এএ`, `ii` → `ইই`, `oo` → `অঅ`, and `uu` → `উউ`. After consonants they still compose from individual signals, e.g. `kaa` → `কাআ`, `kee` → `কেএ`, `kii` → `কিই`, `koo` → `কঅ`, and `kuu` → `কুউ`.
> Lowercase `a` and uppercase `A` are both visible আ/া signals. They are not suppressed before a following conjunct cluster: `kaby` and `kAby` both render `কাব্য`. Use lowercase `o` when the intended sound is the inherent vowel before a cluster, e.g. `bhokt` → `ভক্ত`.
> `aY`/`AY` are atomic অ্যা vowel signals. They let deliberate forms such as `aYp`/`AYp` → `অ্যাপ` and `kaY` → `ক্যা` compose without a word rule. Lowercase `ay` remains আ + য় (`ayp` → `আয়প`).
> Lowercase `oi`/`ou` remain vowel sequences such as `boi` → `বই`; use uppercase `OI`/`OU` for ঐ/ঔ.

The same vowel-sequence rules compose after consonants by using the dependent form of the first vowel plus any following independent vowel or glide.

`iyw` after a consonant, conjunct, or reph unit that already carries short `i` is a deliberate long-ঈয় signal. It rewrites that attached `i` to `I` and consumes the marker `w`, so `tiyw` → `তীয়` and `jatiywta` → `জাতীয়তা`. It does not apply after the atomic `rri` vowel signal. A following lowercase `o` remains the inherent-vowel terminator (`kiywo` → `কীয়`); use uppercase `O` for visible ও-কার (`kiywO` → `কীয়ো`).

### 4.2 Edge Cases and Exceptions

1. **Inherent 'a' Sound Elimination:**
   - To eliminate the inherent 'a' sound at the end of a word, use hasant (্)
   - Hasant is written as `,,`
   
2. **Silent/Half 'a' Sound:**
   - In some cases, the 'a' sound is pronounced halfway
   - No separate notation in this deterministic layer; use the documented Roman rule signal for the intended spelling
   

### 4.3 Vowel Modifications

| Modification | Roman Input | Bengali Output |
|--------------|-------------|----------------|
| Nasalization | vowel + `^` | vowel + ঁ |
| Visarga | `:` | ঃ |

## 4. Consonant + Vowel Combinations

The following examples show how vowels combine with consonants:

| Combination | Roman Input | Bengali Output | 
|-------------|-------------|----------------|
| ক + আ | ka | কা |
| ক + অ্যা | kaY | ক্যা |
| ক + ি | ki | কি |
| ক + ী | kI | কী |
| ক + ু | ku | কু |
| ক + ূ | kU | কূ |
| ক + ে | ke | কে |
| ক + ৈ | kOI | কৈ |
| ক + ো | kO | কো |
| ক + ৌ | kOU | কৌ |
| ক + ৃ | krri | কৃ |
