# Bengali Vowel System in Obadh

## 1. Basic Vowels (স্বরবর্ণ)

| Roman Input | Independent Vowel | Vowel Symbol (Kar) | Name |
|-------------|-------------------|-------------------|------|
| o | অ | - (inherent) | অ-কার (a-kar) |
| A / aa | আ | া | আ-কার (aa-kar) |
| i | ই | ি | ই-কার (i-kar) |
| I / ee / ii | ঈ | ী | ঈ-কার (dirgho i-kar) |
| u / oo | উ | ু | উ-কার (u-kar) |
| U / uu | ঊ | ূ | ঊ-কার (dirgho u-kar) |
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
| Vowel initial | `A` / `aa` | আ | long আ as an independent vowel |
| Vowel initial | `I` / `ee` / `ii` | ঈ | long ঈ as an independent vowel |
| Vowel initial | `u` / `oo` | উ | short উ as an independent vowel |
| Vowel initial | `U` / `uu` | ঊ | long ঊ as an independent vowel |
| Vowel initial | `e` / `E` | এ | এ as an independent vowel |
| After consonant | `k` + `i` | কি | ি after ক |
| After consonant | `k` + `ee` / `ii` | কী | ী after ক |
| After consonant | `t` + `u` | তু | ু after ত |
| After consonant | `t` + `oo` | তু | ু after ত |
| After consonant | `t` + `uu` | তূ | ূ after ত |
| After consonant/conjunct | `tiyw`, `ktiYwta` | তীয়, ক্তীয়তা | typed long-ঈয় signal |

## 3. Vowel 'o' as Conjunct Breaker

One of the most important special rules is using the vowel `o` to prevent conjunct formation:

| Typing Pattern | Bengali Result | Explanation |
|----------------|----------------|-------------|
| `kk` | ক্ক | Forms conjunct: ক + ্ + ক |
| `kok` | কক | Prevents conjunct by inserting inherent অ between consonants |
| `kOk` | কোক | Inserts the visible ও / ো vowel |

This is crucial when you need to represent two consecutive same letters without forming a conjunct. The vowel 'o' acts as a separator while being minimally pronounced in natural speech.

## 4. Special Vowel Rules

### 4.1 Vowel + Vowel Combinations

| Combination | Roman Input | Bengali Output |
|-------------|-------------|----------------|
| a + a | aa | আ |
| a + i | ai | আই |
| a + u | au | আউ |
| a + e | ae | আএ |
| a + o | ao | আও |
| i + a | ia | ইয়া |
| i + o | io | ইও |
| e + o | eo | এও |

> `aa` is a special case equivalent to independent আ (`A`) and আ-কার.
> `ee`/`ii` are explicit long-vowel aliases for `I`. `oo` follows Avro's short-উ signal; use `U` or `uu` for long ঊ.
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
| ক + ি | ki | কি |
| ক + ী | kI | কী |
| ক + ু | ku | কু |
| ক + ূ | kU | কূ |
| ক + ে | ke | কে |
| ক + ৈ | kOI | কৈ |
| ক + ো | kO | কো |
| ক + ৌ | kOU | কৌ |
| ক + ৃ | krri | কৃ |
