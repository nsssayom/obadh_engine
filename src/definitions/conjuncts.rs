//! Definitions for Bengali conjuncts
//!
//! This module provides a comprehensive set of Bengali conjunct definitions
//! based on phonetic components. The engine uses this compiled Rust rule table
//! directly; source CSV data is not parsed or shipped on the runtime path.

use crate::definitions::consonant_value;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Structure to store and manage Bengali conjunct definitions.
#[derive(Debug)]
pub struct ConjunctDefinitions {
    /// Trie for allocation-free conjunct lookup from component parts.
    conjunct_trie: ConjunctTrie,
}

#[derive(Debug, Clone, Copy)]
struct ConjunctRule(&'static str, &'static str);

impl ConjunctRule {
    fn key(self) -> &'static str {
        self.0
    }

    fn value(self) -> &'static str {
        self.1
    }
}

const CONJUNCT_RULES: &[ConjunctRule] = &[
    ConjunctRule("kk", "ক্ক"),
    ConjunctRule("kT", "ক্ট"),
    ConjunctRule("kTr", "ক্ট্র"),
    ConjunctRule("kt", "ক্ত"),
    ConjunctRule("ktr", "ক্ত্র"),
    ConjunctRule("kn", "ক্ন"),
    ConjunctRule("kw", "ক্ব"),
    ConjunctRule("km", "ক্ম"),
    ConjunctRule("ky", "ক্য"),
    ConjunctRule("kY", "ক্য"),
    ConjunctRule("kr", "ক্র"),
    ConjunctRule("kl", "ক্ল"),
    ConjunctRule("kSh", "ক্ষ"),
    ConjunctRule("ksh", "ক্ষ"),
    ConjunctRule("kkh", "ক্ষ"),
    ConjunctRule("kShN", "ক্ষ্ণ"),
    ConjunctRule("kshN", "ক্ষ্ণ"),
    ConjunctRule("kkhN", "ক্ষ্ণ"),
    ConjunctRule("kShw", "ক্ষ্ব"),
    ConjunctRule("kshw", "ক্ষ্ব"),
    ConjunctRule("kkhw", "ক্ষ্ব"),
    ConjunctRule("kShm", "ক্ষ্ম"),
    ConjunctRule("kshm", "ক্ষ্ম"),
    ConjunctRule("kkhm", "ক্ষ্ম"),
    ConjunctRule("kShy", "ক্ষ্য"),
    ConjunctRule("kShY", "ক্ষ্য"),
    ConjunctRule("kshy", "ক্ষ্য"),
    ConjunctRule("kkhy", "ক্ষ্য"),
    ConjunctRule("kkhY", "ক্ষ্য"),
    ConjunctRule("kShmy", "ক্ষ্ম্য"),
    ConjunctRule("kshmy", "ক্ষ্ম্য"),
    ConjunctRule("kkhmy", "ক্ষ্ম্য"),
    ConjunctRule("ks", "ক্স"),
    ConjunctRule("khy", "খ্য"),
    ConjunctRule("khY", "খ্য"),
    ConjunctRule("khr", "খ্র"),
    ConjunctRule("gN", "গ্‌ণ"),
    ConjunctRule("gdh", "গ্ধ"),
    ConjunctRule("gdhy", "গ্ধ্য"),
    ConjunctRule("gdhr", "গ্ধ্র"),
    ConjunctRule("gn", "গ্ন"),
    ConjunctRule("gny", "গ্ন্য"),
    ConjunctRule("gw", "গ্ব"),
    ConjunctRule("gm", "গ্ম"),
    ConjunctRule("gy", "গ্য"),
    ConjunctRule("gY", "গ্য"),
    ConjunctRule("gr", "গ্র"),
    ConjunctRule("gry", "গ্র্য"),
    ConjunctRule("gl", "গ্ল"),
    ConjunctRule("ghn", "ঘ্ন"),
    ConjunctRule("ghy", "ঘ্য"),
    ConjunctRule("ghY", "ঘ্য"),
    ConjunctRule("ghr", "ঘ্র"),
    ConjunctRule("Ngk", "ঙ্ক"),
    ConjunctRule("Ngkt", "ঙ্‌ক্ত"),
    ConjunctRule("Ngky", "ঙ্ক্য"),
    ConjunctRule("NgkY", "ঙ্ক্য"),
    ConjunctRule("NgkSh", "ঙ্ক্ষ"),
    ConjunctRule("Ngksh", "ঙ্ক্ষ"),
    ConjunctRule("Ngkh", "ঙ্খ"),
    ConjunctRule("Ngkhy", "ঙ্খ্য"),
    ConjunctRule("NgkhY", "ঙ্খ্য"),
    ConjunctRule("Ngg", "ঙ্গ"),
    ConjunctRule("Nggy", "ঙ্গ্য"),
    ConjunctRule("NggY", "ঙ্গ্য"),
    ConjunctRule("Nggh", "ঙ্ঘ"),
    ConjunctRule("Ngghy", "ঙ্ঘ্য"),
    ConjunctRule("NgghY", "ঙ্ঘ্য"),
    ConjunctRule("Ngghr", "ঙ্ঘ্র"),
    ConjunctRule("Ngm", "ঙ্ম"),
    ConjunctRule("cc", "চ্চ"),
    ConjunctRule("cch", "চ্ছ"),
    ConjunctRule("cC", "চ্ছ"),
    ConjunctRule("cCh", "চ্ছ"),
    ConjunctRule("cChh", "চ্ছ"),
    ConjunctRule("cchw", "চ্ছ্ব"),
    ConjunctRule("cCw", "চ্ছ্ব"),
    ConjunctRule("cChw", "চ্ছ্ব"),
    ConjunctRule("cChhw", "চ্ছ্ব"),
    ConjunctRule("cchr", "চ্ছ্র"),
    ConjunctRule("cCr", "চ্ছ্র"),
    ConjunctRule("cChr", "চ্ছ্র"),
    ConjunctRule("cChhr", "চ্ছ্র"),
    ConjunctRule("cNG", "চ্ঞ"),
    ConjunctRule("cw", "চ্ব"),
    ConjunctRule("cy", "চ্য"),
    ConjunctRule("cY", "চ্য"),
    ConjunctRule("jj", "জ্জ"),
    ConjunctRule("Jj", "জ্জ"),
    ConjunctRule("jJ", "জ্জ"),
    ConjunctRule("JJ", "জ্জ"),
    ConjunctRule("jjw", "জ্জ্ব"),
    ConjunctRule("Jjw", "জ্জ্ব"),
    ConjunctRule("jJw", "জ্জ্ব"),
    ConjunctRule("JJw", "জ্জ্ব"),
    ConjunctRule("jjh", "জ্ঝ"),
    ConjunctRule("Jjh", "জ্ঝ"),
    ConjunctRule("jNG", "জ্ঞ"),
    ConjunctRule("JNG", "জ্ঞ"),
    ConjunctRule("jn", "জ্ঞ"),
    ConjunctRule("Jn", "জ্ঞ"),
    ConjunctRule("jw", "জ্ব"),
    ConjunctRule("Jw", "জ্ব"),
    ConjunctRule("jy", "জ্য"),
    ConjunctRule("Jy", "জ্য"),
    ConjunctRule("jY", "জ্য"),
    ConjunctRule("JY", "জ্য"),
    ConjunctRule("jr", "জ্র"),
    ConjunctRule("Jr", "জ্র"),
    ConjunctRule("NGc", "ঞ্চ"),
    ConjunctRule("NGch", "ঞ্ছ"),
    ConjunctRule("NGj", "ঞ্জ"),
    ConjunctRule("NGJ", "ঞ্জ"),
    ConjunctRule("NGjh", "ঞ্ঝ"),
    ConjunctRule("TT", "ট্ট"),
    ConjunctRule("Tw", "ট্ব"),
    ConjunctRule("Tm", "ট্ম"),
    ConjunctRule("Ty", "ট্য"),
    ConjunctRule("TY", "ট্য"),
    ConjunctRule("Tr", "ট্র"),
    ConjunctRule("DD", "ড্ড"),
    ConjunctRule("Dw", "ড্ব"),
    ConjunctRule("Dm", "ড্ম"),
    ConjunctRule("Dy", "ড্য"),
    ConjunctRule("DY", "ড্য"),
    ConjunctRule("Dr", "ড্র"),
    ConjunctRule("Rg", "ড়্গ"),
    ConjunctRule("Dhy", "ঢ্য"),
    ConjunctRule("DhY", "ঢ্য"),
    ConjunctRule("Dhr", "ঢ্র"),
    ConjunctRule("NT", "ণ্ট"),
    ConjunctRule("NTh", "ণ্ঠ"),
    ConjunctRule("NThy", "ণ্ঠ্য"),
    ConjunctRule("NThY", "ণ্ঠ্য"),
    ConjunctRule("ND", "ণ্ড"),
    ConjunctRule("NDy", "ণ্ড্য"),
    ConjunctRule("NDY", "ণ্ড্য"),
    ConjunctRule("NDr", "ণ্ড্র"),
    ConjunctRule("NDh", "ণ্ঢ"),
    ConjunctRule("NN", "ণ্ণ"),
    ConjunctRule("Nw", "ণ্ব"),
    ConjunctRule("Nm", "ণ্ম"),
    ConjunctRule("Ny", "ণ্য"),
    ConjunctRule("NY", "ণ্য"),
    ConjunctRule("tk", "ৎক"),
    ConjunctRule("tkh", "ৎখ"),
    ConjunctRule("tt", "ত্ত"),
    ConjunctRule("ttr", "ত্ত্র"),
    ConjunctRule("ttw", "ত্ত্ব"),
    ConjunctRule("tty", "ত্ত্য"),
    ConjunctRule("ttY", "ত্ত্য"),
    ConjunctRule("tth", "ত্থ"),
    ConjunctRule("tn", "ত্ন"),
    ConjunctRule("tp", "ৎপ"),
    ConjunctRule("tw", "ত্ব"),
    ConjunctRule("tm", "ত্ম"),
    ConjunctRule("tmy", "ত্ম্য"),
    ConjunctRule("ty", "ত্য"),
    ConjunctRule("tY", "ত্য"),
    ConjunctRule("tr", "ত্র"),
    ConjunctRule("try", "ত্র্য"),
    ConjunctRule("tl", "ৎল"),
    ConjunctRule("ts", "ৎস"),
    ConjunctRule("thw", "থ্ব"),
    ConjunctRule("thy", "থ্য"),
    ConjunctRule("thY", "থ্য"),
    ConjunctRule("thr", "থ্র"),
    ConjunctRule("dg", "দ্গ"),
    ConjunctRule("dgh", "দ্ঘ"),
    ConjunctRule("dd", "দ্দ"),
    ConjunctRule("ddw", "দ্দ্ব"),
    ConjunctRule("ddh", "দ্ধ"),
    ConjunctRule("dw", "দ্ব"),
    ConjunctRule("dbh", "দ্ভ"),
    ConjunctRule("dv", "দ্ভ"),
    ConjunctRule("dbhr", "দ্ভ্র"),
    ConjunctRule("dvr", "দ্ভ্র"),
    ConjunctRule("dm", "দ্ম"),
    ConjunctRule("dy", "দ্য"),
    ConjunctRule("dY", "দ্য"),
    ConjunctRule("dr", "দ্র"),
    ConjunctRule("dry", "দ্র্য"),
    ConjunctRule("dhn", "ধ্ন"),
    ConjunctRule("dhw", "ধ্ব"),
    ConjunctRule("dhm", "ধ্ম"),
    ConjunctRule("dhy", "ধ্য"),
    ConjunctRule("dhY", "ধ্য"),
    ConjunctRule("dhr", "ধ্র"),
    ConjunctRule("nT", "ন্ট"),
    ConjunctRule("nTr", "ন্ট্র"),
    ConjunctRule("nTh", "ন্ঠ"),
    ConjunctRule("nD", "ন্ড"),
    ConjunctRule("nDr", "ন্ড্র"),
    ConjunctRule("nt", "ন্ত"),
    ConjunctRule("ntw", "ন্ত্ব"),
    ConjunctRule("nty", "ন্ত্য"),
    ConjunctRule("ntY", "ন্ত্য"),
    ConjunctRule("ntr", "ন্ত্র"),
    ConjunctRule("ntry", "ন্ত্র্য"),
    ConjunctRule("nth", "ন্থ"),
    ConjunctRule("nthr", "ন্থ্র"),
    ConjunctRule("nd", "ন্দ"),
    ConjunctRule("ndy", "ন্দ্য"),
    ConjunctRule("ndY", "ন্দ্য"),
    ConjunctRule("ndw", "ন্দ্ব"),
    ConjunctRule("ndr", "ন্দ্র"),
    ConjunctRule("ndh", "ন্ধ"),
    ConjunctRule("ndhy", "ন্ধ্য"),
    ConjunctRule("ndhY", "ন্ধ্য"),
    ConjunctRule("ndhr", "ন্ধ্র"),
    ConjunctRule("nn", "ন্ন"),
    ConjunctRule("nw", "ন্ব"),
    ConjunctRule("nm", "ন্ম"),
    ConjunctRule("ny", "ন্য"),
    ConjunctRule("nY", "ন্য"),
    ConjunctRule("pT", "প্ট"),
    ConjunctRule("pt", "প্ত"),
    ConjunctRule("pn", "প্ন"),
    ConjunctRule("pp", "প্প"),
    ConjunctRule("py", "প্য"),
    ConjunctRule("pY", "প্য"),
    ConjunctRule("pr", "প্র"),
    ConjunctRule("pry", "প্র্য"),
    ConjunctRule("pl", "প্ল"),
    ConjunctRule("ps", "প্স"),
    ConjunctRule("fr", "ফ্র"),
    ConjunctRule("phr", "ফ্র"),
    ConjunctRule("fl", "ফ্ল"),
    ConjunctRule("phl", "ফ্ল"),
    ConjunctRule("bj", "ব্জ"),
    ConjunctRule("bJ", "ব্জ"),
    ConjunctRule("bd", "ব্দ"),
    ConjunctRule("bdh", "ব্ধ"),
    ConjunctRule("bb", "ব্ব"),
    ConjunctRule("bw", "ব্ব"),
    ConjunctRule("by", "ব্য"),
    ConjunctRule("bY", "ব্য"),
    ConjunctRule("br", "ব্র"),
    ConjunctRule("bl", "ব্ল"),
    ConjunctRule("bhw", "ভ্ব"),
    ConjunctRule("vw", "ভ্ব"),
    ConjunctRule("bhy", "ভ্য"),
    ConjunctRule("bhY", "ভ্য"),
    ConjunctRule("vy", "ভ্য"),
    ConjunctRule("vY", "ভ্য"),
    ConjunctRule("bhr", "ভ্র"),
    ConjunctRule("vr", "ভ্র"),
    ConjunctRule("bhl", "ভ্ল"),
    ConjunctRule("vl", "ভ্ল"),
    ConjunctRule("mn", "ম্ন"),
    ConjunctRule("mp", "ম্প"),
    ConjunctRule("mpr", "ম্প্র"),
    ConjunctRule("mf", "ম্ফ"),
    ConjunctRule("mph", "ম্ফ"),
    ConjunctRule("mw", "ম্ব"),
    ConjunctRule("mb", "ম্ব"),
    ConjunctRule("mwr", "ম্ব্র"),
    ConjunctRule("mbr", "ম্ব্র"),
    ConjunctRule("mbh", "ম্ভ"),
    ConjunctRule("mv", "ম্ভ"),
    ConjunctRule("mbhr", "ম্ভ্র"),
    ConjunctRule("mvr", "ম্ভ্র"),
    ConjunctRule("mm", "ম্ম"),
    ConjunctRule("my", "ম্য"),
    ConjunctRule("mY", "ম্য"),
    ConjunctRule("mr", "ম্র"),
    ConjunctRule("ml", "ম্ল"),
    ConjunctRule("yy", "য্য"),
    ConjunctRule("zy", "য্য"),
    ConjunctRule("zY", "য্য"),
    ConjunctRule("rrk", "র্ক"),
    ConjunctRule("rrky", "র্ক্য"),
    ConjunctRule("rrkY", "র্ক্য"),
    ConjunctRule("rrg", "র্গ"),
    ConjunctRule("rrgy", "র্গ্য"),
    ConjunctRule("rrgY", "র্গ্য"),
    ConjunctRule("rrghy", "র্ঘ্য"),
    ConjunctRule("rrghY", "র্ঘ্য"),
    ConjunctRule("rrNgg", "র্ঙ্গ"),
    ConjunctRule("rrcy", "র্চ্য"),
    ConjunctRule("rrcY", "র্চ্য"),
    ConjunctRule("rrjy", "র্জ্য"),
    ConjunctRule("rrjY", "র্জ্য"),
    ConjunctRule("rrJy", "র্জ্য"),
    ConjunctRule("rrJY", "র্জ্য"),
    ConjunctRule("rrjj", "র্জ্জ"),
    ConjunctRule("rrjJ", "র্জ্জ"),
    ConjunctRule("rrJj", "র্জ্জ"),
    ConjunctRule("rrJJ", "র্জ্জ"),
    ConjunctRule("rrjNG", "র্জ্ঞ"),
    ConjunctRule("rrJNG", "র্জ্ঞ"),
    ConjunctRule("rrjn", "র্জ্ঞ"),
    ConjunctRule("rrJn", "র্জ্ঞ"),
    ConjunctRule("rrNy", "র্ণ্য"),
    ConjunctRule("rrNY", "র্ণ্য"),
    ConjunctRule("rrty", "র্ত্য"),
    ConjunctRule("rrtY", "র্ত্য"),
    ConjunctRule("rrthy", "র্থ্য"),
    ConjunctRule("rrthY", "র্থ্য"),
    ConjunctRule("rrwy", "র্ব্য"),
    ConjunctRule("rrwY", "র্ব্য"),
    ConjunctRule("rrmy", "র্ম্য"),
    ConjunctRule("rrmY", "র্ম্য"),
    ConjunctRule("rrshy", "র্শ্য"),
    ConjunctRule("rrshY", "র্শ্য"),
    ConjunctRule("rrSy", "র্শ্য"),
    ConjunctRule("rrSY", "র্শ্য"),
    ConjunctRule("rrShy", "র্ষ্য"),
    ConjunctRule("rrShY", "র্ষ্য"),
    ConjunctRule("rrhy", "র্হ্য"),
    ConjunctRule("rrhY", "র্হ্য"),
    ConjunctRule("rrkh", "র্খ"),
    ConjunctRule("rrgr", "র্গ্র"),
    ConjunctRule("rrgh", "র্ঘ"),
    ConjunctRule("rrc", "র্চ"),
    ConjunctRule("rrch", "র্ছ"),
    ConjunctRule("rrj", "র্জ"),
    ConjunctRule("rrJ", "র্জ"),
    ConjunctRule("rrjh", "র্ঝ"),
    ConjunctRule("rrT", "র্ট"),
    ConjunctRule("rrD", "র্ড"),
    ConjunctRule("rrN", "র্ণ"),
    ConjunctRule("rrt", "র্ত"),
    ConjunctRule("rrtm", "র্ত্ম"),
    ConjunctRule("rrtr", "র্ত্র"),
    ConjunctRule("rrth", "র্থ"),
    ConjunctRule("rrd", "র্দ"),
    ConjunctRule("rrdw", "র্দ্ব"),
    ConjunctRule("rrdr", "র্দ্র"),
    ConjunctRule("rrdh", "র্ধ"),
    ConjunctRule("rrdhw", "র্ধ্ব"),
    ConjunctRule("rrn", "র্ন"),
    ConjunctRule("rrp", "র্প"),
    ConjunctRule("rrf", "র্ফ"),
    ConjunctRule("rrph", "র্ফ"),
    ConjunctRule("rrw", "র্ব"),
    ConjunctRule("rrbh", "র্ভ"),
    ConjunctRule("rrv", "র্ভ"),
    ConjunctRule("rrm", "র্ম"),
    ConjunctRule("rry", "র্য"),
    ConjunctRule("rrY", "র্য"),
    ConjunctRule("rrl", "র্ল"),
    ConjunctRule("rrsh", "র্শ"),
    ConjunctRule("rrS", "র্শ"),
    ConjunctRule("rrshw", "র্শ্ব"),
    ConjunctRule("rrSw", "র্শ্ব"),
    ConjunctRule("rrSh", "র্ষ"),
    ConjunctRule("rrShT", "র্ষ্ট"),
    ConjunctRule("rrShN", "র্ষ্ণ"),
    ConjunctRule("rrShNy", "র্ষ্ণ্য"),
    ConjunctRule("rrShNY", "র্ষ্ণ্য"),
    ConjunctRule("rrs", "র্স"),
    ConjunctRule("rrh", "র্হ"),
    ConjunctRule("rrDhy", "র্ঢ্য"),
    ConjunctRule("rrDhY", "র্ঢ্য"),
    ConjunctRule("lk", "ল্ক"),
    ConjunctRule("lky", "ল্ক্য"),
    ConjunctRule("lkY", "ল্ক্য"),
    ConjunctRule("lg", "ল্গ"),
    ConjunctRule("lT", "ল্ট"),
    ConjunctRule("lD", "ল্ড"),
    ConjunctRule("lp", "ল্প"),
    ConjunctRule("lf", "ল্ফ"),
    ConjunctRule("lph", "ল্ফ"),
    ConjunctRule("lw", "ল্ব"),
    ConjunctRule("lbh", "ল্ভ"),
    ConjunctRule("lv", "ল্ভ"),
    ConjunctRule("lm", "ল্ম"),
    ConjunctRule("ly", "ল্য"),
    ConjunctRule("lY", "ল্য"),
    ConjunctRule("ll", "ল্ল"),
    ConjunctRule("shc", "শ্চ"),
    ConjunctRule("Sc", "শ্চ"),
    ConjunctRule("shch", "শ্ছ"),
    ConjunctRule("Sch", "শ্ছ"),
    ConjunctRule("shn", "শ্ন"),
    ConjunctRule("Sn", "শ্ন"),
    ConjunctRule("shw", "শ্ব"),
    ConjunctRule("Sw", "শ্ব"),
    ConjunctRule("shm", "শ্ম"),
    ConjunctRule("Sm", "শ্ম"),
    ConjunctRule("shy", "শ্য"),
    ConjunctRule("shY", "শ্য"),
    ConjunctRule("Sy", "শ্য"),
    ConjunctRule("SY", "শ্য"),
    ConjunctRule("shr", "শ্র"),
    ConjunctRule("Sr", "শ্র"),
    ConjunctRule("shl", "শ্ল"),
    ConjunctRule("Sl", "শ্ল"),
    ConjunctRule("Shk", "ষ্ক"),
    ConjunctRule("Shkw", "ষ্ক্ব"),
    ConjunctRule("Shkr", "ষ্ক্র"),
    ConjunctRule("ShT", "ষ্ট"),
    ConjunctRule("ShTy", "ষ্ট্য"),
    ConjunctRule("ShTY", "ষ্ট্য"),
    ConjunctRule("ShTr", "ষ্ট্র"),
    ConjunctRule("ShTh", "ষ্ঠ"),
    ConjunctRule("ShThy", "ষ্ঠ্য"),
    ConjunctRule("ShThY", "ষ্ঠ্য"),
    ConjunctRule("ShN", "ষ্ণ"),
    ConjunctRule("ShNw", "ষ্ণ্ব"),
    ConjunctRule("Shp", "ষ্প"),
    ConjunctRule("Shpr", "ষ্প্র"),
    ConjunctRule("Shf", "ষ্ফ"),
    ConjunctRule("Shph", "ষ্ফ"),
    ConjunctRule("Shw", "ষ্ব"),
    ConjunctRule("Shm", "ষ্ম"),
    ConjunctRule("Shy", "ষ্য"),
    ConjunctRule("ShY", "ষ্য"),
    ConjunctRule("sk", "স্ক"),
    ConjunctRule("skr", "স্ক্র"),
    ConjunctRule("skh", "স্খ"),
    ConjunctRule("sT", "স্ট"),
    ConjunctRule("sTr", "স্ট্র"),
    ConjunctRule("st", "স্ত"),
    ConjunctRule("stw", "স্ত্ব"),
    ConjunctRule("sty", "স্ত্য"),
    ConjunctRule("stY", "স্ত্য"),
    ConjunctRule("str", "স্ত্র"),
    ConjunctRule("sth", "স্থ"),
    ConjunctRule("sthy", "স্থ্য"),
    ConjunctRule("sthY", "স্থ্য"),
    ConjunctRule("sn", "স্ন"),
    ConjunctRule("sny", "স্ন্য"),
    ConjunctRule("snY", "স্ন্য"),
    ConjunctRule("sp", "স্প"),
    ConjunctRule("spr", "স্প্র"),
    ConjunctRule("spl", "স্প্‌ল"),
    ConjunctRule("sf", "স্ফ"),
    ConjunctRule("sph", "স্ফ"),
    ConjunctRule("sw", "স্ব"),
    ConjunctRule("sm", "স্ম"),
    ConjunctRule("sy", "স্য"),
    ConjunctRule("sY", "স্য"),
    ConjunctRule("sr", "স্র"),
    ConjunctRule("sl", "স্ল"),
    ConjunctRule("hN", "হ্ণ"),
    ConjunctRule("hn", "হ্ন"),
    ConjunctRule("hw", "হ্ব"),
    ConjunctRule("hm", "হ্ম"),
    ConjunctRule("hy", "হ্য"),
    ConjunctRule("hY", "হ্য"),
    ConjunctRule("hr", "হ্র"),
    ConjunctRule("hl", "হ্ল"),
];

impl ConjunctDefinitions {
    /// Create a new instance of conjunct definitions
    pub fn new() -> Self {
        // Initialize containers
        let conjunct_trie = ConjunctTrie::with_capacity(conjunct_trie_node_capacity());

        let mut instance = ConjunctDefinitions { conjunct_trie };

        for rule in CONJUNCT_RULES {
            instance.add_conjunct(rule.key(), rule.value());
        }

        instance.conjunct_trie.sort_edges();

        // Return the populated instance
        instance
    }

    /// Add a conjunct mapping
    fn add_conjunct(&mut self, key: &'static str, value: &'static str) {
        self.conjunct_trie.insert(key, value);
    }

    /// Check if a sequence can form a valid conjunct
    pub fn can_form_conjunct(&self, key: &str) -> bool {
        self.create_conjunct(key).is_some()
    }

    /// Create a conjunct from a sequence of consonants
    pub fn create_conjunct(&self, key: &str) -> Option<&'static str> {
        let node = self.conjunct_trie.advance(self.conjunct_trie.root(), key)?;
        self.conjunct_trie.value(node)
    }

    fn canonical_conjunct_part<'a>(&self, part: &'a str) -> &'a str {
        canonical_conjunct_part(part)
    }

    /// Create a conjunct from already-tokenized component parts without
    /// allocating a joined key.
    pub fn create_conjunct_from_parts(&self, parts: &[&str]) -> Option<&'static str> {
        if parts.len() < 2 {
            return None;
        }

        let mut node = self.conjunct_trie.root();
        for part in parts {
            node = self
                .conjunct_trie
                .advance(node, self.canonical_conjunct_part(part))?;
        }

        self.conjunct_trie.value(node)
    }

    /// Check component parts without allocating a joined key.
    pub fn can_form_conjunct_from_parts(&self, parts: &[&str]) -> bool {
        self.create_conjunct_from_parts(parts).is_some()
    }

    /// Return the root trie cursor for incremental conjunct matching.
    pub(crate) fn conjunct_match_root(&self) -> usize {
        self.conjunct_trie.root()
    }

    /// Advance an incremental conjunct match by one romanized component.
    pub(crate) fn advance_conjunct_match(&self, node: usize, part: &str) -> Option<usize> {
        self.conjunct_trie
            .advance(node, self.canonical_conjunct_part(part))
    }

    /// Return the conjunct value at an incremental trie cursor, if terminal.
    pub(crate) fn conjunct_match_value(&self, node: usize) -> Option<&'static str> {
        self.conjunct_trie.value(node)
    }

    /// Get romanized consonants for a conjunct
    pub fn get_components(&self, conjunct: &str) -> Option<Vec<String>> {
        for rule in CONJUNCT_RULES {
            if rule.value() == conjunct {
                return Some(self.components_for_key(rule.key()));
            }
        }
        None
    }

    fn components_for_key(&self, key: &str) -> Vec<String> {
        let mut components = Vec::new();
        let mut i = 0;
        while i < key.len() {
            let mut found = false;
            for len in (1..=key.len() - i).rev() {
                let substr = &key[i..i + len];
                if consonant_value(substr).is_some() || is_special_form_key(substr) {
                    components.push(substr.to_string());
                    i += len;
                    found = true;
                    break;
                }
            }
            if !found {
                components.push(key[i..i + 1].to_string());
                i += 1;
            }
        }
        components
    }

    /// Check if a given sequence is a valid conjunct
    pub fn is_valid_conjunct(&self, components: &[String]) -> bool {
        if components.is_empty() {
            return false;
        }

        let mut node = self.conjunct_trie.root();
        for component in components {
            let Some(next_node) = self
                .conjunct_trie
                .advance(node, self.canonical_conjunct_part(component))
            else {
                return false;
            };
            node = next_node;
        }

        self.conjunct_trie.value(node).is_some()
    }

    /// Get all valid conjuncts
    pub fn get_all_valid_conjuncts(&self) -> BTreeSet<&'static str> {
        CONJUNCT_RULES.iter().map(|rule| rule.key()).collect()
    }

    /// Check if a form is a special form (like reph, ya-phola, ba-phola)
    pub fn is_special_form(&self, form: &str) -> bool {
        is_special_form_key(form)
    }
}

impl Default for ConjunctDefinitions {
    fn default() -> Self {
        Self::new()
    }
}

fn canonical_conjunct_part(part: &str) -> &str {
    match part {
        "chh" | "C" | "Ch" | "CH" | "Chh" | "CHH" => "ch",
        "Kh" | "KH" => "kh",
        "Gh" | "GH" => "gh",
        "J" => "j",
        "Jh" | "JH" => "jh",
        "TH" => "Th",
        "DH" => "Dh",
        "Ph" | "PH" | "f" => "ph",
        "Bh" | "BH" | "v" => "bh",
        "Y" => "y",
        "S" => "sh",
        "SH" => "Sh",
        _ => part,
    }
}

fn is_special_form_key(form: &str) -> bool {
    matches!(form, "rr" | "y" | "Y" | "w")
}

/// Return a singleton instance of ConjunctDefinitions
pub fn conjuncts() -> &'static ConjunctDefinitions {
    static INSTANCE: OnceLock<ConjunctDefinitions> = OnceLock::new();
    INSTANCE.get_or_init(ConjunctDefinitions::new)
}

#[derive(Debug)]
struct ConjunctTrie {
    nodes: Vec<ConjunctTrieNode>,
}

#[derive(Debug, Default)]
struct ConjunctTrieNode {
    value: Option<&'static str>,
    edges: Vec<ConjunctTrieEdge>,
}

#[derive(Debug, Clone, Copy)]
struct ConjunctTrieEdge {
    byte: u8,
    node: usize,
}

impl ConjunctTrie {
    fn with_capacity(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(ConjunctTrieNode::default());

        Self { nodes }
    }

    fn root(&self) -> usize {
        0
    }

    fn insert(&mut self, key: &'static str, value: &'static str) {
        let mut node = self.root();

        for byte in key.bytes() {
            node = self.child_or_insert(node, byte);
        }

        assert!(
            self.nodes[node].value.is_none(),
            "duplicate conjunct trie key: {key}"
        );
        self.nodes[node].value = Some(value);
    }

    fn child_or_insert(&mut self, node: usize, byte: u8) -> usize {
        if let Some(edge) = self.nodes[node].edges.iter().find(|edge| edge.byte == byte) {
            return edge.node;
        }

        let child = self.nodes.len();
        self.nodes.push(ConjunctTrieNode::default());
        self.nodes[node]
            .edges
            .push(ConjunctTrieEdge { byte, node: child });
        child
    }

    fn sort_edges(&mut self) {
        for node in &mut self.nodes {
            node.edges.sort_unstable_by_key(|edge| edge.byte);
        }
    }

    fn advance(&self, node: usize, part: &str) -> Option<usize> {
        let mut current = node;

        for byte in part.bytes() {
            current = self.nodes.get(current)?.child(byte)?;
        }

        Some(current)
    }

    fn value(&self, node: usize) -> Option<&'static str> {
        self.nodes.get(node)?.value
    }
}

impl ConjunctTrieNode {
    fn child(&self, byte: u8) -> Option<usize> {
        self.edges
            .binary_search_by_key(&byte, |edge| edge.byte)
            .ok()
            .map(|index| self.edges[index].node)
    }
}

fn conjunct_trie_node_capacity() -> usize {
    1 + CONJUNCT_RULES
        .iter()
        .map(|rule| rule.key().len())
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conjunct_rule_table_has_unique_keys() {
        let mut keys = BTreeSet::new();

        for rule in CONJUNCT_RULES {
            assert!(!rule.key().is_empty());
            assert!(!rule.value().is_empty());
            assert!(
                keys.insert(rule.key()),
                "duplicate conjunct rule key: {}",
                rule.key()
            );
        }
    }

    #[test]
    fn conjunct_definitions_load_every_static_rule() {
        let definitions = ConjunctDefinitions::new();

        assert_eq!(
            definitions.get_all_valid_conjuncts().len(),
            CONJUNCT_RULES.len()
        );

        for rule in CONJUNCT_RULES {
            assert_eq!(definitions.create_conjunct(rule.key()), Some(rule.value()));
        }
    }
}
