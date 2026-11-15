//! Logical chunking (H-GLUE style): paragraph / sentence-group boundaries,
//! never fixed windows. Chunks are ephemeral — they scope extraction spans,
//! they are not stored (no session duplication).

/// One semantically coherent unit of text.
#[derive(Debug, Clone)]
pub struct LogicalUnit {
    pub text: String,
    pub unit_type: &'static str,
    pub start: usize,
    pub end: usize,
}

pub struct LogicalChunker {
    pub min_unit_size: usize,
    pub max_unit_size: usize,
    pub merge_threshold: usize,
}

impl Default for LogicalChunker {
    fn default() -> Self {
        Self { min_unit_size: 100, max_unit_size: 2000, merge_threshold: 150 }
    }
}

impl LogicalChunker {
    pub fn chunk(&self, text: &str) -> Vec<LogicalUnit> {
        let mut units: Vec<LogicalUnit> = Vec::new();
        for (para, start, end) in split_paragraphs(text) {
            if para.trim().len() < self.min_unit_size {
                if let Some(last) = units.last_mut() {
                    if last.text.len() + para.len() < self.max_unit_size {
                        last.text.push_str("\n\n");
                        last.text.push_str(&para);
                        last.end = end;
                        continue;
                    }
                }
            }
            if para.len() > self.max_unit_size {
                units.extend(split_sentence_groups(&para, start, self.max_unit_size));
            } else {
                units.push(LogicalUnit { text: para, unit_type: "paragraph", start, end });
            }
        }
        // Merge trailing tiny units into their predecessor.
        let mut out: Vec<LogicalUnit> = Vec::with_capacity(units.len());
        for unit in units {
            if unit.text.len() < self.merge_threshold {
                if let Some(last) = out.last_mut() {
                    last.text.push_str("\n\n");
                    last.text.push_str(&unit.text);
                    last.end = unit.end;
                    continue;
                }
            }
            out.push(unit);
        }
        out
    }
}

fn split_paragraphs(text: &str) -> Vec<(String, usize, usize)> {
    let mut paras = Vec::new();
    let mut last_end = 0usize;
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("\n\n") {
        let idx = search_from + rel;
        let para = text[last_end..idx].trim().to_string();
        if !para.is_empty() {
            paras.push((para, last_end, idx));
        }
        // Skip the whole blank run.
        let mut end = idx;
        while text[end..].starts_with('\n') {
            end += 1;
        }
        last_end = end;
        search_from = end;
    }
    let tail = text[last_end..].trim().to_string();
    if !tail.is_empty() {
        paras.push((tail, last_end, text.len()));
    }
    paras
}

/// Split text into sentences, each with its char offset into `text`.
/// A sentence ends at `.`/`!`/`?` followed by whitespace (or end of text).
/// Abbreviations that never end a sentence (matched lowercase).
const SENTENCE_ABBREV: &[&str] = &[
    "mr", "mrs", "ms", "dr", "st", "vs", "jr", "sr", "inc", "ltd", "co", "ave", "blvd", "rd",
    "univ", "prof", "gen", "col", "maj", "capt", "lt", "sgt", "rev", "hon", "pres", "gov",
    "sen", "rep", "amb", "sec", "dept", "no", "al", "eg", "ie", "etc", "approx", "fig",
];

/// Token (lowercased, alphanumeric) ending just before `end` (exclusive),
/// or an empty string at text start.
fn preceding_token(chars: &[char], end: usize) -> String {
    let mut s = end;
    while s > 0 && chars[s - 1].is_alphanumeric() {
        s -= 1;
    }
    chars[s..end].iter().collect::<String>().to_lowercase()
}

/// Split text into sentences, returning each trimmed sentence with
/// its exact char span `(text, start, end)` in the input. Offsets are
/// exact (no leading/trailing whitespace included) so callers can map
/// character-indexed annotations into sentence coordinates.
pub fn split_sentences(text: &str) -> Vec<(String, usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < chars.len() && chars[start].is_whitespace() {
        start += 1;
    }
    let mut i = start;
    while i < chars.len() {
        let c = chars[i];
        if matches!(c, '.' | '!' | '?') {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            // Sentence boundary only at whitespace/end (avoids "v1.2" splits).
            // Single uppercase initials ("J. Robert") and abbreviations
            // ("Mr. Smith") never end a sentence.
            let mut boundary = j > i + 1 || j == chars.len();
            if boundary {
                let tok = preceding_token(&chars, i);
                let single_upper = {
                    let t: String = chars[i.saturating_sub(tok.len())..i].iter().collect();
                    t.chars().count() == 1
                        && t.chars().next().map(|ch| ch.is_ascii_uppercase()).unwrap_or(false)
                        && t != "I"
                };
                if single_upper || SENTENCE_ABBREV.contains(&tok.as_str()) {
                    boundary = false;
                }
            }
            if boundary {
                out.push((chars[start..=i].iter().collect(), start, i + 1));
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if start < chars.len() {
        // Trim trailing whitespace from the final segment for an exact end.
        let mut end = chars.len();
        while end > start && chars[end - 1].is_whitespace() {
            end -= 1;
        }
        if end > start {
            out.push((chars[start..end].iter().collect(), start, end));
        }
    }
    out
}

/// Split one sentence into scoring clauses on `;`, keeping fact-relative
/// char offsets. Long multi-clause sentences degrade the pair scorer the
/// same way paragraphs do, so each clause is an independent scoring call
/// (mentions are filtered to the clause span, exactly like sentences).
/// Semicolons inside parentheses/brackets never split: they join
/// appositives and date ranges ("(born X; April 22, 1904 – ...)"), not
/// independent clauses. Sentences without top-level `;` come back whole.
pub fn split_clauses(sent_text: &str, sent_start: usize) -> Vec<(String, usize, usize)> {
    let chars: Vec<char> = sent_text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    let push = |s: usize, e: usize, out: &mut Vec<(String, usize, usize)>| {
        // Trim edge whitespace; drop empties.
        let mut a = s;
        let mut b = e;
        while a < b && chars[a].is_whitespace() {
            a += 1;
        }
        while b > a && chars[b - 1].is_whitespace() {
            b -= 1;
        }
        if b > a {
            out.push((chars[a..b].iter().collect(), sent_start + a, sent_start + b));
        }
    };
    for (i, c) in chars.iter().enumerate() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' if depth > 0 => depth -= 1,
            ';' if depth == 0 => {
                push(start, i, &mut out);
                start = i + 1;
            }
            _ => {}
        }
    }
    push(start, chars.len(), &mut out);
    out
}

fn split_sentence_groups(text: &str, base_offset: usize, max_unit_size: usize) -> Vec<LogicalUnit> {
    let mut units = Vec::new();
    let mut sentences: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            sentences.push(text[start..i + 1].trim());
            start = j;
            i = j;
        } else {
            i += 1;
        }
    }
    if start < text.len() && !text[start..].trim().is_empty() {
        sentences.push(text[start..].trim());
    }

    let mut current: Vec<&str> = Vec::new();
    let mut current_start = base_offset;
    let mut current_len = 0usize;
    for sent in sentences {
        if current_len + sent.len() > max_unit_size && !current.is_empty() {
            let joined = current.join(" ");
            let end = current_start + joined.len();
            units.push(LogicalUnit { text: joined, unit_type: "sentence_group", start: current_start, end });
            current_start = end + 1;
            current.clear();
            current_len = 0;
        }
        current.push(sent);
        current_len += sent.len() + 1;
    }
    if !current.is_empty() {
        let joined = current.join(" ");
        let end = current_start + joined.len();
        units.push(LogicalUnit { text: joined, unit_type: "sentence_group", start: current_start, end });
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_unit() {
        let units = LogicalChunker::default().chunk("User prefers Python for scripting.");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, "paragraph");
    }

    #[test]
    fn paragraphs_split_and_tiny_merges() {
        let long_a = "First substantial paragraph ".repeat(10);
        let long_b = "Second substantial paragraph ".repeat(10);
        let tiny = "Hi.";
        let text = format!("{long_a}\n\n{long_b}\n\n{tiny}");
        let units = LogicalChunker::default().chunk(&text);
        // Two long paragraphs stay separate; the tiny tail merges into the last.
        assert_eq!(units.len(), 2);
        assert!(units[1].text.ends_with("Hi."));
    }
}

#[cfg(test)]
mod split_tests {
    use super::{split_clauses, split_sentences};

    fn texts(s: &str) -> Vec<String> {
        split_sentences(s).into_iter().map(|(t, _, _)| t).collect()
    }

    #[test]
    fn initials_do_not_split() {
        let parts = texts("J. Robert Oppenheimer was a physicist. He directed the lab.");
        assert_eq!(parts.len(), 2);
        assert!(parts[0].starts_with("J. Robert"));
    }

    #[test]
    fn abbreviations_do_not_split() {
        let parts = texts("Mr. Smith met Dr. Jones in Washington. They talked.");
        assert_eq!(parts.len(), 2);
        assert!(parts[0].starts_with("Mr. Smith"));
    }

    #[test]
    fn pronoun_i_still_splits() {
        let parts = texts("Who am I. You are here.");
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn normal_sentences_still_split() {
        let parts = texts("First sentence. Second here! Third?");
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn parenthesized_semicolons_do_not_split() {
        // The birth/death date appositive must stay one clause so its
        // pairs still meet: naive splitting orphaned every Oppenheimer
        // edge in the wild.
        let text = "J. Robert Oppenheimer (born X; April 22, 1904) directed the lab.";
        let clauses = split_clauses(text, 0);
        assert_eq!(clauses.len(), 1);
    }

    #[test]
    fn clauses_split_on_semicolons_with_offsets() {
        let text = "He assembled top scientists at Los Alamos; resolved plutonium implosion challenges.";
        let clauses = split_clauses(text, 10);
        assert_eq!(clauses.len(), 2);
        assert!(clauses[0].0.starts_with("He assembled"));
        assert!(clauses[1].0.starts_with("resolved"));
        // Fact-relative offsets index back into the original text.
        assert_eq!(clauses[0].1, 10);
        assert_eq!(&text[clauses[1].1 - 10..clauses[1].2 - 10], clauses[1].0);
    }

    #[test]
    fn sentences_without_semicolons_come_back_whole() {
        let text = "No semicolons here.";
        let clauses = split_clauses(text, 0);
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].0, text);
    }
}
