//! Rule-based fact categorization (no LLM on the write path).
//!
//! Keyword cues over the fact text; first match wins in priority order.
//! Anything without a cue is `General`.

use crate::types::Category;

const CUES: &[(Category, &[&str])] = &[
    (Category::Preferences, &["prefer", "prefers", "preferred", "like", "likes", "liked", "love", "loves", "loved", "hate", "hates", "hated", "dislike", "dislikes", "favour", "favor", "favourite", "favorite", "always use", "never use", "go-to", "goto", "always", "never", "have to", "has to", "make sure", "be sure to"]),
    (Category::Identity, &["i am", "i'm", "my name is", "i live in", "i work as", "background in", "years old"]),
    (Category::Goals, &["goal is", "trying to", "want to", "plan to", "aim to", "objective", "target is", "aspire"]),
    (Category::Decisions, &["decided", "decide", "decision", "chose", "chosen", "choose", "picked", "pick", "opted for", "went with", "trade-off", "tradeoff", "rationale", "instead of"]),
    (Category::Relationships, &["works with", "works for", "married to", "partner of", "team", "reports to", "manager is", "colleague", "founded by", "ceo of", "member of"]),
    (Category::Projects, &["project", "initiative", "milestone", "launching", "building", "working on", "roadmap", "sprint"]),
    (Category::Routines, &["every day", "every morning", "weekly", "daily", "routine", "habit", "standup", "recurring", "each friday", "each monday"]),
    (Category::Lessons, &["learned", "learn", "lesson", "mistake", "avoid", "don't forget", "remember to", "gotcha", "pitfall", "broke when", "fails when"]),
    (Category::Technical, &["version", "config", "install", "command", "api", "database", "server", "deploy", "error", "stack trace", "crate", "package", "repo", "endpoint", "port ", "uses ", "runs on"]),
    (Category::Work, &["role", "job", "company", "employer", "client", "deadline", "meeting", "okr", "review", "promotion", "salary"]),
];

pub fn classify(text: &str) -> Category {
    let lower = text.to_lowercase();
    // Word-boundary-ish match: pad with spaces so "uses " doesn't hit "excuses".
    let padded = format!(" {lower} ");
    for (category, cues) in CUES {
        for cue in *cues {
            if cue.contains(' ') {
                if padded.contains(cue) {
                    return *category;
                }
            } else if padded.split(|c: char| !c.is_alphanumeric()).any(|w| w == *cue) {
                return *category;
            }
        }
    }
    Category::General
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cue_classification() {
        assert_eq!(classify("User prefers Python for scripting"), Category::Preferences);
        assert_eq!(classify("We decided to use LadybugDB instead of sqlite"), Category::Decisions);
        assert_eq!(classify("Tim Cook is the CEO of Apple"), Category::Relationships);
        assert_eq!(classify("The sky is blue today"), Category::General);
    }
}
