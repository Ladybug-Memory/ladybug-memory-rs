---
name: lb-graph-curation
description: >
  Dedup entity nodes and fill missing historical RelatesTo edges in an
  LadybugDB graph.

  Use when the entity graph created by a GLiNER 2.5 model has duplicate/canonical-split nodes,
  garbage entities, coOccurs-only memories, or wrong-direction typed edges.
---

# LB graph curation (dedup + missing edges)

For shell + schema reference see the **[ladybug-skill](https://github.com/LadybugDB/ladybug-skill)** —
this skill does not duplicate it. Method below assumes that skill's
`lbug` shell and Cypher dialect.

## 0. Binary + backup (do first, every time)

- Backup before writing: `cp <db> <db>.pre-dedup.bak` (+ timestamped `/tmp` copy).
- Token discipline: Use lbug --agents for machine parsable output

## 1. Survey (3 queries, that's all you need)

```cypher
MATCH (e:Entity) RETURN e.id, e.canonical_name, e.entity_type ORDER BY e.id;
MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) WHERE r.relation_type <> 'coOccurs'
  RETURN a.id, a.canonical_name, b.id, b.canonical_name, r.relation_type, r.memory_id ORDER BY r.memory_id;
MATCH (e:Entity)-[r:MentionedIn]->(m:Memory) RETURN e.id, e.canonical_name, m.id ORDER BY m.id, e.id;
```

Also: `MATCH (m:Memory) RETURN m.id, m.content ORDER BY m.id;` — memory
content is the ground truth for what edges *should* exist.

## 2. Dedup nodes

Dup signatures: short-name split (`Oppenheimer` vs `J. Robert Oppenheimer`,
`Julius` vs `Julius Oppenheimer`), adjective/country (`American` vs
`United States`), part/whole (`Berkeley` vs `University of California`),
garbage parses (`c.` from "c. 1836"), doubled `MentionedIn` rows.

Procedure per dup set (canonical C, dup D):
1. Extend `C.aliases` with D's name variants
   (`SET e.aliases = ['Canonical', 'Short']` — list literal works for JSON).
2. Copy `MentionedIn` D→C **unless C already links that memory** (e.g. Berkeley→UC: skip, overlap).
3. Copy `RelatesTo` both directions with the dup-guard (never create parallel
   same-`relation_type`+`memory_id` edges, never self-loops), remapping
   endpoints that are themselves dups (e.g. `26→62` becomes `1→46`):
```cypher
MATCH (dup:Entity)-[r:RelatesTo]->(b:Entity) WHERE dup.id = $D AND NOT b.id IN [$SKIP]
MATCH (can:Entity) WHERE can.id = $C
OPTIONAL MATCH (can)-[x:RelatesTo]->(b)
  WHERE x.relation_type = r.relation_type AND x.memory_id = r.memory_id
WITH dup, r, b, can, x WHERE x IS NULL AND b.id <> $C
CREATE (can)-[r2:RelatesTo]->(b)
SET r2.memory_id = r.memory_id, r2.relation_type = r.relation_type,
    r2.confidence = r.confidence, r2.valid_from = r.valid_from,
    r2.valid_to = r.valid_to, r2.tense = r.tense,
    r2.viewpoint = r.viewpoint, r2.viewpoint_confidence = r.viewpoint_confidence
RETURN count(*);
```
(mirror with `(a)-[r]->(dup)` / `(a)-[r2]->(can)` for incoming; skip the copy when the source is also a dup being remapped.)
4. Deliberately **skip** migrating known-wrong typed edges (they die with D in step 6); migrate `coOccurs` + correct typed edges only.
5. Dedup doubled `MentionedIn`: `MATCH ... WITH r SKIP 1 DELETE r RETURN count(*);`
6. `MATCH (e:Entity) WHERE e.id = $D DETACH DELETE e RETURN count(*);`

## 3. Missing historical edges

Every memory whose entities are `coOccurs`-only (or unlinked) gets typed
edges grounded in its content **and** real-world history. Standard set per
memory + the open-schema relation names used before
(`educated at`, `student of`, `employer`, `director`, `participant in`,
`creator`, `field of work`, `place of birth`, `child of`, `spouse`,
`sibling`, `member of`, `founder`, `opposed`, `award received`,
`grandchild of`):

Template (idempotent — guard skips existing):
```cypher
MATCH (a:Entity) WHERE a.id = $SRC MATCH (b:Entity) WHERE b.id = $DST
OPTIONAL MATCH (a)-[x:RelatesTo]->(b)
  WHERE x.relation_type = '$REL' AND x.memory_id = $MEM
WITH a, b, x WHERE x IS NULL
CREATE (a)-[r2:RelatesTo]->(b)
SET r2.memory_id = $MEM, r2.relation_type = '$REL', r2.confidence = 0.9,
    r2.valid_from = '$VF', r2.valid_to = '$VT', r2.tense = 'past',
    r2.viewpoint = 'fact', r2.viewpoint_confidence = 0.95
RETURN count(*);
```

## 4. Delete wrong edges (conservative — backwards/nonsense only)

Patterns seen: org→event `director/organizer/creator` (lab didn't direct WWII),
backwards `director` (IAS→person), person→org `organizer/creator` for orgs they
didn't found, `student of` parent (replace with `child of`), 4–5 parallel
edges person→org (keep `member of`), test→lab `director`, framework→graduation
`participant in`. Delete by `(src, dst, memory_id, relation_type IN [...])`
and `DELETE r RETURN count(*);` — check the count matches expectation.

## 5. Verify (one batch)

- `MATCH (e:Entity) WHERE e.id IN [$D...] RETURN count(*);` → 0
- `MATCH (e:Entity) RETURN count(*);` → old count minus deleted nodes
- Typed-edge dump (step-1 query 2) — eyeball every row against history
- Spot-check merged memories' `MentionedIn` membership
