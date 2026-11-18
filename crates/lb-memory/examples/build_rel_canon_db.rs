//! Build the schema database (`rel_canon.sqlite`): DBpedia/YAGO-grounded
//! page-type domains with per-domain extraction schemas.
//!
//! ```sh
//! cargo run --release -p lb-memory --example build_rel_canon_db -- src/rel_canon.sqlite
//! ```
//!
//! (needs `lbm fetch-model` once for the model snapshot + ORT lib)
//!
//! Domain set (64): `base` plus 63 types grounded in DBpedia instance
//! counts (Person 1.59M, Place 967K, Work 552K, Organisation 318K,
//! Species 190K), the YAGO 4.5 upper taxonomy, and Wikipedia infobox
//! transclusion counts (film 164K, book 58K, video game 29K, sports
//! medals 444K, WikiProject Biography 2.97M). Each domain carries a
//! description (embedded for nearest-domain lookup), entity-type extras
//! beyond the base labels, and relation questions whose text IS the
//! canonical edge label (verified against the Wikidata property export
//! at seed time), with the natural phrasing the scorer reads.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};

/// (domain, description)
#[rustfmt::skip]
const DOMAINS: &[(&str, &str)] = &[
("base", "General. Everyday facts about people, places, organizations and things that fit no specific category."),
// People (WikiProject Biography: 2.97M tagged pages)
("biography", "Biography. Life stories of individual people: birth, family, education, careers, relationships and achievements."),
("politician", "Politician. Elected officials and political figures: offices held, parties, elections and terms."),
("athlete", "Athlete. Sportspeople and their careers: teams, events, victories and awards."),
("actor", "Actor. Film, television and stage performers: roles, productions and awards."),
("musician", "Musician. Singers, composers and instrumentalists: works, labels and performances."),
("writer", "Writer. Authors and journalists of books and texts: works, publishers and awards."),
("scientist", "Scientist. Researchers and academics: fields, education, mentors and discoveries."),
("artist", "Artist. Visual artists: works, movements and recognition."),
("filmmaker", "Filmmaker. Directors, producers and screenwriters: films and productions."),
("businessperson", "Businessperson. Founders, executives and owners: companies led, founded or owned."),
("journalist", "Journalist. Reporters and correspondents: outlets, beats and stories."),
("soldier", "Soldier. Military personnel: service, branches, conflicts and decorations."),
("saint", "Saint. Venerated religious figures: lives, traditions and recognition."),
("monarch", "Monarch. Kings, queens and royalty: reigns, family and succession."),
// Places (DBpedia Place: 967K)
("country", "Country. Nations and states: capitals, continents, borders and regions."),
("city", "City. Cities, towns and settlements: countries, regions and landmarks."),
("island", "Island. Islands and archipelagos: location, countries and geography."),
("mountain", "Mountain. Peaks and ranges: ranges, countries and elevation context."),
("river", "River. Rivers and waterways: tributaries, countries and basins."),
("lake", "Lake. Lakes and inland waters: location and geography."),
("park", "Park. National parks and protected areas: location and features."),
("road", "Road. Highways and routes: countries, termini and networks."),
("bridge", "Bridge. Bridges and crossings: what they span and where they stand."),
("building", "Building. Structures and architecture: architects, styles and locations."),
("airport", "Airport. Airports and airfields: airlines served and locations."),
("station", "Station. Railway and transit stations: networks and places served."),
// Organizations (DBpedia Organisation: 318K)
("company", "Company. Businesses and corporations: founders, owners, leaders and headquarters."),
("university", "University. Schools, colleges and universities: people, affiliations and places."),
("sports_team", "Sports team. Clubs and franchises: leagues, players and victories."),
("party", "Party. Political parties: members, elections and platforms."),
("agency", "Agency. Government agencies and bodies: parents, functions and jurisdictions."),
("nonprofit", "Nonprofit. Charities and foundations: founders, causes and operators."),
("band", "Band. Music groups: members, labels, genres and works."),
("airline", "Airline. Airlines and carriers: alliances, hubs and fleets."),
("hospital", "Hospital. Hospitals and clinics: operators, places and networks."),
("museum", "Museum. Museums, galleries and libraries: collections, places and operators."),
// Creative works (DBpedia Work: 552K; film 164K, book 58K, video game 29K transclusions)
("film", "Film. Movies and cinema: directors, cast, producers, genres and settings."),
("book", "Book. Books and novels: authors, publishers, genres and characters."),
("album", "Album. Music albums: artists, labels, genres and songs."),
("song", "Song. Songs and singles: performers, composers, albums and labels."),
("tv_series", "TV series. Television shows: creators, cast, networks and settings."),
("video_game", "Video game. Games: developers, publishers, platforms and genres."),
("periodical", "Periodical. Newspapers and magazines: publishers, places and topics."),
("artwork", "Artwork. Paintings and visual works: creators, movements and homes."),
("play", "Play. Stage plays and theatre: writers, performers and productions."),
// Events
("war", "War. Armed conflicts and battles: participants, sides, outcomes and places."),
("election", "Election. Votes and campaigns: candidates, winners, offices and districts."),
("sports_event", "Sports event. Tournaments and games: teams, winners, sports and venues."),
("award", "Award. Prizes and honors: winners, nominees, fields and ceremonies."),
("festival", "Festival. Festivals and celebrations: places, editions and participants."),
("disaster", "Disaster. Natural and man-made disasters: places, dates and impacts."),
("conference", "Conference. Summits and meetings: organizers, places and participants."),
("spaceflight", "Spaceflight. Missions and launches: agencies, craft and outcomes."),
// Things (DBpedia Species: 190K)
("species", "Species. Animals, plants and organisms: taxonomy and habitats."),
("food", "Food. Dishes and cuisines: origins, ingredients and cultures."),
("automobile", "Automobile. Cars and vehicles: makers, origins and models."),
("aircraft", "Aircraft. Planes and aviation: makers, operators and roles."),
("ship", "Ship. Vessels and watercraft: builders, operators and service."),
("software", "Software. Programs and apps: developers, platforms and systems."),
("website", "Website. Sites and online services: owners, operators and topics."),
("language", "Language. Human languages: regions, speakers and families."),
("disease", "Disease. Illnesses and conditions: symptoms, causes and fields."),
("law", "Law. Legislation and legal instruments: jurisdictions, topics and bodies."),
];

/// (domain, label)
#[rustfmt::skip]
const ENTITY_TYPES: &[(&str, &str)] = &[
("biography", "award"), ("biography", "school"), ("biography", "university"),
("politician", "award"),
("athlete", "team"), ("athlete", "award"),
("actor", "award"), ("actor", "character"),
("musician", "award"), ("musician", "song"), ("musician", "album"),
("writer", "award"), ("writer", "book"),
("scientist", "award"), ("scientist", "school"), ("scientist", "university"),
("artist", "award"), ("artist", "artwork"),
("filmmaker", "award"), ("filmmaker", "film"),
("soldier", "award"),
("university", "school"), ("university", "university"),
("sports_team", "team"),
("band", "album"), ("band", "song"),
("airline", "aircraft"),
("museum", "artwork"),
("film", "award"), ("film", "character"),
("book", "award"), ("book", "character"),
("album", "award"), ("album", "song"),
("song", "award"), ("song", "album"),
("tv_series", "award"), ("tv_series", "character"),
("video_game", "award"), ("video_game", "character"),
("artwork", "award"),
("play", "award"), ("play", "character"),
("sports_event", "team"), ("sports_event", "award"),
("spaceflight", "aircraft"),
];

/// (domain, property_id, question, description, priority).
///
/// Person-headed questions (the subject is a person) carry an explicit
/// head-type constraint so non-person pairs never score them: a Trinity
/// test was never "educated at" a laboratory, and Harvard was never
/// "born in" New York. Everything else stays unconstrained.
#[rustfmt::skip]
const PERSON_SUBJECT: &[&str] = &[
    "educated at", "student of", "spouse", "sibling", "relative",
    "place of birth", "occupation", "field of work", "residence",
    "award received", "father", "mother", "child", "doctoral advisor",
    "doctoral student", "author", "notable work", "country of citizenship",
    "work location", "member of political party", "position held",
    "drafted by", "cast member", "performer", "composer", "instrument",
    "record label", "sport", "member of sports team", "nominated for",
    "electoral district",
];

#[rustfmt::skip]
/// Tail-type constraints by question: the value must be an entity of one
/// of these types. Educational institutions are the sharp case: GlINER
/// reliably types universities as `university` (Harvard 0.95) while labs
/// stay `organization`, so `educated at`/`student of` restricted to
/// school-ish tails cannot misfire on laboratories.
const QUESTION_TAILS: &[(&str, &[&str])] = &[
    ("educated at", &["school", "university"]),
    ("student of", &["person", "school", "university"]),
    ("place of birth", &["location"]),
    ("spouse", &["person"]),
    ("sibling", &["person"]),
    ("relative", &["person"]),
    ("father", &["person"]),
    ("mother", &["person"]),
    ("child", &["person"]),
    ("doctoral advisor", &["person"]),
    ("doctoral student", &["person"]),
    ("residence", &["location"]),
    ("work location", &["location"]),
    ("award received", &["concept", "organization", "product", "event"]),
    ("country of citizenship", &["location"]),
    ("member of political party", &["organization"]),
    ("position held", &["organization", "concept"]),
    ("drafted by", &["organization"]),
    ("electoral district", &["location"]),
    ("author", &["person", "organization"]),
    ("notable work", &["product", "concept", "event"]),
];

const RELATIONS: &[(&str, u32, &str, &str, i64)] = &[
// Base battery: always asked.
("base", 463, "member of", "X is a member of Y", 1),
("base", 1344, "participant in", "X participated in Y", 2),
("base", 361, "part of", "X is part of Y", 3),
("base", 170, "creator", "X created Y", 4),
("base", 57, "director", "X directed or led Y", 5),
("base", 108, "employer", "X works for Y", 6),
("base", 664, "organizer", "X organized Y", 7),
("base", 31, "instance of", "X is an instance of Y", 8),
// Biography.
("biography", 69, "educated at", "X graduated from or was educated at Y", 10),
("biography", 1066, "student of", "X studied under Y", 11),
("biography", 19, "place of birth", "X was born in Y", 12),
("biography", 106, "occupation", "X works as Y", 13),
("biography", 101, "field of work", "X works in the field of Y", 14),
("biography", 26, "spouse", "X is married to Y", 15),
("biography", 3373, "sibling", "X is the sibling of Y", 16),
("biography", 1038, "relative", "X is related to Y", 17),
("biography", 551, "residence", "X lives in Y", 18),
("biography", 166, "award received", "X received the award Y", 19),
("biography", 22, "father", "X's father is Y", 20),
("biography", 25, "mother", "X's mother is Y", 21),
("biography", 40, "child", "X is the child of Y", 22),
("biography", 50, "author", "X authored Y", 23),
("biography", 800, "notable work", "X is known for Y", 24),
("biography", 27, "country of citizenship", "X is a citizen of Y", 25),
("biography", 937, "work location", "X works in Y", 26),
// People specializations.
("politician", 39, "position held", "X held the position of Y", 30),
("politician", 102, "member of political party", "X is a member of the Y party", 31),
("politician", 768, "electoral district", "X represents the Y district", 32),
("athlete", 54, "member of sports team", "X plays for Y", 30),
("athlete", 641, "sport", "X competes in Y", 31),
("athlete", 647, "drafted by", "X was drafted by Y", 32),
("actor", 161, "cast member", "X is a cast member of Y", 30),
("actor", 175, "performer", "X performed in Y", 31),
("musician", 175, "performer", "X performed Y", 30),
("musician", 86, "composer", "X composed Y", 31),
("musician", 264, "record label", "X is signed to the Y label", 32),
("musician", 1303, "instrument", "X plays Y", 33),
("writer", 50, "author", "X authored Y", 30),
("writer", 123, "publisher", "X is published by Y", 31),
("scientist", 101, "field of work", "X works in the field of Y", 30),
("scientist", 184, "doctoral advisor", "X's doctoral advisor was Y", 31),
("scientist", 185, "doctoral student", "X's doctoral student was Y", 32),
("artist", 170, "creator", "X created Y", 30),
("artist", 800, "notable work", "X is known for Y", 31),
("filmmaker", 57, "director", "X directed Y", 30),
("filmmaker", 162, "producer", "X produced Y", 31),
("filmmaker", 58, "screenwriter", "X wrote Y", 32),
("businessperson", 1830, "owner of", "X owns Y", 30),
("businessperson", 137, "operator", "X operates Y", 31),
("businessperson", 488, "chairperson", "X chairs Y", 32),
("businessperson", 169, "chief executive officer", "X is the CEO of Y", 33),
("journalist", 50, "author", "X authored Y", 30),
("soldier", 241, "military branch", "X served in the Y", 30),
("soldier", 945, "allegiance", "X pledged allegiance to Y", 31),
// Places.
("country", 36, "capital", "X's capital is Y", 30),
("country", 30, "continent", "X is in Y", 31),
("country", 47, "shares border with", "X borders Y", 32),
("city", 17, "country", "X is in Y", 30),
("island", 17, "country", "X belongs to Y", 30),
("island", 30, "continent", "X is in Y", 31),
("mountain", 4552, "mountain range", "X is in the Y range", 30),
("mountain", 17, "country", "X is in Y", 31),
("river", 974, "tributary", "X is a tributary of Y", 30),
("river", 17, "country", "X flows through Y", 31),
("lake", 17, "country", "X is in Y", 30),
("park", 17, "country", "X is in Y", 30),
("road", 17, "country", "X is in Y", 30),
("bridge", 177, "crosses", "X crosses Y", 30),
("building", 84, "architect", "X was designed by Y", 30),
("building", 149, "architectural style", "X is in the Y style", 31),
("airport", 113, "airline hub", "X is a hub for Y", 30),
("station", 137, "operator", "X is operated by Y", 30),
// Organizations.
("company", 749, "parent organization", "X's parent organization is Y", 30),
("company", 127, "owned by", "X is owned by Y", 31),
("company", 169, "chief executive officer", "X's CEO is Y", 32),
("company", 488, "chairperson", "X's chair is Y", 33),
("company", 159, "headquarters location", "X is headquartered in Y", 34),
("company", 112, "founded by", "X was founded by Y", 35),
("university", 108, "employer", "X employs Y", 30),
("university", 1416, "affiliation", "X is affiliated with Y", 31),
("sports_team", 54, "member of sports team", "X plays for Y", 30),
("sports_team", 1346, "winner", "X won Y", 31),
("party", 102, "member of political party", "X is a member of the Y party", 30),
("party", 1344, "participant in", "X participated in Y", 31),
("agency", 749, "parent organization", "X's parent organization is Y", 30),
("nonprofit", 112, "founded by", "X was founded by Y", 30),
("band", 264, "record label", "X is signed to the Y label", 30),
("band", 136, "genre", "X plays Y music", 31),
("band", 175, "performer", "X performed Y", 32),
("airline", 114, "airline alliance", "X is in the Y alliance", 30),
("hospital", 137, "operator", "X is operated by Y", 30),
("museum", 800, "notable work", "X is known for Y", 30),
// Creative works.
("film", 57, "director", "X was directed by Y", 30),
("film", 162, "producer", "X was produced by Y", 31),
("film", 58, "screenwriter", "X was written by Y", 32),
("film", 161, "cast member", "X stars Y", 33),
("film", 750, "distributed by", "X was distributed by Y", 34),
("film", 136, "genre", "X is a Y work", 35),
("film", 840, "narrative location", "X is set in Y", 36),
("film", 915, "filming location", "X was filmed in Y", 37),
("book", 50, "author", "X was written by Y", 30),
("book", 123, "publisher", "X was published by Y", 31),
("book", 136, "genre", "X is a Y work", 32),
("album", 264, "record label", "X was released on the Y label", 30),
("album", 136, "genre", "X is Y music", 31),
("album", 175, "performer", "X was performed by Y", 32),
("song", 175, "performer", "X was performed by Y", 30),
("song", 86, "composer", "X was composed by Y", 31),
("song", 264, "record label", "X was released on the Y label", 32),
("song", 136, "genre", "X is Y music", 33),
("tv_series", 170, "creator", "X was created by Y", 30),
("tv_series", 161, "cast member", "X stars Y", 31),
("tv_series", 136, "genre", "X is a Y show", 32),
("video_game", 178, "developer", "X was developed by Y", 30),
("video_game", 123, "publisher", "X was published by Y", 31),
("video_game", 136, "genre", "X is a Y game", 32),
("video_game", 400, "platform", "X runs on Y", 33),
("periodical", 123, "publisher", "X is published by Y", 30),
("artwork", 170, "creator", "X was created by Y", 30),
("play", 50, "author", "X was written by Y", 30),
// Events.
("war", 1344, "participant in", "X fought in Y", 30),
("war", 1346, "winner", "X won Y", 31),
("war", 276, "location", "X happened in Y", 32),
("election", 1344, "participant in", "X ran in Y", 30),
("election", 1346, "winner", "X won Y", 31),
("sports_event", 1344, "participant in", "X competed in Y", 30),
("sports_event", 1346, "winner", "X won Y", 31),
("sports_event", 276, "location", "X happened in Y", 32),
("award", 1346, "winner", "X won Y", 30),
("award", 1411, "nominated for", "X was nominated for Y", 31),
("festival", 276, "location", "X happened in Y", 30),
("disaster", 276, "location", "X happened in Y", 30),
("conference", 664, "organizer", "X was organized by Y", 30),
("conference", 276, "location", "X happened in Y", 31),
("spaceflight", 137, "operator", "X was operated by Y", 30),
// Things.
("species", 31, "instance of", "X is a Y", 30),
("food", 495, "country of origin", "X comes from Y", 30),
("automobile", 176, "manufacturer", "X was made by Y", 30),
("aircraft", 176, "manufacturer", "X was made by Y", 30),
("ship", 176, "manufacturer", "X was built by Y", 30),
("software", 178, "developer", "X was developed by Y", 30),
("software", 306, "operating system", "X runs on Y", 31),
("software", 400, "platform", "X runs on Y", 32),
("website", 1830, "owner of", "X owns Y", 30),
("language", 17, "country", "X is spoken in Y", 30),
];

// Exemplar sentences per domain (2-3 each, varied register).
// Same distribution as queries (full sentences), so nearest-domain
// cosine separates better than abstract descriptions alone.

#[rustfmt::skip]
pub const EXEMPLARS: &[(&str, &[&str])] = &[
("base", &[
    "User prefers Python for scripting.",
    "The meeting is at noon tomorrow in the main conference room.",
    "Water boils at 100 degrees Celsius at sea level.",
]),
("biography", &[
    "Marie Curie was born in Warsaw and won the Nobel Prize in Physics for her research on radioactivity.",
    "Leonardo da Vinci painted the Mona Lisa while living in Florence under the patronage of Francis I.",
    "Nelson Mandela studied law in Johannesburg before leading the movement that ended apartheid.",
]),
("politician", &[
    "Abraham Lincoln was elected president in 1860 and led the country through the Civil War.",
    "The senator from Ohio chairs the finance committee and faces reelection next year.",
    "Winston Churchill served as prime minister during World War II and won the Nobel Prize in Literature.",
]),
("athlete", &[
    "Serena Williams won 23 Grand Slam singles titles and was ranked world number one.",
    "Michael Jordan was drafted by the Chicago Bulls and led them to six championships.",
    "Usain Bolt set the 100 meter world record at the Berlin championships.",
]),
("actor", &[
    "Meryl Streep starred in The Devil Wears Prada and has won three Academy Awards.",
    "Tom Hanks played the lead role in Forrest Gump, directed by Robert Zemeckis.",
    "Cate Blanchett performed on stage in Sydney before moving into film.",
]),
("musician", &[
    "Taylor Swift writes her own songs and is signed to Republic Records.",
    "Miles Davis recorded Kind of Blue with his sextet in 1959.",
    "Yo-Yo Ma plays the cello and has performed with orchestras worldwide.",
]),
("writer", &[
    "Jane Austen authored Pride and Prejudice, published in 1813.",
    "Stephen King has published dozens of horror novels through Viking Press.",
    "Toni Morrison won the Nobel Prize in Literature for Beloved.",
]),
("scientist", &[
    "Albert Einstein earned his doctorate in Zurich and developed the theory of relativity.",
    "Rosalind Franklin studied chemistry at Cambridge and contributed to the discovery of DNA.",
    "Richard Feynman worked at Caltech and won the Nobel Prize for quantum electrodynamics.",
]),
("artist", &[
    "Pablo Picasso created Guernica in response to the Spanish Civil War.",
    "Frida Kahlo painted self-portraits while living in Mexico City.",
    "Banksy is an anonymous street artist known for stencil works in London.",
]),
("filmmaker", &[
    "Christopher Nolan directed Inception, starring Leonardo DiCaprio.",
    "Akira Kurosawa wrote and directed Seven Samurai, released in 1954.",
    "Ava DuVernay produced the documentary 13th for Netflix.",
]),
("businessperson", &[
    "Elon Musk founded SpaceX and serves as CEO of Tesla.",
    "Oprah Winfrey built a media empire and owns the OWN network.",
    "Warren Buffett chairs Berkshire Hathaway from Omaha, Nebraska.",
]),
("journalist", &[
    "Christiane Amanpour reports for CNN from conflict zones around the world.",
    "Bob Woodward authored All the President's Men about the Watergate scandal.",
    "Ida B. Wells investigated lynching in the American South.",
]),
("soldier", &[
    "Audie Murphy served in the infantry during World War II and received the Medal of Honor.",
    "Joan of Arc led French forces at Orleans during the Hundred Years War.",
    "Dwight Eisenhower commanded Allied forces before becoming president.",
]),
("saint", &[
    "Francis of Assisi founded the Franciscan order and lived in poverty in Italy.",
    "Mother Teresa worked with the poor in Calcutta and was canonized in 2016.",
    "Thomas Aquinas wrote the Summa Theologica while teaching in Paris.",
]),
("monarch", &[
    "Elizabeth II reigned over the United Kingdom for seventy years.",
    "Louis XIV ruled France from Versailles in an absolute monarchy.",
    "Cleopatra was the last active ruler of Ptolemaic Egypt.",
]),
("country", &[
    "Japan is an island nation in East Asia with Tokyo as its capital.",
    "Brazil borders ten countries and its capital Brasilia was founded in 1960.",
    "France shares borders with Spain, Italy and Germany in Western Europe.",
]),
("city", &[
    "Paris is the capital of France, located on the Seine river.",
    "New York City hosted the World's Fair in Flushing Meadows.",
    "Kyoto was the imperial capital of Japan for over a thousand years.",
]),
("island", &[
    "Iceland is an island nation in the North Atlantic with Reykjavik as its capital.",
    "Manhattan is the most densely populated borough of New York City.",
    "Easter Island belongs to Chile and is famous for its stone statues.",
]),
("mountain", &[
    "Mount Everest lies in the Himalayas on the border of Nepal and Tibet.",
    "Kilimanjaro is the highest mountain in Africa, located in Tanzania.",
    "The Matterhorn straddles the border between Switzerland and Italy.",
]),
("river", &[
    "The Nile flows through eleven countries before reaching the Mediterranean Sea.",
    "The Missouri river is a tributary of the Mississippi.",
    "The Danube passes through Vienna on its way to the Black Sea.",
]),
("lake", &[
    "Lake Superior is the largest of the Great Lakes by surface area.",
    "The Caspian Sea borders five countries in Central Asia.",
    "Lake Titicaca sits high in the Andes between Peru and Bolivia.",
]),
("park", &[
    "Yellowstone was established as the first national park in the United States.",
    "The Serengeti in Tanzania is famous for its annual wildebeest migration.",
    "Central Park was designed by Olmsted and Vaux in Manhattan.",
]),
("road", &[
    "Route 66 runs from Chicago to Santa Monica across eight states.",
    "The Appian Way connected Rome to the port of Brindisi.",
    "The Pan-American Highway spans the Americas from Alaska to Argentina.",
]),
("bridge", &[
    "The Golden Gate Bridge crosses the strait connecting San Francisco Bay to the Pacific.",
    "Tower Bridge crosses the Thames in London and was completed in 1894.",
    "The Millau Viaduct carries traffic across the Tarn valley in France.",
]),
("building", &[
    "The Empire State Building was designed by Shreve, Lamb and Harmon in New York.",
    "The Sydney Opera House was completed in 1973 with its distinctive shell roofs.",
    "The Burj Khalifa in Dubai is the tallest building in the world.",
]),
("airport", &[
    "Heathrow serves as the main hub for British Airways in London.",
    "Changi Airport in Singapore was designed for seamless transfers.",
    "Denver International Airport is operated by the city of Denver.",
]),
("station", &[
    "Grand Central Terminal serves Metro-North trains in New York City.",
    "Shinjuku Station in Tokyo handles millions of passengers daily.",
    "Gare du Nord in Paris connects France to London and Brussels.",
]),
("company", &[
    "Apple was founded by Steve Jobs and is headquartered in Cupertino, California.",
    "Toyota is operated by the Toyoda family and manufactures cars in over 30 countries.",
    "Siemens was founded in Berlin and operates factories worldwide.",
]),
("university", &[
    "Harvard employs thousands of faculty across its schools in Cambridge.",
    "Oxford is affiliated with 39 colleges and educated 30 British prime ministers.",
    "Stanford was founded by Leland Stanford in California.",
]),
("sports_team", &[
    "The Yankees play baseball in the Bronx and have won 27 World Series.",
    "Real Madrid signed several star players and competes in La Liga.",
    "The All Blacks represent New Zealand in international rugby.",
]),
("party", &[
    "The Labour Party won the British general election under Tony Blair.",
    "Democrats hold the Senate majority after the midterm elections.",
    "The Green Party campaigns on environmental issues across Europe.",
]),
("agency", &[
    "NASA was established as the civilian space agency of the United States.",
    "The EPA enforces environmental regulations from Washington.",
    "MI6 handles foreign intelligence for the British government.",
]),
("nonprofit", &[
    "The Red Cross was founded to provide wartime medical aid.",
    "Doctors Without Borders operates clinics in conflict zones.",
    "The Gates Foundation funds global health programs from Seattle.",
]),
("band", &[
    "The Beatles were signed to Parlophone and recorded at Abbey Road.",
    "Radiohead plays alternative rock and released Kid A in 2000.",
    "BTS performs pop music and won multiple American Music Awards.",
]),
("airline", &[
    "Lufthansa is in the Star Alliance and hubs at Frankfurt Airport.",
    "Singapore Airlines operates long-haul routes across Asia and beyond.",
    "Delta maintains a large fleet serving the Americas.",
]),
("hospital", &[
    "Mayo Clinic operates hospitals in Minnesota, Arizona and Florida.",
    "St. Jude treats children regardless of ability to pay.",
    "Charite in Berlin is one of Europe's largest university hospitals.",
]),
("museum", &[
    "The Louvre holds the Mona Lisa among its vast collections in Paris.",
    "MoMA is known for modern art in Manhattan.",
    "The British Museum houses artifacts spanning two million years.",
]),
("film", &[
    "Christopher Nolan directed Inception, starring Leonardo DiCaprio.",
    "The Godfather was produced by Albert Ruddy and distributed by Paramount.",
    "Parasite won the Palme d'Or and is a Korean-language thriller.",
]),
("book", &[
    "George Orwell authored 1984, published by Secker and Warburg.",
    "To Kill a Mockingbird won the Pulitzer Prize for Harper Lee.",
    "Dune is a science fiction novel set on the desert planet Arrakis.",
]),
("album", &[
    "Thriller was performed by Michael Jackson and released on Epic.",
    "Abbey Road was recorded by the Beatles in London studios.",
    "Blue is a folk album by Joni Mitchell released in 1971.",
]),
("song", &[
    "Bohemian Rhapsody was composed by Freddie Mercury and performed by Queen.",
    "Imagine was written and performed by John Lennon in 1971.",
    "Billie Jean was released on the Thriller album by Epic.",
]),
("tv_series", &[
    "Breaking Bad was created by Vince Gilligan and stars Bryan Cranston.",
    "The Crown dramatizes the reign of Elizabeth II across six seasons.",
    "Friends was filmed before a live audience in Los Angeles.",
]),
("video_game", &[
    "Minecraft was developed by Mojang and published by Microsoft.",
    "The Legend of Zelda runs on Nintendo platforms in a fantasy genre.",
    "Tetris was created by Alexey Pajitnov in the Soviet Union.",
]),
("periodical", &[
    "The New York Times is published daily from Manhattan.",
    "Nature publishes peer-reviewed science from London.",
    "Vogue covers fashion and is published by Conde Nast.",
]),
("artwork", &[
    "The Starry Night was created by Vincent van Gogh in 1889.",
    "Guernica depicts the bombing of a Basque town during civil war.",
    "The David was sculpted by Michelangelo from a single marble block.",
]),
("play", &[
    "Hamlet was written by Shakespeare and first performed around 1600.",
    "The Phantom of the Opera was authored by Gaston Leroux as a novel first.",
    "Waiting for Godot premiered in Paris under Beckett's direction.",
]),
("war", &[
    "World War II involved dozens of nations and ended in 1945.",
    "The Hundred Years War was fought between England and France.",
    "Gettysburg was the turning point of the American Civil War.",
]),
("election", &[
    "The 2020 United States presidential election saw record turnout.",
    "Brexit was decided by referendum across the United Kingdom.",
    "Mandela won South Africa's first democratic election in 1994.",
]),
("sports_event", &[
    "The World Cup final took place in Lusail with Argentina winning.",
    "The Olympics were held in Tokyo with thousands of athletes participating.",
    "Wimbledon takes place in London every summer.",
]),
("award", &[
    "The Nobel Prize in Physics was awarded for the discovery of X-rays.",
    "DiCaprio was nominated for the Oscar five times before winning.",
    "The Booker Prize honors the best English-language novel each year.",
]),
("festival", &[
    "Cannes happens each May on the French Riviera.",
    "Burning Man takes place in the Nevada desert with participatory art.",
    "Oktoberfest in Munich celebrates Bavarian culture every autumn.",
]),
("disaster", &[
    "The 2011 earthquake off Japan triggered a devastating tsunami.",
    "Hurricane Katrina struck New Orleans in 2005.",
    "The Titanic sank in the North Atlantic on its maiden voyage.",
]),
("conference", &[
    "The Paris climate conference was organized by the United Nations.",
    "Davos happens annually in Switzerland with world leaders attending.",
    "The Yalta Conference brought Allied leaders together in 1945.",
]),
("spaceflight", &[
    "Apollo 11 was operated by NASA and landed astronauts on the Moon.",
    "The International Space Station is operated by five space agencies.",
    "SpaceX operates reusable rockets from Cape Canaveral.",
]),
("species", &[
    "The giant panda is a bear species native to central China.",
    "Darwin studied finches in the Galapagos Islands.",
    "The dodo was a flightless bird endemic to Mauritius.",
]),
("food", &[
    "Sushi comes from Japan and is made with vinegared rice.",
    "Pizza originated in Naples and is made with wheat dough.",
    "Croissants are made from layered pastry in France.",
]),
("automobile", &[
    "The Model T was made by Ford in Detroit starting in 1908.",
    "Ferrari manufactures sports cars in Maranello, Italy.",
    "The Beetle was originally made by Volkswagen in Germany.",
]),
("aircraft", &[
    "The 747 was made by Boeing as a wide-body airliner.",
    "The Concorde was operated by Air France on supersonic routes.",
    "Airbus manufactures jets in Toulouse and Hamburg.",
]),
("ship", &[
    "The Queen Mary 2 was built for Cunard in France.",
    "Nimitz-class carriers are operated by the United States Navy.",
    "The Cutty Sark was built as a tea clipper in Scotland.",
]),
("software", &[
    "Linux was developed by Linus Torvalds and runs on servers worldwide.",
    "Photoshop runs on Windows and macOS and is published by Adobe.",
    "Python was created by Guido van Rossum for scripting and data work.",
]),
("website", &[
    "Wikipedia is owned by the Wikimedia Foundation and run by volunteers.",
    "YouTube is operated by Google from San Bruno, California.",
    "Reddit hosts communities moderated by volunteers.",
]),
("language", &[
    "Spanish is spoken in Spain and across Latin America.",
    "Mandarin uses characters and is spoken by over a billion people.",
    "Latin developed in ancient Rome and shaped the Romance languages.",
]),
("disease", &[
    "Malaria spreads through mosquitoes in tropical regions.",
    "Diabetes management advanced with the discovery of insulin.",
    "Polio was nearly eradicated through global vaccination campaigns.",
]),
("law", &[
    "The Clean Air Act applies across the United States.",
    "GDPR applies to data processing in the European Union.",
    "The Civil Rights Act was passed by Congress in 1964.",
]),
];

/// (alias, property_id, canonical label): colloquial recall bridge.
const ALIASES: &[(&str, u32, &str)] = &[
    ("brother", 3373, "sibling"),
    ("sister", 3373, "sibling"),
    ("dad", 22, "father"),
    ("mom", 25, "mother"),
    ("mum", 25, "mother"),
    ("wife", 26, "spouse"),
    ("husband", 26, "spouse"),
    ("kid", 40, "child"),
    ("kids", 40, "child"),
    ("son", 40, "child"),
    ("daughter", 40, "child"),
    ("studied", 69, "educated at"),
    ("born", 19, "place of birth"),
    ("married", 26, "spouse"),
    ("directed", 57, "director"),
    ("boss", 108, "employer"),
    ("founded", 112, "founded by"),
];

fn main() -> Result<()> {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| "rel_canon.sqlite".into());
    let old_path = std::env::args().nth(2).unwrap_or_else(|| "rel_canon.v1.sqlite".into());
    anyhow::ensure!(
        DOMAINS.len() == 64,
        "expected 64 domains, have {}",
        DOMAINS.len()
    );
    // Verify every question label against the v1 property inventory.
    let old = Connection::open(&old_path)
        .with_context(|| format!("opening v1 inventory at {old_path}"))?;
    let known: std::collections::HashSet<String> = old
        .prepare("SELECT label FROM properties")?
        .query_map([], |r| r.get(0))?
        .collect::<std::result::Result<_, rusqlite::Error>>()?;
    for (domain, _, q, _, _) in RELATIONS {
        anyhow::ensure!(
            known.contains(*q),
            "question {q:?} (domain {domain}) not in property inventory"
        );
    }
    if std::path::Path::new(&out_path).exists() {
        std::fs::remove_file(&out_path)?;
    }
    let conn = Connection::open(&out_path).with_context(|| format!("opening {out_path}"))?;
    conn.execute_batch(
        "CREATE TABLE properties(id INTEGER PRIMARY KEY, label TEXT NOT NULL);
         CREATE TABLE sk_domains(name TEXT PRIMARY KEY, description TEXT NOT NULL, embedding BLOB NOT NULL);
         CREATE TABLE sk_entity_types(domain TEXT NOT NULL, label TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0);
         CREATE TABLE sk_relations(domain TEXT NOT NULL, property_id INTEGER NOT NULL, question TEXT NOT NULL,
           description TEXT NOT NULL, head_types TEXT NOT NULL DEFAULT '[]', tail_types TEXT NOT NULL DEFAULT '[]',
           priority INTEGER NOT NULL DEFAULT 0);
         CREATE TABLE aliases(alias TEXT PRIMARY KEY, property_id INTEGER NOT NULL, label TEXT NOT NULL);
         CREATE TABLE sk_exemplars(domain TEXT NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL);",
    )?;
    // Carry the vocabulary forward (documents the question labels).
    conn.execute(
        "ATTACH DATABASE ?1 AS v1",
        rusqlite::params![old_path],
    )?;
    conn.execute_batch("INSERT INTO properties SELECT * FROM v1.properties")?;
    conn.execute_batch("DETACH DATABASE v1")?;
    // Domain embeddings via the GLiNER encoder (same space as query embeds).
    let snap = std::env::var("LB_MEMORY_SNAPSHOT").unwrap_or_else(|_| {
        format!(
            "{}/.cache/ladybug-memory/models/gliner2.5-base-v1-nico-onnx",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let mut model = gliner_onnx::GlinerOnnx::load(&snap)?;
    {
        let mut ins = conn.prepare(
            "INSERT INTO sk_domains(name, description, embedding) VALUES (?1, ?2, ?3)",
        )?;
        for (name, desc) in DOMAINS {
            let emb = model.embed(&format!("{name}. {desc}"))?;
            let bytes: Vec<u8> = emb.iter().flat_map(|v| v.to_le_bytes()).collect();
            ins.execute(params![name, desc, bytes])?;
            println!("embedded domain {name}");
        }
    }
    {
        let mut ins = conn.prepare("INSERT INTO sk_exemplars(domain, text, embedding) VALUES (?1, ?2, ?3)")?;
        let mut n = 0;
        for (domain, texts) in EXEMPLARS {
            // Must cover every domain or selection silently ignores it.
            assert!(
                DOMAINS.iter().any(|(d, _)| d == domain),
                "exemplars for unknown domain {domain}"
            );
            for text in *texts {
                let emb = model.embed(text)?;
                let bytes: Vec<u8> = emb.iter().flat_map(|v| v.to_le_bytes()).collect();
                ins.execute(params![domain, text, bytes])?;
                n += 1;
            }
        }
        println!("embedded {n} exemplars");
    }
    {
        let mut ins = conn.prepare(
            "INSERT INTO sk_entity_types(domain, label, priority) VALUES (?1, ?2, ?3)",
        )?;
        for (i, (domain, label)) in ENTITY_TYPES.iter().enumerate() {
            ins.execute(params![domain, label, i as i64])?;
        }
    }
    {
        let mut ins = conn.prepare(
            "INSERT INTO sk_relations(domain, property_id, question, description, head_types, tail_types, priority)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for (domain, pid, q, d, prio) in RELATIONS {
            let heads = if PERSON_SUBJECT.contains(q) { r#"["person"]"# } else { "[]" };
            let tails = QUESTION_TAILS
                .iter()
                .find(|(qq, _)| qq == q)
                .map(|(_, ts)| {
                    let v: Vec<String> = ts.iter().map(|t| t.to_string()).collect();
                    serde_json::to_string(&v).unwrap()
                })
                .unwrap_or_else(|| "[]".to_string());
            ins.execute(params![domain, pid, q, d, heads, tails, prio])?;
        }
    }
    {
        let mut ins =
            conn.prepare("INSERT INTO aliases(alias, property_id, label) VALUES (?1, ?2, ?3)")?;
        for (alias, pid, label) in ALIASES {
            ins.execute(params![alias, pid, label])?;
        }
    }
    conn.execute_batch(&format!(
        "PRAGMA user_version = {};",
        lb_memory::rel_canon::REL_CANON_DB_VERSION
    ))?;
    let (nd, ne, nr, na): (i64, i64, i64, i64) = conn.query_row(
        "SELECT (SELECT COUNT(*) FROM sk_domains), (SELECT COUNT(*) FROM sk_entity_types),
                (SELECT COUNT(*) FROM sk_relations), (SELECT COUNT(*) FROM aliases)",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    )?;
    println!("wrote {out_path}: {nd} domains, {ne} entity types, {nr} questions, {na} aliases");
    Ok(())
}
