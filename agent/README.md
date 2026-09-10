# agent/

One folder per context agent (design: `docs/AGENTS_PLAN.md`). Each folder holds a
`SOURCES.md` listing the datasets, catalogs, papers and docs the agent is grounded on
(retrieval corpus, exemplar libraries, lookup tables, tool docs). Code for each agent
lands in the same folder.

- `agent.literature/` — natural-context brief per scene (runs first)
- `agent.feature/`    — what is this thing, locally (per flagged chip)
- `agent.spatial/`    — where it sits and how it relates to surroundings (per flagged chip)
