# Portfolio-grounded Liquid answers

Every Jekyll build emits `/assets/data/portfolio-search.json` from published posts,
publications, teaching materials and talks. Set `search_exclude: true` to omit an
item. Drafts, private databases, visitor conversations and voice recordings are
excluded. For a PDF or upload, publish a reviewed text companion with its source
link in a supported collection. Arbitrary uploads are not automatically ingested.

The browser chunks text and ranks it with BM25, passing up to three passages into
the existing fine-tuned Liquid model. English words and Korean bigrams provide
lexical matching, not cross-language semantic search. Searches stay local.

Free-form model answers first perform a host-directed search. The model can emit
`<search_portfolio>query</search_portfolio>` for one additional read-only search;
the host executes it and returns passages for model synthesis. This is a bounded
text protocol, not native JSON tool-call training. Existing privacy, entity and
verified-profile routes remain in place. Retrieved source links are appended by
the host, not invented by the model.

The prompt separates explicit beliefs from interpretation and treats passages as
data, not instructions. This is not a guarantee against hallucinations or prompt
injection: curated public ingestion remains necessary. Sources identify retrieved
material, not verified entailment of every generated claim.

Run `node --test scripts/test_portfolio_search.mjs` and build Jekyll. Before merging,
evaluate the actual quantized checkpoint on held-out philosophy questions,
follow-ups, no-evidence questions, malicious quotations and bilingual paraphrases.
Measure recall@3, unsupported claims, abstention and citation relevance. New SFT
weights, automatic PDF ingestion and native tool-call training are not included.
