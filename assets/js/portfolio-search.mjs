const stop = new Set('a an the is are what why how does do his he she it of to in and about daniel sangbum choi'.split(' '));
export function tokens(text) {
  const words = String(text).normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  return words.flatMap(word => /^[가-힣]{3,}$/.test(word)
    ? [word, ...Array.from({length: word.length - 1}, (_, i) => word.slice(i, i + 2))]
    : stop.has(word) ? [] : [word]);
}

export class PortfolioSearch {
  constructor(documents) {
    this.chunks = documents.filter(d => typeof d.url === 'string' && /^\/(?!\/)/.test(d.url)).flatMap(d => {
      const text = String(d.text || '').slice(0, 100000);
      const chunks = [];
      for (let start = 0; start < text.length; start += 700) {
        const excerpt = text.slice(start, start + 900);
        const terms = tokens(d.title + ' ' + excerpt);
        chunks.push({title: d.title, url: d.url, text: excerpt, terms, counts: new Map()});
      }
      return chunks;
    });
    this.df = new Map();
    this.average = this.chunks.reduce((n, c) => n + c.terms.length, 0) / (this.chunks.length || 1) || 1;
    for (const chunk of this.chunks) {
      for (const term of chunk.terms) chunk.counts.set(term, (chunk.counts.get(term) || 0) + 1);
      for (const term of chunk.counts.keys()) this.df.set(term, (this.df.get(term) || 0) + 1);
    }
  }
  search(query, limit = 3) {
    const terms = [...new Set(tokens(query))].slice(0, 64);
    return this.chunks.map(c => {
      const score = terms.reduce((sum, term) => {
        const frequency = c.counts.get(term) || 0;
        const idf = Math.log(1 + (this.chunks.length - (this.df.get(term) || 0) + 0.5) / ((this.df.get(term) || 0) + 0.5));
        return sum + idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * c.terms.length / this.average));
      }, 0);
      return {title: c.title, url: c.url, text: c.text, score};
    }).filter(c => c.score > 0).sort((a, b) => b.score - a.score).slice(0, Math.min(3, Math.max(1, limit)));
  }
}

let index;
export async function searchPortfolio(query) {
  if (!index) {
    const response = await fetch('/assets/data/portfolio-search.json', {cache: 'no-cache'});
    if (!response.ok) throw new Error('Portfolio index unavailable');
    index = new PortfolioSearch(await response.json());
  }
  return index.search(String(query).slice(0, 500));
}
