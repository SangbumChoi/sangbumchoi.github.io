import test from 'node:test';
import assert from 'node:assert/strict';
import {PortfolioSearch, tokens} from '../assets/js/portfolio-search.mjs';
const index = new PortfolioSearch([
  {title:'Open source philosophy', url:'/writing/open-source/', text:'Sharing knowledge helps other developers build and learn. Open source expands access.'},
  {title:'Mobile inference', url:'/writing/mobile/', text:'Quantization reduces memory consumption on mobile devices.'},
  {title:'Private', url:'https://example.com', text:'secret'}
]);
test('retrieves relevant passages with original sources', () => {
  assert.equal(index.search('open source philosophy')[0].url, '/writing/open-source/');
  assert.equal(index.search('mobile quantization')[0].url, '/writing/mobile/');
});
test('unknown queries do not fabricate evidence', () => assert.deepEqual(index.search('astronaut'), []));
test('rejects external URLs and handles empty corpus', () => {
  assert.deepEqual(index.search('secret'), []);
  assert.deepEqual(new PortfolioSearch([]).search('test'), []);
});
test('supports Korean lexical matching', () => assert.ok(tokens('가치관을').includes('가치')));
