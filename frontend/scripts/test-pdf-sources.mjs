// Offline executable regressions; no server or inference required.
import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { pathToFileURL } from 'node:url'
import { mkdir } from 'node:fs/promises'

await mkdir('node_modules/.cache', { recursive: true })
await build({
  stdin: { contents: `export * from './src/api/pdfSources'; export { toolCallSource, stepsFromTrace } from './src/api/chat';`, resolveDir: process.cwd() },
  plugins: [{ name: 'offline-i18n', setup(build) {
    build.onResolve({ filter: /^@\/lib\/i18n$/ }, () => ({ path: 'i18n', namespace: 'offline' }))
    build.onLoad({ filter: /.*/, namespace: 'offline' }, () => ({ contents: 'export default {t: (key) => key}' }))
  }}],
  bundle: true, platform: 'node', format: 'esm', outfile: 'node_modules/.cache/pdf-tests.mjs',
})
const { physicalPages, citationTarget, toolCallSource, stepsFromTrace } = await import(pathToFileURL(`${process.cwd()}/node_modules/.cache/pdf-tests.mjs`))
assert.deepEqual(physicalPages('11'), [11])
assert.deepEqual(physicalPages('2-4,11'), [2, 3, 4, 11])
assert.deepEqual(physicalPages([2, 11]), [2, 11])
for (const value of [undefined, null, '', 'all', '0', '-1', '1.5', '3-1', '1-9999', true, [1, '2']]) {
  assert.deepEqual(physicalPages(value), [], `must not invent a physical page: ${JSON.stringify(value)}`)
}
assert.deepEqual(citationTarget('[source anthropic-source-control p. 11]'), { docName: 'anthropic-source-control', page: 11 })
assert.equal(citationTarget('[source missing]'), null)
assert.equal(citationTarget('[source foo p. 0]'), null)
for (const name of ['get_page_content', 'read_document']) {
  const call = { name, arguments: JSON.stringify({ doc_name: 'original', pages: '11' }) }
  assert.deepEqual(toolCallSource(call), { kind: 'doc', label: 'original', docName: 'original', pages: [11] })
  assert.deepEqual(stepsFromTrace([{ kind: 'tool', ...call }])[0].source.pages, [11])
}
assert.equal(toolCallSource({ name: 'write_file', arguments: '{}' }), null)
for (const arguments_ of ['null', '[]', '42', '"text"', '{broken']) {
  assert.equal(toolCallSource({ name: 'read_document', arguments: arguments_ }), null)
}
console.log('PDF source parser and live/restored provenance regressions passed')
