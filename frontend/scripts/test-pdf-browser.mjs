// Run real React interactions in local Chromium, entirely offline.
import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'

const dir = await mkdtemp(join(tmpdir(), 'openkb-pdf-ui-'))
try {
  const result = await build({
    entryPoints: ['scripts/pdf-source-browser-fixture.tsx'], bundle: true, write: false,
    format: 'esm', jsx: 'automatic', loader: { '.css': 'empty' },
    alias: { '@': `${process.cwd()}/src` },
  })
  const html = join(dir, 'test.html')
  await writeFile(html, `<div id="root"></div><script type="module">${result.outputFiles[0].text.replaceAll('</script', '<\\/script')}</script>`)
  const child = spawnSync(process.env.CHROMIUM || '/usr/bin/chromium', [
    '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking',
    '--no-first-run', `--user-data-dir=${join(dir, 'profile')}`, '--dump-dom',
    '--virtual-time-budget=5000', `file://${html}`,
  ], { encoding: 'utf8', timeout: 30000, maxBuffer: 10 * 1024 * 1024 })
  assert.equal(child.status, 0, child.stderr)
  assert.match(child.stdout, /data-result="PASS"/, child.stdout.slice(-6000))
  console.log('PASS: real Chromium PDF link, wiki citation, missing source, bearer header and #page=11 interactions')
} finally {
  await rm(dir, { recursive: true, force: true })
}
