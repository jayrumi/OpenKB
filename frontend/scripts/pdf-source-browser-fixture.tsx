import React from 'react'
import { createRoot } from 'react-dom/client'
import PdfSourceLink from '../src/components/PdfSourceLink'
import { PdfKbContext } from '../src/components/pdfContext'
import MarkdownView from '../src/components/MarkdownView'

const requests: { url: string; init?: RequestInit }[] = []
localStorage.setItem('openkb_token', 'fixture-token')
window.fetch = async (input, init) => {
  const url = String(input)
  requests.push({ url, init })
  if (url.endsWith('/resolve')) {
    const body = JSON.parse(String(init?.body))
    return body.doc_name === 'original'
      ? new Response(JSON.stringify({ hash: 'a'.repeat(64), doc_name: 'original', page: body.page, pages: 13 }), { headers: { 'Content-Type': 'application/json' } })
      : new Response('', { status: 404 })
  }
  return new Response('%PDF-1.4\nfixture only', { headers: { 'Content-Type': 'application/pdf' } })
}
const sleep = () => new Promise((resolve) => setTimeout(resolve, 80))
const check = (value: unknown, message: string) => { if (!value) throw new Error(message) }
const root = createRoot(document.getElementById('root')!)
try {
  root.render(<PdfKbContext.Provider value="test">
    <PdfSourceLink docName="original" page={11}>Known p. 11</PdfSourceLink>
    <PdfSourceLink docName="missing" page={11}>Missing p. 11</PdfSourceLink>
    <MarkdownView kb="test" source="**Evidence** [source original p. 11] [source missing p. 11]" />
  </PdfKbContext.Provider>)
  await sleep(); await sleep()
  const buttons = [...document.querySelectorAll('button')]
  const known = buttons.find((b) => b.textContent === 'Known p. 11')
  check(known, 'available physical PDF must be clickable')
  check(!buttons.some((b) => b.textContent?.includes('Missing p. 11')), 'missing source must remain plain text')
  check(buttons.some((b) => b.textContent === '[source original p. 11]'), 'wiki citation must use PDF source component')
  known!.click()
  await sleep(); await sleep()
  const frame = document.querySelector('iframe')
  check(frame?.src.startsWith('blob:') && frame.src.endsWith('#page=11'), 'viewer must open physical page 11 in authorized PDF blob')
  check(requests.every((r) => !r.url.includes('fixture-token')), 'token must not leak into URLs')
  check(requests.every((r) => (r.init?.headers as Record<string, string>)?.Authorization === 'Bearer fixture-token'), 'metadata and original PDF require bearer header')
  check(document.querySelector('a[href$="#page=11"]'), 'viewer must offer a direct PDF page link')
  const revoked: string[] = []
  const revoke = URL.revokeObjectURL.bind(URL)
  URL.revokeObjectURL = (value) => { revoked.push(value); revoke(value) }
  root.render(<PdfKbContext.Provider value="other"><PdfSourceLink docName="original" page={11}>Other KB</PdfSourceLink></PdfKbContext.Provider>)
  await sleep(); await sleep()
  check(!document.querySelector('iframe'), 'KB switch must remove the old PDF viewer')
  check(revoked.length === 1, 'KB switch must revoke the old PDF blob')
  check(requests.some((r) => r.url.endsWith('/resolve') && JSON.parse(String(r.init?.body)).kb === 'other'), 'new KB must resolve independently')
  root.unmount()
  document.body.dataset.result = 'PASS'
} catch (error) {
  document.body.dataset.result = 'FAIL'
  document.body.append(String(error))
}
