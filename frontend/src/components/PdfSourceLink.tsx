import { useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { apiFetch, fetchAsBlobUrl } from '@/api/client'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'

import { PdfKbContext } from './pdfContext'

interface Props { kb?: string; docName: string; page: number; children: ReactNode }
interface ResolvedPdf { hash: string; doc_name: string; page: number; pages: number }

/** Resolve before enabling a link. Unavailable originals remain honest plain text. */
export default function PdfSourceLink({ kb, ...props }: Props) {
  const contextKb = useContext(PdfKbContext)
  const activeKb = kb ?? contextKb
  if (!activeKb || !Number.isSafeInteger(props.page) || props.page < 1) return <span>{props.children}</span>
  // Remount across KB/document/page changes: an old KB's blob must never survive a switch.
  return <PdfLink key={`${activeKb}:${props.docName}:${props.page}`} kb={activeKb} {...props} />
}

function PdfLink({ kb, docName, page, children }: Props & { kb: string }) {
  const { t } = useTranslation('common')
  const [resolved, setResolved] = useState<ResolvedPdf | null>(null)
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState<string | null>(null)
  const [error, setError] = useState(false)
  const [loading, setLoading] = useState(false)
  const alive = useRef(true)
  const blob = useRef<string | null>(null)
  useEffect(() => {
    alive.current = true
    let cancelled = false
    apiFetch<ResolvedPdf>('/api/v1/document/pdf/resolve', { body: { kb, doc_name: docName, page } })
      .then((value) => { if (!cancelled) setResolved(value) })
      .catch(() => { /* No original: keep the literal citation, not a fabricated link. */ })
    return () => {
      cancelled = true
      alive.current = false
      if (blob.current) URL.revokeObjectURL(blob.current)
    }
  }, [kb, docName, page])

  const show = async () => {
    if (!resolved) return
    setOpen(true)
    if (url || loading) return
    setLoading(true)
    setError(false)
    try {
      const params = new URLSearchParams({ kb, hash: resolved.hash, page: String(page) })
      // Authorization stays in a header. The iframe/anchor sees only a local blob.
      const next = await fetchAsBlobUrl(`/api/v1/document/pdf?${params}`)
      if (!alive.current) { URL.revokeObjectURL(next); return }
      blob.current = next
      setUrl(`${next}#page=${page}`)
    } catch {
      if (alive.current) setError(true)
    } finally {
      if (alive.current) setLoading(false)
    }
  }
  if (!resolved) return <span>{children}</span>
  return <>
    <button type="button" className="text-accent-brand hover:underline text-left"
      title={t('pdf.open', { page })} onClick={() => void show()}>{children}</button>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-[90vw] h-[90vh] flex flex-col">
        <DialogHeader><DialogTitle>{docName} · {t('pdf.physicalPage', { page })}</DialogTitle></DialogHeader>
        {loading && <p role="status">{t('loading')}</p>}
        {error && <p role="alert">{t('pdf.unavailable')}</p>}
        {url && <>
          <a href={url} target="_blank" rel="noopener noreferrer" className="text-accent-brand underline">{t('pdf.open', { page })}</a>
          <iframe title={t('pdf.physicalPage', { page })} src={url} className="w-full flex-1 min-h-0 border-0" />
        </>}
      </DialogContent>
    </Dialog>
  </>
}
