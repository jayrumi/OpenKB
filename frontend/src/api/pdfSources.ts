/** Physical 1-based PDF pages only. Unknown/malformed/unbounded specs stay inert. */
export function physicalPages(value: unknown): number[] {
  if (Array.isArray(value)) {
    return value.length <= 32 && value.every((p) => Number.isSafeInteger(p) && p > 0)
      ? [...new Set(value as number[])] : []
  }
  if (typeof value === 'number') return Number.isSafeInteger(value) && value > 0 ? [value] : []
  if (typeof value !== 'string' || !value.trim()) return []
  const pages: number[] = []
  for (const part of value.split(',')) {
    const match = /^\s*([1-9]\d*)(?:\s*-\s*([1-9]\d*))?\s*$/.exec(part)
    if (!match) return []
    const start = Number(match[1]), end = Number(match[2] ?? match[1])
    if (!Number.isSafeInteger(end) || end < start || end - start > 31) return []
    for (let page = start; page <= end; page++) {
      if (!pages.includes(page)) pages.push(page)
      if (pages.length > 32) return []
    }
  }
  return pages
}

export function citationTarget(token: string): { docName: string; page: number } | null {
  const match = /^\[source ([^\]\r\n]+?) p\. ([1-9]\d*)\]$/.exec(token)
  if (!match || !Number.isSafeInteger(Number(match[2]))) return null
  return { docName: match[1], page: Number(match[2]) }
}
