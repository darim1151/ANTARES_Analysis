/** FNV-1a 32-bit over UTF-16 code units; used only for detail shard routing. */
export function fnv1a32(value: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

export function shardOf(id: string, shardCount: number): number {
  return fnv1a32(id) % shardCount;
}

export function shardPath(template: string, shard: number): string {
  return template.replace("{shard}", String(shard).padStart(2, "0"));
}
