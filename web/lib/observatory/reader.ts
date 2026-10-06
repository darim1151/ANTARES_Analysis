/**
 * GET-only, immutable bundle reader.
 *
 * Today the bundle is static JSON served next to the app. A later adapter may
 * serve the same payloads from Parquet/Arrow/HATS products or an immutable
 * read API; it only has to implement {@link BundleReader}.
 *
 * Every payload is verified against the manifest's sha256 before it is parsed.
 * A digest mismatch fails closed: the workspace never renders an unverifiable
 * payload. When WebCrypto is unavailable (non-secure context) payloads load as
 * UNVERIFIED and the workspace says so.
 */

import {
  DOMAIN_IDS,
  OBSERVATORY_BUNDLE_CONTRACT,
  type BundleManifest,
  type DomainId,
  type EntityDetail,
  type EntityDetailShard,
  type FileIntegrity,
  type NativeEntityRef,
  type ObservatoryBundle
} from "../../types/observatory.ts";
import { shardOf, shardPath } from "./kernel/shard.ts";

export class BundleError extends Error {
  constructor(
    message: string,
    readonly path: string | null = null
  ) {
    super(message);
    this.name = "BundleError";
  }
}

export interface BundleReader {
  loadBundle(signal?: AbortSignal): Promise<ObservatoryBundle>;
  loadEntityDetail(bundle: ObservatoryBundle, ref: NativeEntityRef): Promise<{ detail: EntityDetail | null; integrity: FileIntegrity }>;
}

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function digest(bytes: ArrayBuffer): Promise<string | null> {
  const subtle = typeof globalThis.crypto !== "undefined" ? globalThis.crypto.subtle : undefined;
  if (!subtle) return null;
  return hex(await subtle.digest("SHA-256", bytes));
}

export class StaticBundleReader implements BundleReader {
  private shardCache = new Map<string, Promise<{ shard: EntityDetailShard; integrity: FileIntegrity }>>();

  constructor(readonly base: string) {}

  private async fetchBytes(rel: string, signal?: AbortSignal): Promise<ArrayBuffer> {
    let response: Response;
    try {
      response = await fetch(`${this.base}/${rel}`, { signal, cache: "force-cache" });
    } catch (error) {
      if ((error as Error).name === "AbortError") throw error;
      throw new BundleError(`Network error while reading ${rel}.`, rel);
    }
    if (!response.ok) throw new BundleError(`${rel} returned HTTP ${response.status}.`, rel);
    return response.arrayBuffer();
  }

  private async verified<T>(
    manifest: BundleManifest,
    rel: string,
    signal?: AbortSignal
  ): Promise<{ value: T; integrity: FileIntegrity }> {
    const entry = manifest.files.find((f) => f.path === rel);
    if (!entry) throw new BundleError(`${rel} is not listed in the bundle manifest.`, rel);
    const bytes = await this.fetchBytes(rel, signal);
    const actual = await digest(bytes);
    if (actual !== null && actual !== entry.sha256) {
      throw new BundleError(`Integrity check failed for ${rel}: sha256 does not match the manifest.`, rel);
    }
    const value = JSON.parse(new TextDecoder().decode(bytes)) as T & { contract?: string };
    if (value.contract !== OBSERVATORY_BUNDLE_CONTRACT) throw new BundleError(`${rel} is not an Observatory bundle payload.`, rel);
    return { value, integrity: { path: rel, expected: entry.sha256, actual, status: actual === null ? "UNVERIFIED" : "VERIFIED" } };
  }

  async loadBundle(signal?: AbortSignal): Promise<ObservatoryBundle> {
    const manifestBytes = await this.fetchBytes("manifest.json", signal);
    const manifestSha256 = await digest(manifestBytes);
    const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as BundleManifest;
    if (manifest.contract !== OBSERVATORY_BUNDLE_CONTRACT || !/^1\./.test(manifest.contract_version)) {
      throw new BundleError(`Unsupported bundle contract ${manifest.contract} ${manifest.contract_version}.`, "manifest.json");
    }
    const integrity: FileIntegrity[] = [];
    const take = async <T,>(rel: string) => {
      const { value, integrity: entry } = await this.verified<T>(manifest, rel, signal);
      integrity.push(entry);
      return value;
    };
    const [basis, capabilities, provenance] = await Promise.all([
      take<ObservatoryBundle["basis"]>(manifest.basis),
      take<ObservatoryBundle["capabilities"]>(manifest.capabilities),
      take<ObservatoryBundle["provenance"]>(manifest.provenance)
    ]);
    const domainEntries = await Promise.all(
      DOMAIN_IDS.map(async (domain) => {
        const refs = manifest.domains[domain];
        if (!refs) throw new BundleError(`Bundle does not provide the ${domain} domain.`, "manifest.json");
        const [time, sky, entities, features] = await Promise.all([
          take<ObservatoryBundle["domains"][DomainId]["time"]>(refs.time),
          take<ObservatoryBundle["domains"][DomainId]["sky"]>(refs.sky),
          take<ObservatoryBundle["domains"][DomainId]["entities"]>(refs.entities),
          take<ObservatoryBundle["domains"][DomainId]["features"]>(refs.features)
        ]);
        for (const payload of [time, sky, entities, features]) {
          if (payload.domain !== domain || payload.build_id !== basis.domains[domain]?.build_id) {
            throw new BundleError(`${domain} payloads do not match the pinned ${domain} build.`);
          }
        }
        return [domain, { time, sky, entities, features }] as const;
      })
    );
    if (capabilities.basis_id !== basis.basis_id || provenance.basis_id !== basis.basis_id) {
      throw new BundleError("Capabilities or provenance were produced for a different basis.");
    }
    return {
      base: this.base,
      manifest,
      manifestSha256,
      basis,
      capabilities,
      provenance,
      domains: Object.fromEntries(domainEntries) as ObservatoryBundle["domains"],
      integrity: {
        method: manifestSha256 === null ? "unavailable" : "sha256",
        files: integrity.sort((a, b) => a.path.localeCompare(b.path))
      }
    };
  }

  async loadEntityDetail(bundle: ObservatoryBundle, ref: NativeEntityRef) {
    const entities = bundle.domains[ref.domain].entities;
    const shard = shardOf(ref.id, entities.detail.shard_count);
    const rel = `domains/${ref.domain}/${shardPath(entities.detail.path_template, shard)}`;
    let pending = this.shardCache.get(rel);
    if (!pending) {
      pending = this.verified<EntityDetailShard>(bundle.manifest, rel).then(({ value, integrity }) => {
        if (value.domain !== ref.domain || value.shard !== shard || value.build_id !== bundle.basis.domains[ref.domain].build_id) {
          throw new BundleError(`${rel} does not belong to the pinned ${ref.domain} build.`, rel);
        }
        return { shard: value, integrity };
      });
      pending.catch(() => this.shardCache.delete(rel));
      this.shardCache.set(rel, pending);
    }
    const { shard: doc, integrity } = await pending;
    const detail = doc.records[ref.id] ?? null;
    if (detail && detail.kind !== ref.kind) throw new BundleError(`${ref.id} is not a ${ref.kind}.`, rel);
    return { detail, integrity };
  }
}
