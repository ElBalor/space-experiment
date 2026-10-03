"""
Memory Lattice: Long-term memory system for Ultron
Implements episodic, semantic, and meta-memory with vector database.
"""
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    id: str
    content: str
    embedding: Optional[torch.Tensor] = None
    timestamp: str = ""
    memory_type: str = "episodic"
    domain: str = "general"
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: str = ""
    related_ids: List[str] = None
    metadata: Dict[str, Any] = None
    tags: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp == "":
            self.timestamp = datetime.now().isoformat()
        if self.related_ids is None:
            self.related_ids = []
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = {}


class MemoryLattice:
    def __init__(self, embedding_dim: int = 256, max_episodic: int = 10000,
                 max_semantic: int = 50000, max_meta: int = 1000,
                 similarity_threshold: float = 0.7, device: str = "cpu"):
        self.embedding_dim = embedding_dim
        self.max_episodic = max_episodic
        self.max_semantic = max_semantic
        self.max_meta = max_meta
        self.similarity_threshold = similarity_threshold
        self.device = torch.device(device)
        self.episodic_memories: Dict[str, MemoryEntry] = {}
        self.semantic_memories: Dict[str, MemoryEntry] = {}
        self.meta_memories: Dict[str, MemoryEntry] = {}
        self.episodic_ids: List[str] = []
        self.semantic_ids: List[str] = []
        self.meta_ids: List[str] = []
        self.episodic_vectors: List[torch.Tensor] = []
        self.semantic_vectors: List[torch.Tensor] = []
        self.meta_vectors: List[torch.Tensor] = []
        self.embedding_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        ).to(self.device)
        self.domain_index: Dict[str, List[str]] = defaultdict(list)
        self.stats = {"total_episodic": 0, "total_semantic": 0, "total_meta": 0,
                      "total_queries": 0, "total_updates": 0}
        self._next_id = 0

    def _generate_id(self, memory_type: str) -> str:
        self._next_id += 1
        return f"{memory_type}_{self._next_id}_{datetime.now().timestamp()}"

    def _normalize_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(emb, p=2, dim=-1)

    def _compute_embedding(self, content: str, context_embedding=None) -> torch.Tensor:
        content_hash = hash(content) % (2**31)
        gen = torch.Generator()
        gen.manual_seed(content_hash)
        base_emb = torch.randn(self.embedding_dim, generator=gen) * 0.05
        base_emb[0] = float(content_hash) / (2**31)
        if next(self.embedding_projector.parameters()).device != self.device:
            self.embedding_projector = self.embedding_projector.to(self.device)
        if context_embedding is not None:
            context_embedding = context_embedding.to(self.device)
            base_emb = base_emb.to(self.device) + 0.5 * context_embedding
        else:
            base_emb = base_emb.to(self.device)
        with torch.no_grad():
            emb = self.embedding_projector(base_emb.unsqueeze(0)).squeeze(0)
        return self._normalize_embedding(emb)

    def _record_domain(self, domain, memory_id):
        self.domain_index[domain].append(memory_id)

    def _remove_domain(self, domain, memory_id):
        if domain in self.domain_index:
            self.domain_index[domain] = [m for m in self.domain_index[domain] if m != memory_id]

    def _evict_episodic(self):
        if not self.episodic_memories: return
        oldest = min(self.episodic_memories.values(), key=lambda m: m.last_accessed or m.timestamp)
        del self.episodic_memories[oldest.id]
        self.episodic_ids = [i for i in self.episodic_ids if i != oldest.id]
        self.episodic_vectors = [self.episodic_memories[mid].embedding for mid in self.episodic_ids]
        self._remove_domain(oldest.domain, oldest.id)

    def _evict_semantic(self):
        if not self.semantic_memories: return
        lowest = min(self.semantic_memories.values(), key=lambda m: m.confidence)
        del self.semantic_memories[lowest.id]
        self.semantic_ids = [i for i in self.semantic_ids if i != lowest.id]
        self.semantic_vectors = [self.semantic_memories[mid].embedding for mid in self.semantic_ids]

    def _evict_meta(self):
        if not self.meta_memories: return
        oldest = min(self.meta_memories.values(), key=lambda m: m.timestamp)
        del self.meta_memories[oldest.id]
        self.meta_ids = [i for i in self.meta_ids if i != oldest.id]

    def store_episodic(self, content, context="", domain="general",
                       context_embedding=None, embedding_override=None, tags=None) -> str:
        if len(self.episodic_memories) >= self.max_episodic: self._evict_episodic()
        memory_id = self._generate_id("episodic")
        embedding = self._normalize_embedding(embedding_override.to(self.device)) if embedding_override is not None else self._compute_embedding(content, context_embedding)
        entry = MemoryEntry(id=memory_id, content=content, embedding=embedding,
                            memory_type="episodic", domain=domain,
                            metadata={"context": context}, tags=tags or {})
        self.episodic_memories[memory_id] = entry
        self.episodic_ids.append(memory_id)
        self.episodic_vectors.append(embedding)
        self._record_domain(domain, memory_id)
        self.stats["total_episodic"] += 1
        return memory_id

    def store_semantic(self, content, domain="general", confidence=1.0,
                       related_ids=None, context_embedding=None, embedding_override=None, tags=None) -> str:
        if len(self.semantic_memories) >= self.max_semantic: self._evict_semantic()
        memory_id = self._generate_id("semantic")
        embedding = self._normalize_embedding(embedding_override.to(self.device)) if embedding_override is not None else self._compute_embedding(content, context_embedding)
        entry = MemoryEntry(id=memory_id, content=content, embedding=embedding,
                            memory_type="semantic", domain=domain, confidence=confidence,
                            related_ids=related_ids or [], tags=tags or {})
        self.semantic_memories[memory_id] = entry
        self.semantic_ids.append(memory_id)
        self.semantic_vectors.append(embedding)
        self._record_domain(domain, memory_id)
        self.stats["total_semantic"] += 1
        return memory_id

    def store_meta(self, content, improvement_type="general",
                   context_embedding=None, embedding_override=None, tags=None) -> str:
        if len(self.meta_memories) >= self.max_meta: self._evict_meta()
        memory_id = self._generate_id("meta")
        embedding = self._normalize_embedding(embedding_override.to(self.device)) if embedding_override is not None else self._compute_embedding(content, context_embedding)
        entry = MemoryEntry(id=memory_id, content=content, embedding=embedding,
                            memory_type="meta", domain="self_model",
                            metadata={"improvement_type": improvement_type}, tags=tags or {})
        self.meta_memories[memory_id] = entry
        self.meta_ids.append(memory_id)
        self.meta_vectors.append(embedding)
        self.stats["total_meta"] += 1
        return memory_id

    def retrieve(self, query, memory_type=None, domain=None, top_k=5,
                 query_embedding=None, prefer_reward=True):
        self.stats["total_queries"] += 1
        if query_embedding is None:
            query_embedding = self._compute_embedding(query)
        else:
            query_embedding = self._normalize_embedding(query_embedding)
        query_embedding = query_embedding.to(self.device)
        results = []

        def _score(entry, sim):
            reward = entry.tags.get("reward", 0.0) if entry.tags else 0.0
            return sim + (0.05 * reward if prefer_reward else 0.0)

        for mem_type, ids, vectors, store in [
            ("episodic", self.episodic_ids, self.episodic_vectors, self.episodic_memories),
            ("semantic", self.semantic_ids, self.semantic_vectors, self.semantic_memories),
            ("meta",     self.meta_ids,     self.meta_vectors,     self.meta_memories),
        ]:
            if memory_type and memory_type != mem_type: continue
            if not vectors: continue
            tensor = torch.stack(vectors).to(self.device).squeeze()
            if tensor.dim() == 1: tensor = tensor.unsqueeze(0)
            sims = F.cosine_similarity(query_embedding.view(1, -1), tensor, dim=-1)
            for mid, sim in zip(ids, sims):
                if domain and store[mid].domain != domain: continue
                if sim.item() >= self.similarity_threshold:
                    entry = store[mid]
                    entry.access_count += 1
                    entry.last_accessed = datetime.now().isoformat()
                    results.append((entry, _score(entry, sim.item())))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def update_confidence(self, memory_id, new_confidence):
        if memory_id in self.semantic_memories:
            self.semantic_memories[memory_id].confidence = max(0.0, min(1.0, new_confidence))
            self.stats["total_updates"] += 1

    def link_memories(self, memory_id1, memory_id2):
        all_memories = {**self.episodic_memories, **self.semantic_memories, **self.meta_memories}
        if memory_id1 in all_memories and memory_id2 in all_memories:
            if memory_id2 not in all_memories[memory_id1].related_ids:
                all_memories[memory_id1].related_ids.append(memory_id2)
            if memory_id1 not in all_memories[memory_id2].related_ids:
                all_memories[memory_id2].related_ids.append(memory_id1)

    def get_stats(self):
        return {**self.stats, "episodic_count": len(self.episodic_memories),
                "semantic_count": len(self.semantic_memories),
                "meta_count": len(self.meta_memories),
                "domains": list(self.domain_index.keys())}

    def sample_recent(self, memory_type="episodic", limit=50):
        store = {"episodic": self.episodic_memories, "semantic": self.semantic_memories,
                 "meta": self.meta_memories}.get(memory_type, {})
        values = sorted(store.values(), key=lambda m: m.last_accessed or m.timestamp, reverse=True)
        return values[:limit]

    def sample_goals(self, limit=3):
        metas = [m for m in self.meta_memories.values()
                 if m.tags and m.tags.get("type") in {"goal", "rule"}]
        metas.sort(key=lambda m: m.tags.get("priority", 0.5), reverse=True)
        return metas[:limit]

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "episodic": {k: asdict(v) for k, v in self.episodic_memories.items()},
            "semantic": {k: asdict(v) for k, v in self.semantic_memories.items()},
            "meta":     {k: asdict(v) for k, v in self.meta_memories.items()},
            "domain_index": dict(self.domain_index),
            "stats": self.stats, "next_id": self._next_id
        }
        for mem_type in ["episodic", "semantic", "meta"]:
            for entry_dict in data[mem_type].values():
                if entry_dict.get("embedding") is not None and isinstance(entry_dict["embedding"], torch.Tensor):
                    entry_dict["embedding"] = entry_dict["embedding"].tolist()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        for mem_type, ids, vectors, store in [
            ("episodic", self.episodic_ids, self.episodic_vectors, self.episodic_memories),
            ("semantic", self.semantic_ids, self.semantic_vectors, self.semantic_memories),
            ("meta",     self.meta_ids,     self.meta_vectors,     self.meta_memories),
        ]:
            for mem_id, entry_dict in data.get(mem_type, {}).items():
                if entry_dict.get("embedding"):
                    entry_dict["embedding"] = torch.tensor(entry_dict["embedding"]).to(self.device)
                store[mem_id] = MemoryEntry(**entry_dict)
                ids.append(mem_id)
                vectors.append(store[mem_id].embedding.to(self.device))
        self.domain_index = defaultdict(list, data.get("domain_index", {}))
        self.embedding_projector.to(self.device)
        self.stats = data.get("stats", self.stats)
        self._next_id = data.get("next_id", 0)
