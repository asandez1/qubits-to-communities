"""
LLM-driven community member wrapping CommunityMember with Ollama-based decisions.

Replaces rule-based stochastic decisions with LLM inference for:
- generate_requests(): LLM decides whether and what to request
- declare_availability(): LLM decides which services to offer
- decide_bridge_redeem(): LLM decides whether to redeem credits

Requires: ollama server running with phi or mistral model loaded.
Usage: LLMMember(base_member, model="phi", ...) then call same interface.

Note: This module is a proof-of-concept for Task 4 (LLM Agent Phase).
The Ollama server must be running and accessible at localhost:11434.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np

from .member import CommunityMember, ServiceOffer, ServiceRequest

logger = logging.getLogger(__name__)

# Short system prompt for all LLM calls (<200 tokens)
SYSTEM_PROMPT = (
    "You are a community economy member. You have skills, a credit balance, "
    "and preferences. Make practical economic decisions. Respond ONLY with "
    "valid JSON. No explanations."
)


def _build_member_context(member: CommunityMember, prices: Dict[str, float]) -> str:
    """Build a concise context string for the LLM about this member."""
    return json.dumps({
        "archetype": member.archetype.value,
        "skills": member.skills,
        "balance": round(member.credit_balance, 1),
        "reputation": round(member.reputation, 2),
        "total_earned": round(member.total_earned, 1),
        "total_spent": round(member.total_spent, 1),
        "prices": {k: round(v, 1) for k, v in prices.items()},
    })


class LLMMember:
    """Wraps a CommunityMember with LLM-based decision-making via Ollama."""

    def __init__(
        self,
        base_member: CommunityMember,
        model: str = "phi",
        ollama_host: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout: float = 5.0,
    ):
        self.base = base_member
        self.model = model
        self.ollama_host = ollama_host
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy-init Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.ollama_host)
            except ImportError:
                raise ImportError(
                    "ollama package required: pip install ollama"
                )
        return self._client

    def _query_llm(self, prompt: str) -> Optional[dict]:
        """Send a prompt to the LLM and parse JSON response."""
        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": self.temperature},
            )
            content = response["message"]["content"].strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in content:
                start = content.index("```") + 3
                if content[start:].startswith("json"):
                    start += 4
                end = content.index("```", start)
                content = content[start:end].strip()
            return json.loads(content)
        except Exception as e:
            logger.debug(f"LLM query failed for {self.base.member_id}: {e}")
            return None

    def generate_requests(
        self, cycle: int, all_categories: List[str], prices: Dict[str, float]
    ) -> List[ServiceRequest]:
        """LLM decides whether and what to request."""
        if not self.base.active:
            return []

        context = _build_member_context(self.base, prices)
        prompt = (
            f"Context: {context}\n"
            f"Available categories: {all_categories}\n"
            f"Cycle: {cycle}\n\n"
            f"Should this member request a service this cycle? "
            f"Respond with JSON: {{\"request\": true/false, \"category\": \"...\"}}"
        )

        result = self._query_llm(prompt)
        if result and result.get("request") and result.get("category") in all_categories:
            import uuid
            return [ServiceRequest(
                request_id=f"req_{self.base.member_id}_{cycle}_{uuid.uuid4().hex[:6]}",
                requester_id=self.base.member_id,
                category=result["category"],
                cycle=cycle,
            )]

        # Fallback to rule-based
        return self.base.generate_requests(cycle, all_categories)

    def declare_availability(
        self, cycle: int, prices: Dict[str, float]
    ) -> List[ServiceOffer]:
        """LLM decides which services to offer."""
        if not self.base.active:
            return []

        context = _build_member_context(self.base, prices)
        prompt = (
            f"Context: {context}\n"
            f"Cycle: {cycle}\n\n"
            f"Which of your skills should you offer this cycle? "
            f"Consider prices and your time. "
            f"Respond with JSON: {{\"offer\": [\"skill1\", ...]}}"
        )

        result = self._query_llm(prompt)
        offers = []
        if result and isinstance(result.get("offer"), list):
            for skill in result["offer"]:
                if skill in self.base.skills and self.base.current_tasks < self.base.max_task_load:
                    offers.append(ServiceOffer(
                        offer_id=f"off_{self.base.member_id}_{cycle}_{skill}",
                        provider_id=self.base.member_id,
                        category=skill,
                        cycle=cycle,
                        skill_level=self.base.skill_levels.get(skill, 0.5),
                    ))

        if not offers:
            # Fallback to rule-based
            return self.base.declare_availability(cycle, prices)
        return offers

    def decide_bridge_redeem(self, cycle: int, prices: Dict[str, float]) -> Optional[float]:
        """LLM decides whether to redeem credits via fiat bridge."""
        if not self.base.active or self.base.credit_balance < 10:
            return None

        context = _build_member_context(self.base, prices)
        prompt = (
            f"Context: {context}\n"
            f"Bridge rules: progressive discount tiers (0% up to 50 credits, "
            f"10% at 100, 30% at 200, 60% at 500).\n"
            f"Cycle: {cycle}\n\n"
            f"Should this member redeem credits for fiat? If yes, how much? "
            f"Respond with JSON: {{\"redeem\": true/false, \"amount\": N}}"
        )

        result = self._query_llm(prompt)
        if result and result.get("redeem") and isinstance(result.get("amount"), (int, float)):
            amount = float(result["amount"])
            if 0 < amount <= self.base.credit_balance:
                return amount

        # Fallback to rule-based
        return self.base.decide_bridge_redeem(cycle)


def create_llm_members(
    base_members: List[CommunityMember],
    model: str = "phi",
    ollama_host: str = "http://localhost:11434",
) -> List[LLMMember]:
    """Wrap a list of base members with LLM decision-making."""
    return [
        LLMMember(m, model=model, ollama_host=ollama_host)
        for m in base_members
    ]
