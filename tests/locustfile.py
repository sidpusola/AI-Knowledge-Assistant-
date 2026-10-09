"""
Load testing with Locust.

    PURE_PYTHON=1 locust -f tests/locustfile.py --host http://127.0.0.1:8000

PURE_PYTHON=1 is not optional on this machine. Windows Smart App Control
is in enforcement mode and blocks gevent's compiled _gevent_cgreenlet
extension, so locust fails to import without it. The flag makes gevent
use its pure-Python implementations instead - slower, irrelevant at this
scale, and the alternative (disabling Smart App Control) is irreversible.

Then open http://localhost:8089 and pick a user count.

What this is actually measuring
-------------------------------
Not throughput in the usual sense. One local Ollama serves roughly one
request per model at a time, so concurrent users queue rather than run in
parallel. The interesting number is how latency degrades as they pile up,
and whether anything breaks or merely slows down.

Expect p95 to climb close to linearly with user count. If it does, the
bottleneck is the model, not the app - which is worth knowing before
optimising anything else.

Compare against monitoring.py's /stats while a run is in progress: that
reports server-side retrieval and LLM time separately, so you can see how
much of a slow request was queueing versus work.
"""

import random

from locust import HttpUser, between, events, task

# Mixed difficulty on purpose - short ones route to qwen2.5:3b and long
# ones to qwen2.5:7b, so a run exercises both and the model-switching
# behaviour shows up under load.
QUESTIONS = [
    "what is a data warehouse",
    "what is semantic HTML",
    "what is ETL",
    "what are the assessment prerequisites",
    "harmony in the family",
    "HTML table tags",
    "explain in detail how a data warehouse differs from an operational database system",
    "why should I use semantic HTML elements instead of generic div containers",
    "what do I need to prepare before my placement assessment",
]

# Should be rejected by the relevance gate, so they skip retrieval work
# and show the cheaper path.
OUT_OF_SCOPE = [
    "what is the capital of France",
    "who won the world cup",
    "how do I bake sourdough bread",
]


class StudentUser(HttpUser):
    """One student using the assistant.

    wait_time models reading the answer before asking again. Without it,
    Locust hammers the moment a response lands, which measures the queue
    rather than anything resembling use.
    """

    wait_time = between(2, 6)

    def on_start(self):
        # Each user gets its own conversation, as a real student would.
        self.session_id = f"load-{random.randint(1, 10**9)}"

    @task(10)
    def ask_question(self):
        """The main path. Weighted heaviest because it's what students do."""
        self.client.post(
            "/ask",
            json={"question": random.choice(QUESTIONS), "session_id": self.session_id},
            name="/ask",
            timeout=120,
        )

    @task(4)
    def ask_streaming(self):
        """The endpoint the UI actually uses.

        stream=True so the response is consumed as it arrives - without it
        Locust buffers the whole thing and the timing says nothing about
        time-to-first-token.
        """
        with self.client.post(
            "/ask/stream",
            json={"question": random.choice(QUESTIONS), "session_id": self.session_id},
            name="/ask/stream",
            stream=True,
            timeout=120,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")
                return
            for _ in response.iter_lines():
                pass
            response.success()

    @task(2)
    def ask_out_of_scope(self):
        """Rejected by the relevance gate - no retrieved context."""
        self.client.post(
            "/ask",
            json={"question": random.choice(OUT_OF_SCOPE), "session_id": self.session_id},
            name="/ask (out of scope)",
            timeout=120,
        )

    @task(1)
    def browse_knowledge_base(self):
        """Cheap read. Included so a run isn't entirely LLM-bound - it
        shows whether the API stays responsive while the model is busy."""
        self.client.get("/documents", name="/documents")

    @task(1)
    def check_stats(self):
        self.client.get("/stats", name="/stats")


@events.test_start.add_listener
def warm_up(environment, **kwargs):
    """One request before the clock starts.

    A cold process loads the embedding model, the cross-encoder and an
    Ollama model on its first request - measured at ~35s, which otherwise
    lands in the results as a single catastrophic outlier and skews every
    percentile in a short run.
    """
    import requests

    print("\nwarming up (first request loads embeddings, reranker and the LLM)...")
    try:
        requests.post(
            f"{environment.host}/ask",
            json={"question": "warmup", "session_id": "warmup"},
            timeout=180,
        )
        print("warm.\n")
    except Exception as e:
        print(f"warm-up failed, results will include cold-start cost: {e}\n")


@events.test_stop.add_listener
def report_server_side_timings(environment, **kwargs):
    """Pull /stats at the end, so the run finishes with the server's own
    view next to Locust's.

    They measure different things: Locust sees total wall-clock including
    queueing, /stats sees only time actually spent working. A large gap
    between them is queueing, which is the thing worth finding.
    """
    import requests

    try:
        # environment.host, not environment.client - Environment has no
        # client of its own, only the per-user ones.
        response = requests.get(f"{environment.host}/stats", timeout=10)
        latency = response.json().get("latency_seconds", {})
    except Exception as e:
        print(f"\n(could not read /stats: {e})")
        return

    print("\n--- server-side, from monitoring.py ---")
    for key in ("avg_total", "p95_total", "avg_retrieval", "avg_llm", "avg_time_to_first_token"):
        if key in latency:
            print(f"  {key:26} {latency[key]:.3f}s")
    print("  (Locust's numbers include queue wait; these don't)")
