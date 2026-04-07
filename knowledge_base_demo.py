#!/usr/bin/env python3
"""
Demo: Seed mock English meeting data and test the RAG knowledge base.
"""

from knowledge_base import KnowledgeBase

# ── Mock Meeting Data ──────────────────────────────────────────

MEETINGS = {
    "sprint-planning-apr1": {
        "title": "Sprint Planning - April 1",
        "turns": [
            {"role": "Alice", "content": "Good morning everyone. Let's plan the sprint for this week. We have three main items on the backlog."},
            {"role": "Bob", "content": "I think the authentication refactor should be our top priority. The current session management is causing random logouts."},
            {"role": "Carol", "content": "Agreed. We got 15 support tickets about that last week. I can take the backend part."},
            {"role": "Alice", "content": "Great. Bob, can you handle the frontend token refresh? And Carol, you'll do the backend session store."},
            {"role": "Bob", "content": "Sure. I estimate about 3 days for the frontend changes. We need to update the interceptor logic and add retry mechanisms."},
            {"role": "Dave", "content": "What about the performance issue on the dashboard? Load times are over 4 seconds."},
            {"role": "Alice", "content": "Good point. Dave, you own the dashboard optimization. Target is under 1.5 seconds. We can use the new caching layer."},
            {"role": "Carol", "content": "The third item is the API rate limiter. We need it before the public beta launch on April 15th."},
            {"role": "Alice", "content": "Right. The beta launch is April 15th. All features must be code-complete by April 12th. Any concerns?"},
            {"role": "Bob", "content": "I think we can make it if we pair on the rate limiter Thursday and Friday."},
            {"role": "Dave", "content": "Sounds good to me. I'll have the dashboard optimization done by Wednesday."},
        ],
    },
    "design-review-mar28": {
        "title": "Design Review - March 28",
        "turns": [
            {"role": "Eve", "content": "Let's review the new onboarding flow designs. The goal is to reduce drop-off from 40% to under 20%."},
            {"role": "Frank", "content": "We simplified it from 5 steps to 3 steps. Step 1 is account creation, step 2 is profile setup, step 3 is first project."},
            {"role": "Alice", "content": "I like the simplification. But do we still collect enough data for personalization?"},
            {"role": "Eve", "content": "Good question. We moved the personalization questions into step 2 as optional. Users can skip and we'll infer preferences later."},
            {"role": "Grace", "content": "The A/B test showed the 3-step flow has 72% completion vs 58% for the old flow. That's a big improvement."},
            {"role": "Frank", "content": "One concern: the mobile layout needs work. The profile setup form overflows on iPhone SE screens."},
            {"role": "Alice", "content": "Let's fix that before the beta launch. Can we get a responsive fix by Friday?"},
            {"role": "Eve", "content": "I'll have it ready. Also, we decided to use the teal color scheme for the onboarding, matching our spring campaign."},
        ],
    },
    "bug-triage-mar30": {
        "title": "Bug Triage - March 30",
        "turns": [
            {"role": "Carol", "content": "We have 23 open bugs. Let's triage the critical ones first."},
            {"role": "Bob", "content": "Bug 847 is the highest priority: users can't upload files larger than 10MB. It's blocking enterprise customers."},
            {"role": "Dave", "content": "That's a server-side issue. The Nginx config has a hard limit. I'll fix it today and bump it to 100MB."},
            {"role": "Carol", "content": "Bug 851: the search results are showing deleted items. It's a cache invalidation problem."},
            {"role": "Grace", "content": "I can fix the cache invalidation. We need to add a webhook that purges the search index when items are deleted."},
            {"role": "Alice", "content": "Bug 849: dark mode is broken on the settings page. Low priority but easy fix."},
            {"role": "Bob", "content": "I'll take that one. It's just missing CSS variables. Ten minute fix."},
            {"role": "Carol", "content": "Summary: Dave fixes upload limit today, Grace handles cache bug by Thursday, Bob fixes dark mode CSS this week."},
        ],
    },
}


def seed_meetings(kb: KnowledgeBase):
    """Seed the knowledge base with mock meeting data."""
    print("=== Seeding mock meeting data ===\n")
    for session_id, meeting in MEETINGS.items():
        kb.add_meeting_transcript(
            turns=meeting["turns"],
            session_id=session_id,
            meeting_title=meeting["title"],
        )
        print(f"  Added: {meeting['title']} ({len(meeting['turns'])} turns)")
    print()


def run_demo(kb: KnowledgeBase):
    """Interactive demo: ask questions, get RAG-powered answers."""
    print("=== Knowledge Base RAG Demo ===")
    print(f"Stats: {kb.stats()}\n")

    demo_questions = [
        "What is the deadline for the public beta launch?",
        "Who is working on the authentication refactor?",
        "What bugs were discussed in the last triage?",
        "How many steps is the new onboarding flow?",
        "What is the target load time for the dashboard?",
    ]

    for i, question in enumerate(demo_questions, 1):
        print(f"--- Q{i}: {question} ---")
        answer = kb.query(question, max_tokens=200)
        print(f"A: {answer}\n")

    # Interactive mode
    print("=== Interactive Mode (type 'quit' to exit) ===")
    while True:
        try:
            question = input("\nYou: ").strip()
            if not question or question.lower() in ("quit", "exit", "q"):
                break
            answer = kb.query(question, max_tokens=300)
            print(f"Assistant: {answer}")
        except (KeyboardInterrupt, EOFError):
            break
    print("\nBye!")


if __name__ == "__main__":
    import sys

    kb = KnowledgeBase()

    # Check LLM availability
    import httpx
    try:
        resp = httpx.get(
            "http://127.0.0.1:12345/v1/models",
            headers={"Authorization": "Bearer 1234"},
            timeout=5.0,
        )
        models = [m["id"] for m in resp.json().get("data", [])]
        print(f"LLM available, models: {models}\n")
    except Exception as e:
        print(f"ERROR: LLM not available at http://127.0.0.1:12345 - {e}")
        print("Start omlx first, then rerun this demo.")
        sys.exit(1)

    # Reset and seed
    kb.clear_all()
    seed_meetings(kb)

    # Run demo
    run_demo(kb)
