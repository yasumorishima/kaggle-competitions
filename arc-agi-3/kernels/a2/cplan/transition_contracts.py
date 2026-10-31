"""Observation-only planning and falsifiable transition evidence; no game knowledge."""
from collections import deque
from dataclasses import dataclass, field


def shortest_path(graph, start, goals, blocked=()):
    """Graph values are (action, destination). Build ALL nodes before edges."""
    goals, blocked = set(goals), set(blocked)
    queue = deque([start])
    parent = {start: None}
    while queue:
        node = queue.popleft()
        if node in goals:
            path = []
            while parent[node] is not None:
                prev, action = parent[node]
                path.append(action)
                node = prev
            return path[::-1]
        for action, nxt in graph.get(node, ()):
            if nxt in graph and nxt not in parent and nxt not in blocked:
                parent[nxt] = (node, action)
                queue.append(nxt)
    return None


def compare(expected, actual):
    """Exact subset contract. Missing fields are failures, never confirmations."""
    return {key: {'expected': value, 'actual': actual.get(key),
                  'missing': key not in actual}
            for key, value in expected.items()
            if key not in actual or actual[key] != value}


def checked_plan(actions, expectations, execute, observe, max_steps=64):
    """Execute one action, observe, verify; never execute a stale plan suffix.

    All expectations and observer output must be JSON-like scalar/list/dict data.
    Expectations are caller-supplied hypotheses, not guaranteed world models.
    Terminal transitions are reported separately, not counted as refutations.
    Exceptions from execute/observe propagate; no action is retried automatically.
    """
    actions, expectations = list(actions), list(expectations)
    if len(actions) != len(expectations) or len(actions) > max_steps:
        raise ValueError('one nonempty expectation per action; bounded plan required')
    if any(not isinstance(e, dict) or not e for e in expectations):
        raise ValueError('empty/non-dict contracts cannot verify a plan')
    journal = []
    for act, expected in zip(actions, expectations):
        result = execute(act) or {}
        if result.get('executed') is False:
            return {'status': 'refused', 'journal': journal, 'result': result}
        terminal = next((k for k in ('game_over', 'level_completed', 'done', 'run_complete')
                         if result.get(k)), None)
        if terminal:
            journal.append({'action': act, 'terminal': terminal})
            return {'status': terminal, 'journal': journal}
        actual = observe()
        if not isinstance(actual, dict):
            raise TypeError('observe() must return a dict')
        mismatch = compare(expected, actual)
        journal.append({'action': act, 'expected': expected, 'actual': actual,
                        'mismatch': mismatch})
        if mismatch:
            return {'status': 'prediction_mismatch', 'journal': journal}
    return {'status': 'verified', 'journal': journal}


@dataclass
class Evidence:
    """Independent transition IDs avoid counting copied reasoning as evidence."""
    observations: dict = field(default_factory=dict)

    def add(self, event_id, level, condition, predicted, observed):
        row = (level, condition, predicted, observed)
        if event_id in self.observations and self.observations[event_id] != row:
            raise ValueError('conflicting evidence for the same transition ID')
        self.observations[event_id] = row

    def status(self, condition, level):
        rows = [r for r in self.observations.values() if r[1] == condition]
        if not rows:
            return 'untested'
        if any(compare(r[2], r[3]) for r in rows):
            return 'refuted_or_context_missing'
        if not any(r[0] == level for r in rows):
            return 'transfer_requires_probe'
        # Repeated support is local evidence, not proof of causality/generalization.
        return 'supported_in_context'


def discriminating_probe(predictions, legal_actions, forbidden=()):
    """Select maximum outcome disagreement among caller-supplied hypotheses.

    predictions[action][hypothesis] is a hashable predicted outcome. An action
    is eligible only if every supplied hypothesis predicts its outcome and the
    caller has not marked it unsafe. This cannot certify unknown hazards.
    """
    forbidden = set(forbidden)
    models = set().union(*(set(v) for v in predictions.values())) if predictions else set()
    ranked = []
    for action in legal_actions:
        outcomes = predictions.get(action, {})
        if action in forbidden or not models or set(outcomes) != models:
            continue
        disagreement = len(set(outcomes.values()))
        if disagreement > 1:
            ranked.append((-disagreement, repr(action), action))
    return min(ranked)[2] if ranked else None


def robust_plan(start_states, actions, predict, is_goal, horizon=16, max_states=5000):
    """Bounded belief-state BFS over competing, observation-grounded models.

    A state is hashable and includes actor, objects, phase/queues as necessary.
    predict(model, state, action) returns next state or None if unsafe/unknown.
    All surviving models must allow each step. Unknown is NOT treated as safe.
    Caller must validate its models separately; a plan is conditional on them.
    Returns (actions or None, diagnostic). No game-specific transition rules.
    """
    models=tuple(sorted(start_states))
    initial=tuple(start_states[m] for m in models)
    if not models:return None,'no_models'
    q=deque([(initial,())]);seen={initial}
    while q:
        belief,path=q.popleft()
        if all(is_goal(m,s) for m,s in zip(models,belief)):
            return list(path),'goal_all_models'
        if len(path)>=horizon:continue
        for action in actions:
            nxt=tuple(predict(m,s,action) for m,s in zip(models,belief))
            if any(s is None for s in nxt) or nxt in seen:continue
            if len(seen)>=max_states:return None,'search_limit'
            seen.add(nxt);q.append((nxt,path+(action,)))
    return None,'no_safe_plan_within_horizon'
