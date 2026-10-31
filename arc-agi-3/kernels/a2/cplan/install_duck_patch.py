"""Install only checked-plan support into the PROVIDED Duck V23 source layout.

Run after applying the user's duck_harness_patch.diff, BEFORE benchmark import.
Default writes a reviewable diff; --apply additionally edits two source files.
Fail closed if anchors changed. Never touches model/serving or other guards.
"""
import argparse, ast, difflib, inspect, textwrap
from pathlib import Path
import transition_contracts as tc

MARKER = '# ASTRA_CHECKED_PLAN_V1'
GUIDANCE = ('- `checked_plan(actions, expectations, observe)` executes one action at a time. '
            'Provide one nonempty dict of predicted observable features per action and a '
            'zero-argument observer returning those features from the CURRENT frame. '
            'It stops on a mismatch, refusal or terminal transition and returns a journal. '
            'Inspect its status and replan; do not execute the old suffix. '
            'Keep entity type, relative pattern, size and orientation in hypotheses; '
            'a successful level does not validate all statements in your rule.\n')


def block():
    source=inspect.getsource(tc.compare)+'\n'+inspect.getsource(tc.checked_plan)
    source=source.replace('compare(', '_astra_compare(').replace('checked_plan(', '_astra_checked_plan_core(')
    source += '''
def checked_plan(actions, expectations, observe):
    return _astra_checked_plan_core(actions, expectations,
                                   lambda item: action([item]), observe)
runtime_globals["checked_plan"] = checked_plan
'''
    # The target embeds this block inside an r-triple-double-quoted bootstrap.
    source=source.replace('"""', "'''")
    return textwrap.indent(MARKER+'\n'+source,'        ')


def patch_text(sandbox, prompts):
    if MARKER in sandbox:raise ValueError('already installed; no double wrapping')
    anchor='        runtime_globals["action"] = action\n'
    prompt_anchor='STEP_VERIFICATION_ADDENDUM = (\n'
    if sandbox.count(anchor)!=1 or prompts.count(prompt_anchor)!=1:
        raise ValueError('source version mismatch: expected one verified V23 anchor in each file')
    candidate=sandbox.replace(anchor,anchor+block())
    new_prompts=prompts.replace(prompt_anchor,prompt_anchor+'    '+repr(GUIDANCE)+'\n')
    ast.parse(candidate);ast.parse(new_prompts)
    return candidate,new_prompts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--apply',action='store_true');ap.add_argument('--diff',default='duck_agent_patch.diff')
    args=ap.parse_args();root=Path(args.root)
    files=[root/'inference/agent/python_tool_sandbox.py',root/'inference/agent/prompts.py']
    old=[p.read_text() for p in files];new=patch_text(*old)
    patch=''
    for p,a,b in zip(files,old,new):
        rel=p.relative_to(root)
        patch+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+str(rel),tofile='b/'+str(rel)))
    Path(args.diff).write_text(patch)
    if args.apply:
        # All anchors and Python syntax have been checked before either write.
        for p,s in zip(files,new):p.write_text(s)
    print('applied' if args.apply else 'diff only',args.diff)


if __name__=='__main__':main()
