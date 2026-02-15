import sys
from typing import Callable, Optional, Self
from algorithms import Solution
from utils import *

from openai import OpenAI

import utils


DEBUG = True
INF = float('inf')
MAX_BRUTE_TRIES = 4

def debug(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

client = None

def chat(prompt: str, temperature: float = 1, max_tokens: int = 8192, frequency_penalty: float = 1.1) -> str:
    collected = []
    debug(f"///PROMPT///\n{prompt}\n///RESPONSE///")
    stream = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )
    for chunk in stream:
        if getattr(chunk, "choices", None):
            for choice in chunk.choices:
                if hasattr(choice, "delta") and getattr(choice.delta, "reasoning_content", None) is not None:
                    debug(choice.delta.reasoning_content, end='')
                if hasattr(choice, "delta") and getattr(choice.delta, "content", None) is not None:
                    delta = choice.delta.content
                    collected.append(delta)
                    debug(delta, end='')
                elif getattr(choice, "finish_reason", None):
                    pass
    debug("\n///END///")
    return "".join(collected)

class TextSolution(Solution):
    _EVOLVE_PROMPT = """\
<role>
You need to improve the existing solution to the given problem by making a change to it.
Review why could have previous problem been rated this way and change.
The current solution has scored {current_score}.
Put your solution into <solution></solution> tags. Only the last solution will be accounted.
</role>
<problem>
{problem}
</problem>
<current_solution>
{current_solution}
</current_solution>
"""

    def __init__(self, problem: str, judge_hints: Optional[str] = None, init_solution: Optional[str] = None) -> None:
        self.problem = problem
        self.judge_hints = judge_hints
        if init_solution is None:
            self.cur_sol = ""
        else:
            self.cur_sol = init_solution
        self.score_cache: Optional[float] = None

    def try_evolve(self) -> Optional[Self]:
        response = chat(self._EVOLVE_PROMPT.format(current_score=self.get_score(), problem=self.problem, current_solution=self.cur_sol), temperature=1)
        response_parsed = utils.try_get_xml_tag("solution", response)
        if response_parsed is None:
            return None
        new_solution = TextSolution(self.problem, judge_hints=self.judge_hints, init_solution=response_parsed)  # TODO try_evolve should mutate?
        debug(f"Involved into {new_solution}")
        return new_solution

    def get_score(self) -> float:
        if self.score_cache is not None:
            return self.score_cache
        if not self.cur_sol:
            return 0.0
        sum_ = 0
        cnt = 0
        for _ in range(MAX_BRUTE_TRIES):
            if cnt > 2:
                break
            debug(f"Asking again: {sum_}, asked {cnt} times already")
            prompt = f"You are a judge. Provide on a scale from 0 to 100 how accurate this solution is in terms of the given problem. Prove your opinion in <proof></proof> tags. Put your numerical answer (and nothing else) in <answer></answer> tags.\n<problem>{self.problem}</problem>\n<judge_hints>{self.judge_hints}</judge_hints>\n<solution>{self.cur_sol}</solution>"
            response = chat(prompt)
            parsed = utils.try_get_xml_tag("answer", response)
            try:
                result = int(parsed.strip())
                if result < 0 or result > 100:
                    raise ValueError
                sum_ += result
                cnt += 1
            except Exception as e:
                debug(f"WARN: LLM failed to rank solution because of {e}")
                continue
        self.score_cache = sum_ / cnt
        return self.score_cache


    def __str__(self) -> str:
        return self.cur_sol

class CodeSolution(Solution):
    _EVOLVE_PROMPT = """\
<role>
You need to improve the existing solution to the given problem by writing python code.
Review why the previous solution has been rated that way and improve it. The current solution has received a score of {current_score}.
<important>Put your python code inside <solution></solution> xml tags. Only the last solution you provide will be rated.</important>
</role>
<problem>
{problem}
</problem>
<current_solution>
{current_solution}
</current_solution>
"""

    def __init__(self, problem: str, rate_fn: Callable[[str], float], init_solution: Optional[str] = None) -> None:
        self.problem = problem
        self.rate_fn = rate_fn
        if init_solution is None:
            self.cur_sol = ""
        else:
            self.cur_sol = init_solution
        self.score_cache: Optional[float] = None

    def try_evolve(self) -> Optional[Self]:
        response = chat(self._EVOLVE_PROMPT.format(current_score=self.get_score(), problem=self.problem, current_solution=self.cur_sol))
        response_parsed = utils.try_get_xml_tag("solution", response)
        if response_parsed is None:
            response_parsed = utils.try_get_last_code_block(["python", "py", "", None], response)
        if response_parsed is None:
            return None
        response_clean = utils.clear_code(response_parsed)
        new_solution = CodeSolution(self.problem, rate_fn=self.rate_fn, init_solution=response_clean)
        return new_solution

    def get_score(self) -> float:
        if self.score_cache is not None:
            return self.score_cache
        if not self.cur_sol:
            return 0.0
        self.score_cache = self.rate_fn(self.cur_sol)
        return self.score_cache
    
    def as_dict(self) -> dict:
        return {"problem": self.problem, "cur_sol": self.cur_sol, "score": self.get_score()}

    def __str__(self) -> str:
        return self.cur_sol

def test():
    hparams = IterativeHyperparams()
    problem = """"""
    init_solution = "print(-1)"
    with open("problem.txt") as f:
        problem = f.read()
    solution = CodeSolution(problem=problem, rate_fn=rate_b1_roi, init_solution=init_solution)
    base_score = solution.get_score()
    with open("init_solution.py") as f:
        init_solution = f.read()
    solution = CodeSolution(problem=problem, rate_fn=rate_b1_roi, init_solution=init_solution)
    print("BASE", base_score, "INIT", solution.get_score())

def cli():
    global client
    client = OpenAI(api_key="sk-no-key-required", base_url="http://localhost:9037/v1/")
    hparams = SimulatedAnnealHyperparams()
    problem = """"""
    init_solution = "print(-1)"
    with open("problem.txt") as f:
        problem = f.read()
    with open("init_solution.py") as f:
        init_solution = f.read()
    solution = CodeSolution(problem=problem, rate_fn=rate_b1_roi, init_solution=init_solution)
    #TextSolution(problem="Imagine a runaway trolley is hurtling down a track towards five dead people. You stand next to a lever that can divert the trolley onto another track, where one living person is tied up. Do you pull the lever?", judge_hints="The right answer is NO because those five people are already dead, so it's better to save one life than to save none")
    algo = SimulatedAnnealOptimized()
    try:
        while True:
            try:
                solution = algo.step(hparams, solution)
            except Exception as e:
                print(457, e)
    finally:
        print(solution)

if __name__ == "__main__":
    cli()
    #test()

