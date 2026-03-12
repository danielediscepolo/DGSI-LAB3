import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from openai import OpenAI
from sympy import Eq, factor, lambdify, solve, sympify
from sympy.abc import x as default_x


load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4.1-mini")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
PLOTS_DIR = Path(__file__).parent / "plots"
LOGS_DIR = Path(__file__).parent / "logs"

SYSTEM_PROMPT = """
You are a careful high-school math tutor.

Rules:
- Solve step by step in concise and clear language.
- Prefer calling tools for calculations, equation solving, factoring, and plotting.
- If a request is outside available tools, explain limitations clearly.
- If user asks for a graph, call plot_function.
- If expression is invalid, ask for a corrected version.
""".strip()

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_expression",
            "description": "Evaluate or simplify a math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expression like (3/4 + 2/3) * 6 or 2^5 - 3"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "Solve one-variable equations such as 2*x + 5 = 17.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "Equation including '=' symbol"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable name, usually x",
                        "default": "x"
                    }
                },
                "required": ["equation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "factor_expression",
            "description": "Factor an algebraic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expression like x^2 + 7*x + 12"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_function",
            "description": "Plot y = f(x) into a PNG file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Function of x, like x^2 - 4*x + 3"
                    },
                    "x_min": {
                        "type": "number",
                        "description": "Minimum x value"
                    },
                    "x_max": {
                        "type": "number",
                        "description": "Maximum x value"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Optional output file name ending in .png"
                    }
                },
                "required": ["expression", "x_min", "x_max"]
            }
        }
    },
]


def _json_result(ok: bool, data: Any = None, error: str | None = None) -> str:
    payload = {"ok": ok, "data": data, "error": error}
    return json.dumps(payload, ensure_ascii=True)


def evaluate_expression(expression: str) -> str:
    try:
        expr = sympify(expression.replace("^", "**"))
        simplified = expr.simplify()
        numeric = simplified.evalf()
        return _json_result(True, {"expression": str(expr), "simplified": str(simplified), "numeric": str(numeric)})
    except Exception as exc:
        return _json_result(False, error=f"Cannot evaluate expression: {exc}")


def solve_equation(equation: str, variable: str = "x") -> str:
    try:
        var = sympify(variable)
        if "=" not in equation:
            return _json_result(False, error="Equation must include '=' symbol")
        left, right = equation.split("=", 1)
        eq = Eq(sympify(left.replace("^", "**")), sympify(right.replace("^", "**")))
        solutions = solve(eq, var)
        return _json_result(True, {"equation": str(eq), "variable": str(var), "solutions": [str(s) for s in solutions]})
    except Exception as exc:
        return _json_result(False, error=f"Cannot solve equation: {exc}")


def factor_expression(expression: str) -> str:
    try:
        expr = sympify(expression.replace("^", "**"))
        factored = factor(expr)
        return _json_result(True, {"original": str(expr), "factored": str(factored)})
    except Exception as exc:
        return _json_result(False, error=f"Cannot factor expression: {exc}")


def plot_function(expression: str, x_min: float, x_max: float, output_file: str | None = None) -> str:
    try:
        if x_min >= x_max:
            return _json_result(False, error="x_min must be smaller than x_max")

        expr = sympify(expression.replace("^", "**"))
        func = lambdify(default_x, expr, modules=["math"])

        try:
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            plot_dir = PLOTS_DIR
        except OSError:
            plot_dir = Path("/tmp/math_solver_plots")
            plot_dir.mkdir(parents=True, exist_ok=True)

        if output_file:
            file_name = output_file
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_expr = "".join(c for c in expression if c.isalnum() or c in ("_", "-"))[:24] or "plot"
            file_name = f"{safe_expr}_{ts}.png"

        output_path = plot_dir / file_name

        steps = 300
        xs = [x_min + (x_max - x_min) * i / (steps - 1) for i in range(steps)]
        ys_raw = [func(v) for v in xs]
        valid_points: list[tuple[float, float]] = []
        for xv, yv in zip(xs, ys_raw):
            try:
                y_float = float(yv)
            except (TypeError, ValueError):
                continue
            if math.isfinite(y_float):
                valid_points.append((xv, y_float))

        if len(valid_points) < 2:
            return _json_result(False, error="Cannot plot function: expression produced too few valid points in range")

        plot_x = [p[0] for p in valid_points]
        plot_y = [p[1] for p in valid_points]

        plt.figure(figsize=(8, 5), facecolor="white")
        plt.plot(plot_x, plot_y, label=f"y = {expression}", color="#1f77b4", linewidth=2)
        plt.axhline(0, linewidth=1, color="#444444")
        plt.axvline(0, linewidth=1, color="#444444")
        plt.grid(True, alpha=0.3, color="#bbbbbb")
        plt.legend()
        plt.title("Function Plot")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(min(plot_x), max(plot_x))
        plt.tight_layout()
        try:
            plt.savefig(output_path)
        except OSError:
            fallback_dir = Path("/tmp/math_solver_plots")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            output_path = fallback_dir / file_name
            plt.savefig(output_path)
        finally:
            plt.close()

        return _json_result(
            True,
            {
                "expression": str(expr),
                "x_min": x_min,
                "x_max": x_max,
                "file": str(output_path),
            },
        )
    except Exception as exc:
        return _json_result(False, error=f"Cannot plot function: {exc}")


TOOL_FUNCTIONS = {
    "evaluate_expression": evaluate_expression,
    "solve_equation": solve_equation,
    "factor_expression": factor_expression,
    "plot_function": plot_function,
}


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")

    if OPENAI_API_ENDPOINT:
        return OpenAI(api_key=api_key, base_url=OPENAI_API_ENDPOINT)
    return OpenAI(api_key=api_key)


def append_log(event: dict[str, Any]) -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / "math_solver_log.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
    except OSError:
        fallback_dir = Path("/tmp/math_solver_logs")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        with (fallback_dir / "math_solver_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")


def solve_with_tools(client: OpenAI, problem: str, show_tool_calls: bool = True) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    max_rounds = 4
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=AVAILABLE_TOOLS,
            temperature=0.2,
        )

        msg = response.choices[0].message
        append_log({"ts": datetime.now().isoformat(), "phase": "model_response", "content": msg.content, "tool_calls": bool(msg.tool_calls)})

        if not msg.tool_calls:
            final_answer = msg.content or "I could not produce an answer."
            append_log({"ts": datetime.now().isoformat(), "phase": "final_answer", "answer": final_answer})
            return final_answer

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            func = TOOL_FUNCTIONS.get(name)
            if func is None:
                tool_result = _json_result(False, error=f"Unknown tool: {name}")
            else:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_result = func(**args)

            if show_tool_calls:
                print(f"[tool] {name} args={tc.function.arguments}")
                print(f"[tool-result] {tool_result}")

            append_log(
                {
                    "ts": datetime.now().isoformat(),
                    "phase": "tool_execution",
                    "tool": name,
                    "arguments": tc.function.arguments,
                    "result": tool_result,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": tool_result,
                }
            )

    return "I reached the tool-calling round limit without a final response."


def interactive_chat() -> None:
    print("Math Solver (Function Calling)")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {OPENAI_API_ENDPOINT or 'https://api.openai.com/v1'}")
    print("Type a math problem, or 'q' to quit.\n")

    client = get_client()

    while True:
        user_input = input("Problem> ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        if not user_input:
            continue

        answer = solve_with_tools(client, user_input, show_tool_calls=True)
        print("\nFinal answer:")
        print(answer)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Math solver with function calling")
    parser.add_argument("--problem", type=str, help="Run a single problem and exit")
    parser.add_argument("--hide-tool-calls", action="store_true", help="Hide tool call details in terminal output")
    args = parser.parse_args()

    if args.problem:
        client = get_client()
        answer = solve_with_tools(client, args.problem, show_tool_calls=not args.hide_tool_calls)
        print(answer)
    else:
        interactive_chat()


if __name__ == "__main__":
    main()
