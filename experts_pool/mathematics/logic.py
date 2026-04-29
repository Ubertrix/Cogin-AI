import numpy as np
import re
import random
from experts_pool.base_expert import BaseExpert
from experts_pool.mathematics.knowledge import MATH_DATA
from kernel.shape_guard import shape_guard
from kernel.ops.alu import evaluate_arithmetic_expression

class MathematicsExpert(BaseExpert):
    """Advanced Mathematics Expert: Solves Algebra, Calculus, and Arithmetic operations"""
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__(name="Math", input_dim=input_dim, output_dim=output_dim)
        self.knowledge = MATH_DATA
        print(f"{self.name} Expert: Algebraic & Symbolic Logic Active.")

    def _solve_calculus(self, text):
        """Solve simple calculus problems"""
        text = text.lower()
        deriv_match = re.search(r'(derivative|diff|اشتق|تفاضل)\s*(of|لـ)?\s*x\^?(\d*)', text)
        if deriv_match:
            power = deriv_match.group(3)
            power = int(power) if power != "" else 1
            if power == 0: return "0"
            if power == 1: return "1"
            return f"{power}x^{power-1}" if power-1 > 1 else f"{power}x"

        integ_match = re.search(r'(integral|integ|تكامل)\s*(of|لـ)?\s*x\^?(\d*)', text)
        if integ_match:
            power = integ_match.group(3)
            power = int(power) if power != "" else 1
            return f"(x^{power+1})/{power+1} + C"
        return None

    def _solve_algebra(self, text):
        """Algebraic equation solver engine (x^2=C, x+A=B)"""
        text = text.replace(" ", "").lower()
        
        # 1. Solve x^2 = C equations
        sq_match = re.search(r'x\^2=(\d+)', text)
        if sq_match:
            val = float(sq_match.group(1))
            res = np.sqrt(val)
            if res == int(res): res = int(res)
            return f"x = {res} or x = -{res}" if res != 0 else "x = 0"

        # 2. Solve x + A = B or x - A = B equations
        lin_match = re.search(r'x([+\-])(\d+)=(\d+)', text)
        if lin_match:
            op = lin_match.group(1)
            a = float(lin_match.group(2))
            b = float(lin_match.group(3))
            res = b - a if op == '+' else b + a
            if res == int(res): res = int(res)
            return f"x = {res}"

        return None

    def _human_respond(self, result, category, is_ar):
        templates_en = [
            f"The solution for your {category} is {result}.",
            f"I found the answer! {result}.",
            f"Mathematically speaking, {result}."
        ]
        templates_ar = [
            f"حل مسألة {category} هو {result}.",
            f"لقد وجدت الحل! {result}.",
            f"من الناحية الرياضية، الإجابة هي {result}."
        ]
        return random.choice(templates_ar if is_ar else templates_en)

    @shape_guard
    def process(self, x, input_text=""):
        activity, _ = super().process(x)
        is_ar = any(ord(c) > 128 for c in input_text)
        
        # 1. Calculus
        calc_res = self._solve_calculus(input_text)
        if calc_res: return activity, self._human_respond(calc_res, "calculus", is_ar)

        # 2. Algebra
        alg_res = self._solve_algebra(input_text)
        if alg_res: return activity, self._human_respond(alg_res, "algebra", is_ar)

        # 3. Arithmetic calculations
        try:
            calc_expr = re.sub(r'[^\d+\-*/().^]', '', input_text)
            calc_expr = calc_expr.replace(' ', '')
            if calc_expr and any(c in calc_expr for c in "+-*/") and re.search(r'\d', calc_expr):
                result = evaluate_arithmetic_expression(calc_expr)
                return activity, self._human_respond(result, "arithmetic", is_ar)
        except Exception:
            pass
            
        # 4. Conceptual Fallback
        category = "general"
        if "algebra" in input_text.lower() or "جبر" in input_text.lower(): category = "algebra"
        if category in self.knowledge:
            rule = random.choice(self.knowledge[category]["ar" if is_ar else "en"])
            return activity, f"{'Info' if not is_ar else 'معلومة'}: {rule}"

        return activity, None
