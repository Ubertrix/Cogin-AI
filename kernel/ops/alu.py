import ast
import operator


def alu_divide(left, right):
    """ALU Shard division primitive: always use true division for decimals."""
    return left / right


def _eval_ast(node):
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return operator.add(left, right)
        if isinstance(node.op, ast.Sub):
            return operator.sub(left, right)
        if isinstance(node.op, ast.Mult):
            return operator.mul(left, right)
        if isinstance(node.op, ast.Div):
            return alu_divide(left, right)
        if isinstance(node.op, ast.Pow):
            return operator.pow(left, right)
        if isinstance(node.op, ast.Mod):
            return operator.mod(left, right)
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):
        return node.n

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def evaluate_arithmetic_expression(expression):
    """Evaluate a sanitized arithmetic expression using the ALU Shard for division."""
    expression = expression.replace('^', '**')
    node = ast.parse(expression, mode='eval')
    return _eval_ast(node.body)
