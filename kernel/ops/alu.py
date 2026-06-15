def evaluate_arithmetic_expression(expression):
    try:
        # Simple evaluation for benchmark purposes
        return str(eval(expression))
    except:
        return expression
