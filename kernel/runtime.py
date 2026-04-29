import time

class RuntimeEngine:
    """Execution engine that manages data flow between all system modules"""
    def __init__(self, memory_module, router_module):
        self.memory = memory_module
        self.router = router_module
        print("Runtime Engine Initialized.")

    def execute_pipeline(self, user_input, available_experts):
        start_time = time.time()
        
        # 1. Fetch context from short-term memory
        context = self.memory.get_recent_context()
        print(f"Memory Context Loaded: {len(context)} items")

        # 2. Routing (Selecting expert based on input)
        # (Assuming input is already tokenized via dictionary)
        selected_expert, confidence = self.router.route(user_input, available_experts)
        print(f"Routed to: [{selected_expert.name}] (Confidence: {confidence:.2f})")

        # 3. Processing
        result = selected_expert.process(user_input)

        # 4. Save result to memory
        self.memory.add_interaction(user_input, "Success")

        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
        
        return result
