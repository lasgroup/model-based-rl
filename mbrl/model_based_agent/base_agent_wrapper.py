class BaseAgentWrapper:
    def __init__(self, agent, *args, **kwargs):
        self.agent = agent(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.agent, name)