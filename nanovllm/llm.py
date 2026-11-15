from nanovllm.engine.llm_engine import LLMEngine


# 对外的 LLM 接口直接复用引擎实现。
class LLM(LLMEngine):
    pass
