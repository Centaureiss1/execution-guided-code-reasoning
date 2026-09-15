import unittest

from react_bench.config import RunnerConfig
from react_bench.providers.openai_compatible import OpenAICompatibleProvider, VLLMProvider


class ProviderTests(unittest.TestCase):
    def test_vllm_recovery_extra_body_disables_thinking(self):
        provider = VLLMProvider("model")
        body = provider._extra_body(RunnerConfig(recovery_disable_thinking=True), is_recovery=True)
        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(body["enable_thinking"], False)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["min_p"], 0.0)

    def test_vllm_normal_call_sends_vllm_sampling_extra_body(self):
        provider = VLLMProvider("model")
        body = provider._extra_body(RunnerConfig(recovery_disable_thinking=True), is_recovery=False)
        self.assertEqual(body, {"top_k": 20, "min_p": 0.0})

    def test_vllm_recovery_sampling_overrides_normal_sampling(self):
        provider = VLLMProvider("model")
        config = RunnerConfig(
            top_k=20,
            min_p=0.0,
            recovery_top_k=10,
            recovery_min_p=0.1,
            recovery_disable_thinking=False,
        )
        body = provider._extra_body(config, is_recovery=True)
        self.assertEqual(body, {"top_k": 10, "min_p": 0.1})

    def test_generic_openai_compatible_has_no_extra_body(self):
        provider = OpenAICompatibleProvider("model")
        body = provider._extra_body(RunnerConfig(recovery_disable_thinking=True), is_recovery=True)
        self.assertEqual(body, {})


if __name__ == "__main__":
    unittest.main()
