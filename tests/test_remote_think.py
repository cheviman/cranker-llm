"""1.7.0's tests: a server elsewhere that thinks past its answer (20-11-2026: three OutRunners tasks came back empty
after 64,000 to 200,000 tokens of thinking, with no RECOVERED). Against tests/fake_llama.py as the server elsewhere.

    py -m unittest discover -s tests -v

Standard library only.
"""
import json
import unittest

from test_worker import Home, check_for, user_text


class RemoteThinking(unittest.TestCase):
    KEY = {"CRANKER_REMOTE_KEY": "its-key"}

    def home(self):
        """No local server; the remote one with "plain" fields that think all the same, as the user's has"""
        h = Home(self, serve=False)
        h.save(remote={"url": h.elsewhere(), "model": "deepseek-there", "roots": [str(h.work)],
                       "plain": {"reasoning_effort": "medium"}})
        return h

    def test_an_answer_that_was_all_thinking_and_none_of_it_sent_is_asked_for_again_thinking_less(self):
        h = self.home()
        rc, out, err = h.worker("ask", "--remote", "--think", "--effort", "xhigh", "--max-tokens", "1000",
                                "-o", "a.md", "-p", "[[hidden-think:high]][[say:the answer]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "the answer")
        sent = h.requests(h.away)
        self.assertEqual([r["reasoning_effort"] for r in sent], ["high", "medium"])
        self.assertEqual([r["max_tokens"] for r in sent], [1000, 1000])  # the same room, not more of it
        self.assertIn("effort lowered xhigh -> medium", out)
        rec = h.log("usage.jsonl")[-1]
        self.assertEqual(rec["lowered"], "medium")
        self.assertGreaterEqual(rec["out"], 1000)  # the thinking that was thrown away is counted as spent

    def test_its_check_does_not_raise_the_room_of_an_answer_that_was_all_thinking(self):
        h = self.home()
        h.file("tasks.jsonl", json.dumps({"id": "t", "prompt": "[[hidden-think:medium,low]][[say:right]]",
                                          "out": "t.md", "think": True, "effort": "medium", "max_tokens": 1000}))
        rc, out, err = h.worker("batch", "tasks.jsonl", "--remote", "--check", check_for(h, "right"), **self.KEY)
        self.assertEqual(rc, 2, out + err)
        sent = h.requests(h.away)
        self.assertEqual([r["reasoning_effort"] for r in sent], ["medium", "low"])
        self.assertEqual({r["max_tokens"] for r in sent}, {1000})  # no x1.5: more room buys more thinking
        self.assertIn("with nothing written: all of it went to thinking, even with the effort lowered medium -> low",
                      out)
        self.assertIn("--effort low", out)
        self.assertEqual(json.loads((h.work / "t.md.failed.json").read_text(encoding="utf-8"))["why"], "cut")

    def test_a_plain_request_that_thinks_is_recovered_with_the_room_it_had(self):
        h = self.home()
        rc, out, err = h.worker("ask", "--remote", "--max-tokens", "9000", "-o", "a.md",
                                "-p", "[[think-text:I worked it out: it is 42]][[recover-room:8000]]", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual((h.work / "a.md").read_text(encoding="utf-8"), "recovered")
        self.assertIn("RECOVERED", out)
        first, again = h.requests(h.away)
        self.assertEqual(first["reasoning_effort"], "medium")  # no --think: the plain fields, which think
        self.assertEqual(again["max_tokens"], 9000)  # not 4096, which the plain fields would think away
        self.assertIn("I worked it out: it is 42", user_text(again))
        self.assertTrue(h.log("usage.jsonl")[-1]["recovered"])

    def test_the_reasoning_handed_back_is_its_end(self):
        h = self.home()
        long = "START " + "step " * 26000 + "so the answer is END"
        h.file("prompt.md", f"[[think-text:{long}]]")
        rc, out, err = h.worker("ask", "--remote", "--max-tokens", "9000", "-o", "a.md", "--prompt-file", "prompt.md",
                                "--keep-thinking", **self.KEY)
        self.assertEqual(rc, 0, out + err)
        handed = user_text(h.requests(h.away)[-1]).split("=== YOUR REASONING ON THIS")[1]  # the prompt holds it all
        self.assertNotIn("START", handed)
        self.assertIn("so the answer is END", handed)
        self.assertIn("its start left out", handed)
        self.assertIn("START", (h.work / "a.md.thinking.md").read_text(encoding="utf-8"))  # kept whole beside it


if __name__ == "__main__":
    unittest.main()
