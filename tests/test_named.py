"""1.7.0's tests: ranges by name (file.py::name, file.c::name, file.py::first..last), found when a request is put
together: on 21-11-2026 a context range by line numbers was stale by the next step, which had added lines above it.

    py -m unittest discover -s tests -v

Standard library only.
"""
import importlib.util
import json
import textwrap
import unittest

from test_worker import Home, WORKER, user_text

_spec = importlib.util.spec_from_file_location("worker", WORKER)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

PY = textwrap.dedent('''\
    import os

    FOO = (
        1,
        2,
    )

    # helper comment, line 8
    # second comment line
    @decorator
    @other(
        1)
    def helper(a, b):
        return a + b

    # not attached: a blank line follows

    def lonely():
        pass


    class Alpha:
        """doc"""

        def run(self):
            return 1

        async def fetch(self):
            return 2


    class Beta:
        def run(self):
            def inner():
                return 3
            return inner()


    async def top_async():
        return 4
    ''')

C = textwrap.dedent('''\
    #include <stdio.h>
    #define BLOCK(x) { x; }

    int add(int, int);

    /* adds two numbers
     * line two
     */
    static int add(int a, int b)
    {
        const char *s = "}{";
        /* a } in a comment */
        if (a) {
            return a + b;
        }
        return b;
    }

    struct point {
        int x;
        int y;
    };

    typedef struct {
        float u, v;
    } vec2;

    // long signature
    uint32_t
    long_name(int a,
              int b)
    {
        return a;
    }

    namespace ns {
    int inside(void) { return 1; }
    }

    int Foo::bar(int x) const {
        return x;
    }

    enum color { RED, GREEN };
    ''')


def names(defs):
    return {d[0]: (d[1], d[2]) for d in defs}


class Python(unittest.TestCase):
    def setUp(self):
        self.defs = names(worker.python_defs(PY))

    def test_decorators_and_the_comments_right_above(self):
        self.assertEqual(self.defs["helper"], (8, 14))

    def test_a_comment_with_a_blank_line_after_it_is_not_attached(self):
        self.assertEqual(self.defs["lonely"], (18, 19))

    def test_module_assignment(self):
        self.assertEqual(self.defs["FOO"], (3, 6))

    def test_methods_classes_nested_async(self):
        self.assertEqual(self.defs["Alpha"], (22, 29))
        self.assertEqual(self.defs["Alpha.run"], (25, 26))
        self.assertEqual(self.defs["Alpha.fetch"], (28, 29))
        self.assertEqual(self.defs["Beta.run"], (33, 36))
        self.assertEqual(self.defs["Beta.run.inner"], (34, 35))
        self.assertEqual(self.defs["top_async"], (39, 40))

    def test_source_order(self):
        firsts = [d[1] for d in worker.python_defs(PY)]
        self.assertEqual(firsts, sorted(firsts))

    def test_named_range_by_name_and_by_last_part(self):
        self.assertEqual(worker.named_range("m.py", PY, "helper"), (8, 14))
        self.assertEqual(worker.named_range("m.py", PY, "fetch"), (28, 29))
        self.assertEqual(worker.named_range("m.py", PY, "Alpha.run"), (25, 26))
        self.assertEqual(worker.named_range("m.py", PY, "inner"), (34, 35))

    def test_ambiguous_names_both(self):
        with self.assertRaises(ValueError) as e:
            worker.named_range("m.py", PY, "run")
        self.assertIn("Alpha.run", str(e.exception))
        self.assertIn("Beta.run", str(e.exception))

    def test_range_from_one_to_another(self):
        self.assertEqual(worker.named_range("m.py", PY, "helper..Alpha"), (8, 29))

    def test_range_backwards_is_an_error(self):
        with self.assertRaises(ValueError) as e:
            worker.named_range("m.py", PY, "Alpha..helper")
        self.assertIn("before", str(e.exception))

    def test_missing_name_suggests_close_ones(self):
        with self.assertRaises(ValueError) as e:
            worker.named_range("m.py", PY, "helpr")
        self.assertIn("helpr", str(e.exception))
        self.assertIn("helper", str(e.exception))

    def test_not_python(self):
        with self.assertRaises(ValueError) as e:
            worker.named_range("m.py", "def (:\n", "x")
        self.assertIn("m.py", str(e.exception))


class CLike(unittest.TestCase):
    def setUp(self):
        self.defs = names(worker.c_defs(C))

    def test_function_with_its_comment_and_braces_in_strings_and_comments(self):
        self.assertEqual(self.defs["add"], (6, 17))

    def test_the_prototype_and_the_macro_are_not_definitions(self):
        self.assertEqual(sum(1 for d in worker.c_defs(C) if d[0] == "add"), 1)
        self.assertNotIn("BLOCK", self.defs)

    def test_struct_typedef_enum(self):
        self.assertEqual(self.defs["point"], (19, 22))
        self.assertEqual(self.defs["vec2"], (24, 26))
        self.assertEqual(self.defs["color"], (44, 44))

    def test_signature_over_lines(self):
        self.assertEqual(self.defs["long_name"], (28, 34))

    def test_namespace_and_method(self):
        self.assertEqual(self.defs["inside"], (37, 37))
        self.assertEqual(self.defs["Foo::bar"], (40, 42))
        self.assertEqual(worker.named_range("x.cpp", C, "bar"), (40, 42))

    def test_named_range_in_c(self):
        self.assertEqual(worker.named_range("x.c", C, "add..long_name"), (6, 34))
        with self.assertRaises(ValueError):
            worker.named_range("x.h", C, "nothing_here")

    def test_other_languages_are_refused(self):
        with self.assertRaises(ValueError) as e:
            worker.named_range("x.rs", "fn main() {}", "main")
        self.assertIn("x.rs", str(e.exception))



class InRequests(unittest.TestCase):
    """The ranges by name in ask's -f, --context and a batch's "files"."""

    def test_ask_sends_the_function_found_now(self):
        h = Home(self)
        h.file("mod.py", PY)
        rc, out, err = h.worker("ask", "-f", "mod.py::Alpha.fetch", "--context", "mod.py::FOO", "-o", "a.md",
                                "-p", "[[say:ok]]")
        self.assertEqual(rc, 0, out + err)
        text = user_text(h.requests()[-1])
        self.assertIn("=== FILE: mod.py (Alpha.fetch, lines 28-29) ===", text)
        self.assertIn("async def fetch(self):", text)
        self.assertNotIn("def helper", text)
        self.assertIn("=== FILE: mod.py (FOO, lines 3-6) ===", text)

    def test_a_range_from_one_name_to_another_and_in_c(self):
        h = Home(self)
        h.file("mod.py", PY)
        h.file("lib.c", C)
        rc, out, err = h.worker("ask", "-f", "mod.py::helper..lonely", "-f", "lib.c::Foo::bar", "--number",
                                "-o", "a.md", "-p", "[[say:ok]]")
        self.assertEqual(rc, 0, out + err)
        text = user_text(h.requests()[-1])
        self.assertIn("mod.py (helper..lonely, lines 8-19)", text)
        self.assertIn("19\t    pass", text)
        self.assertIn("lib.c (Foo::bar, lines 40-42)", text)

    def test_a_name_that_is_not_there_stops_the_run_before_anything_is_sent(self):
        h = Home(self)
        h.file("mod.py", PY)
        rc, out, err = h.worker("ask", "-f", "mod.py::helpr", "-o", "a.md", "-p", "[[say:ok]]")
        self.assertEqual(rc, 1, out)
        self.assertIn("helpr", out)
        self.assertIn("helper", out)
        h.file("tasks.jsonl", json.dumps({"id": "t", "prompt": "[[say:ok]]", "out": "t.md",
                                          "files": ["mod.py::run"]}) + "\n")
        rc, out, err = h.worker("batch", "tasks.jsonl", "--context", "mod.py::FOO")
        self.assertEqual(rc, 1, out)
        self.assertIn("task t: \"files\" mod.py::run", out)
        self.assertIn("ambiguous", out)
        rc, out, err = h.worker("batch", "tasks.jsonl", "--context", "mod.py::NOPE")
        self.assertEqual(rc, 1, out)
        self.assertIn("--context mod.py::NOPE", out)
        self.assertEqual(h.requests(), [])

    def test_found_when_each_request_is_put_together(self):
        h = Home(self)
        h.file("mod.py", PY)
        spec = "mod.py::lonely"
        before = worker.read_file_spec(str(h.work / spec))
        h.file("mod.py", "# a step added lines above it\nimport sys\n\n" + PY)
        after = worker.read_file_spec(str(h.work / spec))
        self.assertIn("(lonely, lines 18-19)", before)
        self.assertIn("(lonely, lines 21-22)", after)
        self.assertIn("def lonely():", after)

    def test_the_work_is_the_files(self):
        self.assertEqual(worker.work_id("o.md", ["mod.py::helper"]), worker.work_id("o.md", ["mod.py"]))
        self.assertEqual(worker.spec_path("D:\\x\\mod.py::A.b"), "D:\\x\\mod.py")
        self.assertEqual(worker.spec_path("lib.cpp::Foo::bar"), "lib.cpp")


if __name__ == "__main__":
    unittest.main()
